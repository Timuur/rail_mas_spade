import asyncio
import json
import time
import os
import csv
import random
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import networkx as nx
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message

from utils.io_utils import load_json, dump_json

# ---------- Message "types" ----------
TYPE_ORDER_CFP = "order_cfp"
TYPE_ORDER_PROPOSE = "order_propose"
TYPE_ORDER_ACCEPT = "order_accept"
TYPE_ORDER_REJECT = "order_reject"
TYPE_ORDER_INFORM = "order_inform"

TYPE_WAGON_CFP = "wagon_cfp"
TYPE_WAGON_PROPOSE = "wagon_propose"

TYPE_HELLO_PING = "hello_ping"
TYPE_HELLO_PONG = "hello_pong"
TYPE_HELLO_START = "hello_start"


# ---------- Graph helpers ----------
def build_graph(edges: List[dict]) -> nx.Graph:
    g = nx.Graph()
    for e in edges:
        u, v = e["u"], e["v"]
        g.add_edge(
            u,
            v,
            distance_km=float(e["distance_km"]),
            speed_kmh=float(e.get("speed_kmh", 60)),
        )
    return g


def path_stats(g: nx.Graph, path: List[str]) -> Tuple[float, float]:
    d = t = 0.0
    for i in range(len(path) - 1):
        data = g.get_edge_data(path[i], path[i + 1])
        dist = float(data["distance_km"])
        speed = float(data.get("speed_kmh", 60))
        d += dist
        t += dist / speed
    return round(d, 2), round(t, 2)


def shortest_path(g: nx.Graph, a: str, b: str) -> List[str]:
    return nx.shortest_path(g, a, b, weight=lambda u, v, d: d["distance_km"])


class StationAgent(Agent):
    def __init__(
        self,
        jid,
        password,
        station_id: str,
        data_path: str,
        is_last: bool = False,
        **kwargs,
    ):
        super().__init__(jid, password, **kwargs)
        self.station_id = station_id
        self.data_path = data_path
        self.is_last = is_last

        # Data
        self.data: Dict[str, Any] = {}
        self.graph: Optional[nx.Graph] = None
        self.stations: List[dict] = []
        self.orders: List[dict] = []

        # Runtime
        self.offers_box: Dict[str, List[dict]] = {}      # order_id -> offers (по составам)
        self.wagon_offers: Dict[str, List[dict]] = {}    # order_id -> wagon-offers (аренда)
        self.output_dir = "output"
        self._written_ids: set[tuple[str, str]] = set()

        # Calendars / params
        self.turnaround_h: float = 0.5
        self.wagon_repos_cost_per_km: float = 0.5
        self.free_at_loco: Dict[str, float] = {}
        self.free_at_wagon: Dict[str, float] = {}

        # Idle / sync
        self.last_active: float = time.time()
        # единая очередь управления только на агенте
        self.bus: asyncio.Queue = asyncio.Queue()
        self.ready_stations: set[str] = set()
        self.all_ready_event: asyncio.Event = asyncio.Event()

        self.trading_started: bool = False  # <- флаг, что торги уже стартовали

        # тайминги протокола
        self.CFP_COLLECT_MAX_S = 20.0  # максимум собираем офферы столько
        self.CFP_QUIET_S = 3.0         # тихий период без новых офферов для завершения
        self.WAGONS_COLLECT_S = 5.0    # окно сбора PROPOSE_WAGONS

        print(f"[{self.station_id}] Я родился!")

    # ---------------- Behaviours ----------------

    class Setup(OneShotBehaviour):
        async def run(self):
            # 1. Load data
            self.agent.data = load_json(self.agent.data_path)
            self.agent.graph = build_graph(self.agent.data["graph"]["edges"])
            self.agent.stations = self.agent.data["stations"]
            self.agent.orders = list(self.agent.data["orders"])

            # Параметры станции и календарь — один блок инициализации
            my = next(s for s in self.agent.stations if s["id"] == self.agent.station_id)
            self.agent.turnaround_h = float(my.get("turnaround_h", 0.5))
            self.agent.wagon_repos_cost_per_km = float(
                my.get("wagon_repos_cost_per_km", 0.5)
            )

            # Calendars from available_from_h
            for l in my.get("locomotives", []):
                self.agent.free_at_loco[l["id"]] = float(l.get("available_from_h", 0.0))
            for w in my.get("wagons", []):
                self.agent.free_at_wagon[w["id"]] = float(w.get("available_from_h", 0.0))

            # 1) control-очередь
            lc = self.agent.ListenControl()
            self.agent.add_behaviour(lc)
            await asyncio.sleep(0.1)

            # 2) слушатели
            self.agent.add_behaviour(self.agent.ListenHello())
            self.agent.add_behaviour(self.agent.ListenWagons())
            self.agent.add_behaviour(self.agent.ListenCFP())
            await asyncio.sleep(0.1)

            # 3) помечаем себя как живого
            self.agent.ready_stations.add(self.agent.station_id)
            print(f"[{self.agent.station_id}] Я зарегистрировал себя как живого агента.")

            # 4) если этот агент последний — он инициатор барьера
            if self.agent.is_last:
                print(f"[{self.agent.station_id}] Последний агент: опрашиваю всех PING'ом...")
                for st in self.agent.stations:
                    msg = Message(to=st["jid"])
                    msg.set_metadata("type", TYPE_HELLO_PING)
                    msg.body = json.dumps(
                        {"station_id": self.agent.station_id}, ensure_ascii=False
                    )
                    await self.send(msg)

                # ждём, пока увидим всех (или таймаут)
                try:
                    await asyncio.wait_for(self.agent.all_ready_event.wait(), timeout=120.0)
                    print(
                        f"[{self.agent.station_id}] Барьер пройден: я увидел всех агентов. "
                        f"Рассылаю сигнал старта торгов."
                    )
                except asyncio.TimeoutError:
                    print(
                        f"[{self.agent.station_id}] Не все ответили за 120 секунд, "
                        f"запускаю торги с теми, кто онлайн. "
                        f"Рассылаю сигнал старта торгов."
                    )

                # рассылаем старт всем, кроме себя
                for st in self.agent.stations:
                    if st["id"] == self.agent.station_id:
                        continue
                    msg = Message(to=st["jid"])
                    msg.set_metadata("type", TYPE_HELLO_START)
                    msg.body = json.dumps(
                        {"station_id": self.agent.station_id}, ensure_ascii=False
                    )
                    await self.send(msg)

                # запускаем торги у последнего агента
                if not self.agent.trading_started:
                    self.agent.trading_started = True
                    self.agent.add_behaviour(self.agent.InitiateCFP())
                    print(f"[{self.agent.station_id}] Я последний агент и уже начал торги.")

            else:
                # не последний агент: ждём старта от последнего
                print(
                    f"[{self.agent.station_id}] Я не последний агент, "
                    f"торги начну только после сигнала старта от последней станции."
                )

            # 5) idle killer
            self.agent.add_behaviour(self.agent.IdleKiller(timeout_s=180))

            print(
                f"[{self.agent.station_id}] Готов. Оборот={self.agent.turnaround_h} ч, "
                f"аренда вагонов={self.agent.wagon_repos_cost_per_km} ₽/км"
            )

    # --------- Hello PING/PONG ---------
    class ListenHello(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return

            self.agent.last_active = time.time()
            meta = msg.metadata or {}
            mtype = meta.get("type")

            # Если старт торгов
            if mtype == TYPE_HELLO_START:
                payload = json.loads(msg.body)
                starter_id = payload.get("station_id")
                print(
                    f"[{self.agent.station_id}] Получен сигнал старта торгов от последней станции "
                    f"{starter_id}. Запускаю торги."
                )
                if not self.agent.trading_started:
                    self.agent.trading_started = True
                    self.agent.add_behaviour(self.agent.InitiateCFP())
                return

            # PING/PONG обработка
            if mtype not in (TYPE_HELLO_PING, TYPE_HELLO_PONG):
                # forward other stuff to control
                if hasattr(self.agent, "bus") and self.agent.bus:
                    await self.agent.bus.put(msg)
                return

            payload = json.loads(msg.body)
            st_id = payload.get("station_id")
            if not st_id:
                return

            is_new = st_id not in self.agent.ready_stations
            if is_new:
                self.agent.ready_stations.add(st_id)
                if self.agent.is_last:
                    # последний агент логирует, кого увидел
                    print(
                        f"[{self.agent.station_id}] (последний) увидел живого агента: {st_id}. "
                        f"Всего вижу: {len(self.agent.ready_stations)}/{len(self.agent.stations)}"
                    )
                else:
                    # НЕ последний агент логирует, что увидел другого агента
                    print(
                        f"[{self.agent.station_id}] (не последний) увидел агента: {st_id}. "
                        f"Сейчас вижу: {len(self.agent.ready_stations)} агента(ов)."
                    )

            # if ping → reply pong
            if mtype == TYPE_HELLO_PING:
                reply = Message(to=str(msg.sender))
                reply.set_metadata("type", TYPE_HELLO_PONG)
                reply.body = json.dumps(
                    {"station_id": self.agent.station_id}, ensure_ascii=False
                )
                await self.send(reply)

            # Барьер: только последний агент реально ждёт всех
            if self.agent.is_last and len(self.agent.ready_stations) == len(self.agent.stations):
                if not self.agent.all_ready_event.is_set():
                    print(
                        f"[{self.agent.station_id}] (последний) увидел всех "
                        f"{len(self.agent.stations)} агентов. Барьер готов."
                    )
                    self.agent.all_ready_event.set()

    # --------- Wagon rental responses ---------
    class ListenWagons(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return

            self.agent.last_active = time.time()
            meta = msg.metadata or {}
            mtype = meta.get("type")

            if mtype != TYPE_WAGON_CFP:
                if hasattr(self.agent, "bus") and self.agent.bus:
                    await self.agent.bus.put(msg)
                return

            payload = json.loads(msg.body)
            cargo_type = payload["type"]
            need_t = float(payload["need_tonnage"])
            origin = payload["origin"]

            print(
                f"[{self.agent.station_id}] CFP_WAGONS: тип={cargo_type}, "
                f"нужно={need_t} т, origin={origin}"
            )

            my = next(s for s in self.agent.stations if s["id"] == self.agent.station_id)
            offers = []
            acc_t = 0.0
            for w in sorted(my.get("wagons", []), key=lambda x: -float(x["capacity_t"])):
                if w["type"] != cargo_type:
                    continue
                wid = w["id"]
                free = float(
                    self.agent.free_at_wagon.get(
                        wid, float(w.get("available_from_h", 0.0))
                    )
                )
                if self.agent.station_id == origin:
                    path = [origin]
                    dist = 0.0
                    time_h = 0.0
                else:
                    path = shortest_path(self.agent.graph, self.agent.station_id, origin)
                    dist, time_h = path_stats(self.agent.graph, path)

                arrive_origin = free + time_h
                repos_cost = dist * float(self.agent.wagon_repos_cost_per_km)
                offers.append(
                    {
                        "owner_station": self.agent.station_id,
                        "wagon": {
                            "id": wid,
                            "type": w["type"],
                            "capacity_t": float(w["capacity_t"]),
                        },
                        "empty_path": path,
                        "empty_distance_km": dist,
                        "arrive_origin_h": round(arrive_origin, 2),
                        "repos_cost": round(repos_cost, 2),
                    }
                )
                acc_t += float(w["capacity_t"])
                if acc_t >= need_t:
                    break

            print(
                f"[{self.agent.station_id}] PROPOSE_WAGONS: {len(offers)} шт, "
                f"сумм.ёмкость={acc_t} т"
            )

            reply = Message(to=str(msg.sender))
            reply.set_metadata("type", TYPE_WAGON_PROPOSE)
            reply.body = json.dumps(
                {
                    "order_id": payload["order_id"],
                    "from": self.agent.station_id,
                    "wagons": offers,
                },
                ensure_ascii=False,
            )
            await self.send(reply)

    # --------- Handle order CFP ---------
    class ListenCFP(CyclicBehaviour):
        async def run(self):
            print(f"[{self.agent.station_id}] ListenCFP: жду... {time.time()}")
            msg = await self.receive(timeout=1)
            if not msg:
                return

            self.agent.last_active = time.time()
            meta = msg.metadata or {}
            mtype = meta.get("type")

            if mtype != TYPE_ORDER_CFP:
                if hasattr(self.agent, "bus") and self.agent.bus:
                    await self.agent.bus.put(msg)
                return

            payload = json.loads(msg.body)
            od = payload["order"]

            print(f"[{self.agent.station_id}] CFP по {od['id']} получен.")
            # передаём текущий behaviour в _try_build_offer
            offer = await self.agent._try_build_offer(
                od,
                proposer_jid=str(self.agent.jid),
                ctx=self,
            )
            if not offer:
                print(
                    f"[{self.agent.station_id}] Не могу предложить по {od['id']} "
                    f"(нет состава/дедлайн)."
                )
                return

            print(
                f"[{self.agent.station_id}] PROPOSE по {od['id']}: "
                f"cost={offer['total_cost']} T={offer['depart_h']}→{offer['arrive_h']}"
            )
            reply = Message(to=str(msg.sender))
            reply.set_metadata("type", TYPE_ORDER_PROPOSE)
            reply.body = json.dumps(
                {"order_id": od["id"], "offer": offer}, ensure_ascii=False
            )
            await self.send(reply)

    # --------- Control queue (proposals, informs, wagon_propose) ---------
    class ListenControl(CyclicBehaviour):
        async def run(self):
            # очередь теперь только на агенте
            msg = await self.agent.bus.get()

            self.agent.last_active = time.time()
            meta = msg.metadata or {}
            mtype = meta.get("type")

            if mtype == TYPE_ORDER_PROPOSE:
                payload = json.loads(msg.body)
                order_id = payload["order_id"]
                offer = payload["offer"]
                self.agent.offers_box.setdefault(order_id, []).append(offer)

            elif mtype == TYPE_ORDER_INFORM:
                payload = json.loads(msg.body)
                order_id = payload.get("order_id")
                owner = payload.get("owner")  # инициатор заказа
                plan = payload["result"]

                is_owner = (owner == self.agent.station_id)

                if is_owner:
                    print(f"[{self.agent.station_id}] INFORM по {order_id} "
                          f"— записываю результат.")
                else:
                    print(f"[{self.agent.station_id}] INFORM по {order_id} "
                          f"— чужой заказ, обновляю только ресурсы.")

                # инициатор пишет файлы и обновляет календари,
                # остальные — только календари (persist=False)
                await self.agent._append_schedule(plan, persist=is_owner)

            elif mtype == TYPE_WAGON_PROPOSE:
                # аккумулируем ответы по аренде вагонов
                payload = json.loads(msg.body)
                order_id = payload.get("order_id")
                wagons = payload.get("wagons", [])
                if order_id:
                    self.agent.wagon_offers.setdefault(order_id, []).extend(wagons)

            else:
                # другие типы сейчас игнорируем
                pass

    # --------- Initiator for own orders ---------
    class InitiateCFP(OneShotBehaviour):
        async def run(self):
            # свои заказы — по station_id
            my_orders = [
                o for o in self.agent.orders if o["from"] == self.agent.station_id
            ]
            my_orders.sort(
                key=lambda o: (
                    -int(o.get("priority", 0)),
                    float(o.get("earliest_depart_h", 0.0)),
                )
            )

            # список станций нужен для широковещательного INFORM
            recipients = [s for s in self.agent.stations]

            first = True
            for od in my_orders:
                print(
                    f"[{self.agent.station_id}] Новый заказ {od['id']}: "
                    f"{od['from']}→{od['to']} {od['type']} {od['tonnage']} т "
                    f"(prio={od.get('priority', 0)})"
                )
                # Задержка только перед первым заказом
                if first:
                    delay = random.uniform(5, 6)
                    print(
                        f"[{self.agent.station_id}] Delay {delay:.2f}s before first CFP "
                        f"(order {od['id']})"
                    )
                    await asyncio.sleep(delay)
                    first = False

                # Исполнителем считаем станцию-отправителя заказа
                try:
                    executor = next(
                        s for s in self.agent.stations if s["id"] == od["from"]
                    )
                except StopIteration:
                    print(
                        f"[{self.agent.station_id}] Нет исполнителя для заказа {od['id']} "
                        f"(from={od['from']}). Пропускаю."
                    )
                    continue

                msg = Message(to=executor["jid"])
                msg.set_metadata("type", TYPE_ORDER_CFP)
                msg.body = json.dumps({"order": od}, ensure_ascii=False)
                await self.send(msg)

                # Collect offers (от конкретного исполнителя)
                self.agent.offers_box[od["id"]] = []
                start = asyncio.get_event_loop().time()
                last_count = 0
                while True:
                    await asyncio.sleep(0.5)
                    elapsed = asyncio.get_event_loop().time() - start
                    offers = self.agent.offers_box.get(od["id"], [])
                    if elapsed >= self.agent.CFP_COLLECT_MAX_S:
                        print(
                            f"[{self.agent.station_id}] Окно сбора офферов {od['id']} "
                            f"закрыто по максимуму "
                            f"({self.agent.CFP_COLLECT_MAX_S}s). "
                            f"Получено: {len(offers)}"
                        )
                        break
                    # «тихий период»: если количество не меняется в конце окна — выходим
                    if elapsed >= self.agent.CFP_COLLECT_MAX_S - self.agent.CFP_QUIET_S:
                        if len(offers) == last_count:
                            print(
                                f"[{self.agent.station_id}] Тихий период по {od['id']} "
                                f"— завершаю сбор. Получено: {len(offers)}"
                            )
                            break
                        last_count = len(offers)

                offers = self.agent.offers_box.get(od["id"], [])
                print(
                    f"[{self.agent.station_id}] Получено PROPOSE по {od['id']}: "
                    f"{len(offers)}"
                )

                if not offers:
                    print(f"[{self.agent.station_id}] Нет офферов по {od['id']}.")
                    continue

                # choose best by score
                best = min(offers, key=lambda x: self.agent._score_offer(x, od))
                print(
                    f"[{self.agent.station_id}] Победитель {best['proposer']}: "
                    f"стоимость={best['total_cost']} "
                    f"T={best['depart_h']}→{best['arrive_h']}"
                )

                # notify участника
                for of in offers:
                    m = Message(to=of["proposer_jid"])
                    if of is best:
                        m.set_metadata("type", TYPE_ORDER_ACCEPT)
                        m.body = json.dumps(
                            {"order_id": od["id"], "chosen_offer": best},
                            ensure_ascii=False,
                        )
                    else:
                        m.set_metadata("type", TYPE_ORDER_REJECT)
                        m.body = json.dumps(
                            {"order_id": od["id"]}, ensure_ascii=False
                        )
                    await self.send(m)

                # broadcast final INFORM
                for s in recipients:
                    im = Message(to=s["jid"])
                    im.set_metadata("type", TYPE_ORDER_INFORM)
                    im.body = json.dumps(
                        {
                            "order_id": od["id"],
                            "result": best,
                            "owner": self.agent.station_id,  # инициатор заказа
                        },
                        ensure_ascii=False,
                    )
                    await self.send(im)

    # --------- Idle killer ---------
    class IdleKiller(CyclicBehaviour):
        def __init__(self, timeout_s=180):
            super().__init__()
            self.timeout_s = timeout_s

        async def run(self):
            if time.time() - self.agent.last_active > self.timeout_s:
                print(
                    f"[{self.agent.station_id}] Agent idle {self.timeout_s}s — stopping."
                )
                await self.agent.stop()
                return
            await asyncio.sleep(5)

    # ---------------- Offer construction (with wagon rental) ----------------

    async def _try_build_offer(
        self,
        od: Dict[str, Any],
        proposer_jid: str,
        ctx: CyclicBehaviour,
    ) -> Optional[Dict[str, Any]]:
        origin = od["from"]
        dest = od["to"]
        cargo_type = od["type"]
        tonnage = float(od["tonnage"])

        print(
            f"[{self.station_id}] Собираю состав под {od['id']} "
            f"({cargo_type} {tonnage} т)"
        )

        # 1. Own wagons
        own, remaining = self._select_own_wagons(cargo_type, tonnage)
        picked = own[:]
        latest_wagon_eta = max((w["arrive_origin_h"] for w in picked), default=0.0)
        repos_total_cost = 0.0

        # 2. Rent wagons if needed
        if remaining > 1e-6:
            extra, latest_extra_eta, extra_cost = await self._gather_wagons(od, remaining, ctx)
            if not extra:
                print(
                    f"[{self.station_id}] Недостаточно вагонов "
                    f"(даже с арендой) для {od['id']}"
                )
                return None
            picked.extend(extra)
            latest_wagon_eta = max(latest_wagon_eta, latest_extra_eta)
            repos_total_cost += extra_cost

        total_capacity = sum(float(w["wagon"]["capacity_t"]) for w in picked)

        # 3. Choose locomotive
        my = next(s for s in self.stations if s["id"] == self.station_id)
        locos = my.get("locomotives", [])
        loco = None
        for l in sorted(locos, key=lambda x: float(x["cost_per_km"])):
            if float(l["max_tonnage"]) >= total_capacity:
                loco = l
                break
        if not loco:
            print(
                f"[{self.station_id}] Нет подходящей тяги "
                f"({total_capacity} т) для {od['id']}"
            )
            return None

        # 4. Loco empty move to origin
        if self.station_id == origin:
            empty_path = [origin]
            dist_empty = 0.0
            time_empty = 0.0
        else:
            empty_path = shortest_path(self.graph, self.station_id, origin)
            dist_empty, time_empty = path_stats(self.graph, empty_path)

        # 5. Loaded move origin -> dest
        loaded_path = shortest_path(self.graph, origin, dest)
        dist_loaded, time_loaded = path_stats(self.graph, loaded_path)

        # 6. Availability and depart
        free_loco = float(
            self.free_at_loco.get(loco["id"], float(loco.get("available_from_h", 0.0)))
        )
        earliest_order = float(od.get("earliest_depart_h", 0.0))
        depart_h = max(0.0, free_loco, latest_wagon_eta, earliest_order)
        arrive_h = depart_h + time_empty + time_loaded

        # дедлайн
        latest = od.get("latest_arrival_h")
        if latest is not None and arrive_h > float(latest):
            print(
                f"[{self.station_id}] Опоздание по дедлайну для {od['id']}: "
                f"ETA={round(arrive_h, 2)} > {latest}"
            )
            return None

        # 7. Cost
        cost = (dist_empty + dist_loaded) * float(loco["cost_per_km"]) + repos_total_cost

        offer = {
            "order_id": od["id"],
            "proposer": self.station_id,
            "proposer_jid": proposer_jid,
            "loco_id": loco["id"],
            "wagons": [
                {
                    "id": w["wagon"]["id"],
                    "type": w["wagon"]["type"],
                    "capacity_t": w["wagon"]["capacity_t"],
                    "owner": w["owner_station"],
                }
                for w in picked
            ],
            "wagon_moves": [
                {
                    "owner": w["owner_station"],
                    "wagon_id": w["wagon"]["id"],
                    "empty_path": w["empty_path"],
                    "empty_distance_km": w["empty_distance_km"],
                    "arrive_origin_h": w["arrive_origin_h"],
                    "repos_cost": w["repos_cost"],
                }
                for w in picked
            ],
            "empty_path": empty_path,
            "empty_distance_km": round(dist_empty, 2),
            "loaded_path": loaded_path,
            "loaded_distance_km": round(dist_loaded, 2),
            "total_time_h": round(time_empty + time_loaded, 2),
            "total_cost": round(cost, 2),
            "depart_h": round(depart_h, 2),
            "arrive_h": round(arrive_h, 2),
        }
        print(
            f"[{self.station_id}] Состав {od['id']}: локо={loco['id']}, "
            f"вагонов={len(offer['wagons'])}, cap={total_capacity} т, "
            f"cost={offer['total_cost']}, T={offer['depart_h']}→{offer['arrive_h']}"
        )
        return offer

    def _select_own_wagons(
        self, cargo_type: str, tonnage: float
    ) -> Tuple[List[dict], float]:
        my = next(s for s in self.stations if s["id"] == self.station_id)
        sel: List[dict] = []
        remaining = tonnage
        for w in sorted(my.get("wagons", []), key=lambda x: -float(x["capacity_t"])):
            if w["type"] != cargo_type:
                continue
            wid = w["id"]
            free = float(
                self.free_at_wagon.get(wid, float(w.get("available_from_h", 0.0)))
            )
            sel.append(
                {
                    "owner_station": self.station_id,
                    "wagon": {
                        "id": wid,
                        "type": w["type"],
                        "capacity_t": float(w["capacity_t"]),
                    },
                    "empty_path": [self.station_id],
                    "empty_distance_km": 0.0,
                    "arrive_origin_h": free,
                    "repos_cost": 0.0,
                }
            )
            remaining -= float(w["capacity_t"])
            if remaining <= 1e-6:
                break
        return sel, max(0.0, remaining)

    async def _gather_wagons(
        self,
        order: dict,
        need_t: float,
        ctx: CyclicBehaviour,
    ) -> Tuple[List[dict], float, float]:
        origin = order["from"]
        cargo_type = order["type"]
        print(f"[{self.station_id}] Аренда: type={cargo_type}, need={need_t} т, origin={origin}")

        order_id = order["id"]

        # FIX: очищаем старые предложения ДО рассылки CFP,
        # чтобы не потерять быстрые ответы по гонке
        self.wagon_offers.pop(order_id, None)

        # Broadcast wagon_cfp
        for st in self.stations:
            # себя не спрашиваем, свои вагоны уже учтены в _select_own_wagons
            if st["id"] == self.station_id:
                continue
            msg = Message(to=st["jid"])
            msg.set_metadata("type", TYPE_WAGON_CFP)
            msg.body = json.dumps(
                {
                    "order_id": order_id,
                    "origin": origin,
                    "type": cargo_type,
                    "need_tonnage": need_t,
                },
                ensure_ascii=False,
            )
            await ctx.send(msg)

        proposals: List[tuple[float, dict]] = []

        start = asyncio.get_event_loop().time()
        deadline = start + self.WAGONS_COLLECT_S

        # ждём предложения, которые ListenControl складывает в self.wagon_offers
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.2)
            batch = self.wagon_offers.pop(order_id, [])
            if not batch:
                continue
            for item in batch:
                cap = float(item["wagon"]["capacity_t"])
                # «цена» за тонну: repos_cost / capacity
                ppt = (float(item["repos_cost"]) + 1e-6) / max(cap, 1e-6)
                proposals.append((ppt, item))

        # сортируем по цене за тонну
        proposals.sort(key=lambda x: x[0])
        picked: List[dict] = []
        acc = 0.0
        latest = 0.0
        cost = 0.0

        # Суммарный repos_cost = сумма по выбранным вагонам,
        # ETA = максимум arrive_origin_h по выбранным вагонам
        for _, it in proposals:
            picked.append(it)
            acc += float(it["wagon"]["capacity_t"])
            latest = max(latest, float(it["arrive_origin_h"]))
            cost += float(it["repos_cost"])
            if acc >= need_t:
                break

        print(
            f"[{self.station_id}] Аренда итог: {len(picked)} шт, "
            f"сумм.ёмкость={acc} т, ETA={round(latest,2)} ч, "
            f"repos_cost={round(cost,2)}"
        )

        if acc < need_t:
            return [], 0.0, 0.0
        return picked, latest, cost

    # ---------------- Scoring & schedule writing ----------------

    def _score_offer(self, offer: dict, order: dict) -> float:
        cost = float(offer["total_cost"])
        arrive = float(offer["arrive_h"])
        latest = order.get("latest_arrival_h")
        priority = float(order.get("priority", 0))

        lateness = max(0.0, arrive - float(latest)) if latest is not None else 0.0

        ALPHA = 10000.0  # penalty per hour late
        BETA = 2.0       # prefer earlier arrival
        GAMMA = 200.0    # bonus per priority unit

        return cost + ALPHA * lateness + BETA * arrive - GAMMA * priority

    async def _append_schedule(self, plan: Dict[str, Any], persist: bool = True):
        """
        persist=True  -> писать schedule.json/csv + обновлять календари (инициатор).
        persist=False -> только обновлять календари (участники/владельцы ресурсов).
        """
        key_mem = (str(plan.get("order_id")), str(plan.get("proposer")))

        if persist:
            print(f"[{self.station_id}] Записываю результат: {plan.get('order_id')}")

            # in-memory de-dupe только для файлов
            if key_mem in self._written_ids:
                # календари уже должны были быть обновлены в прошлый раз
                return
            self._written_ids.add(key_mem)

            os.makedirs(self.output_dir, exist_ok=True)

            # JSON with de-dupe
            jpath = f"{self.output_dir}/schedule.json"
            try:
                cur = load_json(jpath)
                arr = cur.get("schedule", [])
            except Exception:
                arr = []
            if not any(
                    str(p.get("order_id")) == key_mem[0]
                    and str(p.get("proposer")) == key_mem[1]
                    for p in arr
            ):
                arr.append(plan)
                dump_json({"schedule": arr}, jpath)

            # CSV with de-dupe
            cpath = f"{self.output_dir}/schedule.csv"
            header = [
                "order_id",
                "proposer",
                "loco_id",
                "wagons",
                "empty_distance_km",
                "loaded_distance_km",
                "total_time_h",
                "total_cost",
                "depart_h",
                "arrive_h",
                "empty_path",
                "loaded_path",
            ]

            existing = set()
            if Path(cpath).exists():
                with open(cpath, "r", encoding="utf-8", newline="") as f:
                    try:
                        rd = csv.DictReader(f)
                        for r in rd:
                            existing.add((r.get("order_id"), r.get("proposer")))
                    except Exception:
                        existing = set()

            if key_mem not in existing:
                row = {
                    "order_id": plan.get("order_id"),
                    "proposer": plan.get("proposer"),
                    "loco_id": plan.get("loco_id"),
                    "wagons": ";".join(
                        (w.get("id") or w.get("wagon_id") or "")
                        for w in plan.get("wagons", [])
                    ),
                    "empty_distance_km": plan.get("empty_distance_km", 0.0),
                    "loaded_distance_km": plan.get("loaded_distance_km", 0.0),
                    "total_time_h": plan.get("total_time_h", 0.0),
                    "total_cost": plan.get("total_cost", 0.0),
                    "depart_h": plan.get("depart_h", 0.0),
                    "arrive_h": plan.get("arrive_h", 0.0),
                    "empty_path": " - ".join(plan.get("empty_path", [])),
                    "loaded_path": " - ".join(plan.get("loaded_path", [])),
                }
                write_header = not Path(cpath).exists()
                with open(cpath, "a", encoding="utf-8", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=header)
                    if write_header:
                        wr.writeheader()
                    wr.writerow(row)

        # ---- Calendars update (rental-aware) ----
        end = float(plan.get("arrive_h", 0.0)) + float(
            getattr(self, "turnaround_h", 0.5)
        )

        if not hasattr(self, "free_at_loco"):
            self.free_at_loco = {}
        if not hasattr(self, "free_at_wagon"):
            self.free_at_wagon = {}

        def bump(d: dict, key: str, t: float):
            d[key] = max(float(d.get(key, 0.0)), float(t))

        # Proposer updates its loco
        if plan.get("proposer") == self.station_id and plan.get("loco_id"):
            bump(self.free_at_loco, plan["loco_id"], end)

        # Owners update wagons based on wagon_moves
        for mv in plan.get("wagon_moves", []):
            if mv.get("owner") == self.station_id and mv.get("wagon_id"):
                bump(self.free_at_wagon, mv["wagon_id"], end)

        # If wagons list has owner == me, also bump
        for w in plan.get("wagons", []):
            wid = w.get("id")
            if wid and w.get("owner") == self.station_id:
                bump(self.free_at_wagon, wid, end)

    async def setup(self):
        self.add_behaviour(self.Setup())
