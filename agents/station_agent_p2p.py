import asyncio
import json
import time
import random
from typing import Dict, Any, List, Optional

import networkx as nx
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message

from utils.io_utils import load_json

from agents.module.message_types import *
from agents.module.graph_utils import build_graph, path_stats, shortest_path
from agents.module.offer_utils import *
from agents.module.schedule_utils import append_schedule
from agents.module.behaviours import ListenHello, IdleKiller


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
        self.loco_offers: Dict[str, List[dict]] = {}     # order_id -> loco-offers (аренда)
        self.output_dir = "output"
        self._written_ids: set[tuple[str, str]] = set()

        # Calendars / params
        self.turnaround_h: float = 0.5
        self.wagon_repos_cost_per_km: float = 0.5
        self.free_at_loco: Dict[str, float] = {}
        self.free_at_wagon: Dict[str, float] = {}

        # Текущее положение ресурсов
        self.loco_pos: Dict[str, str] = {}   # loco_id -> station_id
        self.wagon_pos: Dict[str, str] = {}  # wagon_id -> station_id

        # Подтверждения / отказы по заказам (после ACCEPT)
        self.order_confirms: Dict[str, Dict[str, Any]] = {}  # order_id -> план
        self.order_failures: set[str] = set()                # order_id с FAIL

        # Idle / sync
        self.last_active: float = time.time()
        self.bus: asyncio.Queue = asyncio.Queue()
        self.ready_stations: set[str] = set()
        self.all_ready_event: asyncio.Event = asyncio.Event()

        self.trading_started: bool = False

        # Тайминги протокола
        self.CFP_COLLECT_MAX_S = 20.0   # максимум собираем офферы столько
        self.CFP_QUIET_S = 3.0          # тихий период без новых офферов
        self.WAGONS_COLLECT_S = 5.0     # окно сбора PROPOSE_WAGONS/LOCO

        # Повторы заказов после неуспеха ACCEPT/CONFIRM
        self.ORDER_RETRY_MAX_ATTEMPTS = 3   # максимум попыток
        self.ORDER_RETRY_DELAY_S = 3.0      # пауза между попытками

        print(f"[{self.station_id}] Я родился!")

    # ---------------- Behaviours ----------------

    async def setup(self):
        self.add_behaviour(self.Setup())

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
                lid = l["id"]
                self.agent.free_at_loco[lid] = float(l.get("available_from_h", 0.0))
                self.agent.loco_pos[lid] = self.agent.station_id  # стартовая позиция
            for w in my.get("wagons", []):
                wid = w["id"]
                self.agent.free_at_wagon[wid] = float(w.get("available_from_h", 0.0))
                self.agent.wagon_pos[wid] = self.agent.station_id  # стартовая позиция

            # 1) control-очередь
            lc = self.agent.ListenControl()
            self.agent.add_behaviour(lc)
            await asyncio.sleep(0.1)

            # 2) слушатели
            self.agent.add_behaviour(ListenHello())
            self.agent.add_behaviour(self.agent.ListenWagons())
            self.agent.add_behaviour(self.agent.ListenCFP())
            self.agent.add_behaviour(self.agent.ListenLocos())
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
                    await asyncio.wait_for(
                        self.agent.all_ready_event.wait(), timeout=120.0
                    )
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
                print(
                    f"[{self.agent.station_id}] Я не последний агент, "
                    f"торги начну только после сигнала старта от последней станции."
                )

            # 5) idle killer
            self.agent.add_behaviour(IdleKiller(timeout_s=180))

            print(
                f"[{self.agent.station_id}] Готов. Оборот={self.agent.turnaround_h} ч, "
                f"аренда вагонов={self.agent.wagon_repos_cost_per_km} ₽/км"
            )

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

                start_station = self.agent.wagon_pos.get(wid, self.agent.station_id)
                if start_station == origin:
                    path = [origin]
                    dist = 0.0
                    time_h = 0.0
                else:
                    path = shortest_path(self.agent.graph, start_station, origin)
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

    # --------- Loco rental responses ---------
    class ListenLocos(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return

            self.agent.last_active = time.time()
            meta = msg.metadata or {}
            mtype = meta.get("type")

            if mtype != TYPE_LOCO_CFP:
                if hasattr(self.agent, "bus") and self.agent.bus:
                    await self.agent.bus.put(msg)
                return

            payload = json.loads(msg.body)
            order_id = payload["order_id"]
            origin = payload["origin"]
            need_tonnage = float(payload["need_tonnage"])

            print(
                f"[{self.agent.station_id}] CFP_LOCO: нужно тяги под {need_tonnage} т, "
                f"origin={origin}"
            )

            my = next(s for s in self.agent.stations if s["id"] == self.agent.station_id)
            offers = []

            for l in sorted(my.get("locomotives", []), key=lambda x: float(x["cost_per_km"])):
                if float(l["max_tonnage"]) < need_tonnage:
                    continue
                lid = l["id"]
                free = float(
                    self.agent.free_at_loco.get(
                        lid, float(l.get("available_from_h", 0.0))
                    )
                )
                start_station = self.agent.loco_pos.get(lid, self.agent.station_id)
                if start_station == origin:
                    path = [origin]
                    dist = 0.0
                    time_h = 0.0
                else:
                    path = shortest_path(self.agent.graph, start_station, origin)
                    dist, time_h = path_stats(self.agent.graph, path)

                arrive_origin = free + time_h
                repos_cost = dist * float(l["cost_per_km"])

                offers.append(
                    {
                        "owner_station": self.agent.station_id,
                        "loco": {
                            "id": lid,
                            "max_tonnage": float(l["max_tonnage"]),
                            "cost_per_km": float(l["cost_per_km"]),
                        },
                        "empty_path": path,
                        "empty_distance_km": round(dist, 2),
                        "arrive_origin_h": round(arrive_origin, 2),
                        "repos_cost": round(repos_cost, 2),
                    }
                )

            print(
                f"[{self.agent.station_id}] PROPOSE_LOCO: {len(offers)} шт"
            )

            reply = Message(to=str(msg.sender))
            reply.set_metadata("type", TYPE_LOCO_PROPOSE)
            reply.body = json.dumps(
                {
                    "order_id": order_id,
                    "from": self.agent.station_id,
                    "locos": offers,
                },
                ensure_ascii=False,
            )
            await self.send(reply)

    # --------- Handle order CFP ---------
    class ListenCFP(CyclicBehaviour):
        async def run(self):
            # print(f"[{self.agent.station_id}] ListenCFP: жду... {time.time()}")
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
            offer = await try_build_offer(
                self.agent,
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

    # --------- Control queue (proposals, informs, wagon/loco_propose, accept/confirm) ---------
    class ListenControl(CyclicBehaviour):
        async def run(self):
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

                await append_schedule(self.agent, plan, persist=is_owner)

            elif mtype == TYPE_WAGON_PROPOSE:
                payload = json.loads(msg.body)
                order_id = payload.get("order_id")
                wagons = payload.get("wagons", [])
                if order_id:
                    self.agent.wagon_offers.setdefault(order_id, []).extend(wagons)

            elif mtype == TYPE_LOCO_PROPOSE:
                payload = json.loads(msg.body)
                order_id = payload.get("order_id")
                locos = payload.get("locos", [])
                if order_id:
                    self.agent.loco_offers.setdefault(order_id, []).extend(locos)

            elif mtype == TYPE_ORDER_ACCEPT:
                payload = json.loads(msg.body)
                order_id = payload["order_id"]
                chosen = payload["chosen_offer"]

                if chosen.get("proposer") == self.agent.station_id:
                    print(f"[{self.agent.station_id}] ACCEPT по заказу {order_id}, "
                          f"проверяю выполнимость плана.")
                    await handle_order_accept(
                        self.agent,
                        msg=msg,
                        order_id=order_id,
                        chosen_offer=chosen,
                        ctx=self,
                    )

            elif mtype == TYPE_ORDER_REJECT:
                payload = json.loads(msg.body)
                print(f"[{self.agent.station_id}] REJECT по заказу {payload.get('order_id')}")

            elif mtype == TYPE_ORDER_CONFIRM:
                payload = json.loads(msg.body)
                order_id = payload.get("order_id")
                plan = payload.get("plan")
                if order_id and plan:
                    print(f"[{self.agent.station_id}] CONFIRM по заказу {order_id} получен.")
                    self.agent.order_confirms[order_id] = plan

            elif mtype == TYPE_ORDER_FAIL:
                payload = json.loads(msg.body)
                order_id = payload.get("order_id")
                if order_id:
                    print(f"[{self.agent.station_id}] FAIL по заказу {order_id} от исполнителя.")
                    self.agent.order_failures.add(order_id)

            else:
                # другие типы сейчас игнорируем
                pass

    # --------- Initiator for own orders (with retries & confirm) ---------
    class InitiateCFP(OneShotBehaviour):
        async def run(self):
            my_orders = [
                o for o in self.agent.orders if o["from"] == self.agent.station_id
            ]
            my_orders.sort(
                key=lambda o: (
                    -int(o.get("priority", 0)),
                    float(o.get("earliest_depart_h", 0.0)),
                )
            )

            recipients = [s for s in self.agent.stations]

            first = True
            for od in my_orders:
                order_id = od["id"]
                print(
                    f"[{self.agent.station_id}] Новый заказ {order_id}: "
                    f"{od['from']}→{od['to']} {od['type']} {od['tonnage']} т "
                    f"(prio={od.get('priority', 0)})"
                )
                if first:
                    delay = random.uniform(5, 6)
                    print(
                        f"[{self.agent.station_id}] Delay {delay:.2f}s before first CFP "
                        f"(order {order_id})"
                    )
                    await asyncio.sleep(delay)
                    first = False

                try:
                    executor = next(
                        s for s in self.agent.stations if s["id"] == od["from"]
                    )
                except StopIteration:
                    print(
                        f"[{self.agent.station_id}] Нет исполнителя для заказа {order_id} "
                        f"(from={od['from']}). Пропускаю."
                    )
                    continue

                # ---- ЦИКЛ ПОВТОРОВ ПО ОДНОМУ ЗАКАЗУ ----
                attempts = 0

                while attempts < self.agent.ORDER_RETRY_MAX_ATTEMPTS:
                    # Проверка: не просрочен ли заказ
                    if not order_not_expired(self.agent, od):
                        print(
                            f"[{self.agent.station_id}] Заказ {order_id} уже просрочен. "
                            f"Повторять не буду."
                        )
                        break

                    attempts += 1
                    print(
                        f"[{self.agent.station_id}] Попытка #{attempts} выполнить заказ {order_id}"
                    )

                    # --- Шлём CFP исполнителю ---
                    msg = Message(to=executor["jid"])
                    msg.set_metadata("type", TYPE_ORDER_CFP)
                    msg.body = json.dumps({"order": od}, ensure_ascii=False)
                    await self.send(msg)

                    # --- Сбор офферов ---
                    self.agent.offers_box[order_id] = []
                    start = asyncio.get_event_loop().time()
                    last_count = 0
                    while True:
                        await asyncio.sleep(0.5)
                        elapsed = asyncio.get_event_loop().time() - start
                        offers = self.agent.offers_box.get(order_id, [])
                        if elapsed >= self.agent.CFP_COLLECT_MAX_S:
                            print(
                                f"[{self.agent.station_id}] Окно сбора офферов {order_id} "
                                f"закрыто по максимуму "
                                f"({self.agent.CFP_COLLECT_MAX_S}s). "
                                f"Получено: {len(offers)}"
                            )
                            break
                        if elapsed >= self.agent.CFP_COLLECT_MAX_S - self.agent.CFP_QUIET_S:
                            if len(offers) == last_count:
                                print(
                                    f"[{self.agent.station_id}] Тихий период по {order_id} "
                                    f"— завершаю сбор. Получено: {len(offers)}"
                                )
                                break
                            last_count = len(offers)

                    offers = self.agent.offers_box.get(order_id, [])
                    print(
                        f"[{self.agent.station_id}] Получено PROPOSE по {order_id}: "
                        f"{len(offers)}"
                    )

                    if not offers:
                        print(
                            f"[{self.agent.station_id}] Нет офферов по {order_id} "
                            f"на попытке #{attempts}."
                        )
                        if attempts < self.agent.ORDER_RETRY_MAX_ATTEMPTS:
                            await asyncio.sleep(self.agent.ORDER_RETRY_DELAY_S)
                            continue
                        else:
                            break

                    # choose best by score
                    best = min(offers, key=lambda x: score_offer(x, od))
                    print(
                        f"[{self.agent.station_id}] Победитель {best['proposer']}: "
                        f"стоимость={best['total_cost']} "
                        f"T={best['depart_h']}→{best['arrive_h']}"
                    )

                    # --- ACCEPT победителю, REJECT остальным ---
                    for of in offers:
                        m = Message(to=of["proposer_jid"])
                        if of is best:
                            m.set_metadata("type", TYPE_ORDER_ACCEPT)
                            m.body = json.dumps(
                                {"order_id": order_id, "chosen_offer": best},
                                ensure_ascii=False,
                            )
                        else:
                            m.set_metadata("type", TYPE_ORDER_REJECT)
                            m.body = json.dumps(
                                {"order_id": order_id},
                                ensure_ascii=False,
                            )
                        await self.send(m)

                    # --- Ждём CONFIRM/FAIL ---
                    self.agent.order_confirms.pop(order_id, None)
                    self.agent.order_failures.discard(order_id)

                    confirm_timeout = 15.0  # сек
                    start_wait = asyncio.get_event_loop().time()
                    confirmed_plan = None

                    while True:
                        await asyncio.sleep(0.5)
                        if order_id in self.agent.order_failures:
                            print(
                                f"[{self.agent.station_id}] Исполнитель отказался от заказа {order_id} "
                                f"после ACCEPT (FAIL)."
                            )
                            break

                        plan = self.agent.order_confirms.get(order_id)
                        if plan:
                            confirmed_plan = plan
                            break

                        if asyncio.get_event_loop().time() - start_wait > confirm_timeout:
                            print(
                                f"[{self.agent.station_id}] Не дождался подтверждения по {order_id} "
                                f"(timeout {confirm_timeout}s)."
                            )
                            break

                    if not confirmed_plan:
                        if (
                            attempts < self.agent.ORDER_RETRY_MAX_ATTEMPTS
                            and order_not_expired(self.agent, od)
                        ):
                            print(
                                f"[{self.agent.station_id}] Повторяю заказ {order_id} "
                                f"через {self.agent.ORDER_RETRY_DELAY_S}s "
                                f"(срок ещё не истёк)."
                            )
                            await asyncio.sleep(self.agent.ORDER_RETRY_DELAY_S)
                            continue
                        else:
                            print(
                                f"[{self.agent.station_id}] Заказ {order_id} остаётся невыполненным "
                                f"после {attempts} попыток."
                            )
                            break

                    # ---- Есть подтверждённый план: рассылаем INFORM ----
                    print(
                        f"[{self.agent.station_id}] Заказ {order_id} подтверждён исполнителем "
                        f"{confirmed_plan['proposer']}. Рассылаю INFORM."
                    )

                    for s in recipients:
                        im = Message(to=s["jid"])
                        im.set_metadata("type", TYPE_ORDER_INFORM)
                        im.body = json.dumps(
                            {
                                "order_id": order_id,
                                "result": confirmed_plan,
                                "owner": self.agent.station_id,
                            },
                            ensure_ascii=False,
                        )
                        await self.send(im)

                    # заказ успешно выполнен, выходим из цикла попыток
                    break
