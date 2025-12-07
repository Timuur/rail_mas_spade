from typing import Dict, Any, List
import networkx as nx
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from pathlib import Path
import os, csv, asyncio, json, time
from utils.io_utils import load_json, dump_json

# --- перформативы ---
CFP = "CFP"
PROPOSE = "PROPOSE"
ACCEPT = "ACCEPT"
REJECT = "REJECT"
INFORM = "INFORM"
CFP_WAGONS = "CFP_WAGONS"
PROPOSE_WAGONS = "PROPOSE_WAGONS"
HELLO_PING = "HELLO_PING"
HELLO_PONG = "HELLO_PONG"


# --- графовые утилиты ---
def build_graph(edges: list[dict]) -> nx.Graph:
    g = nx.Graph()
    for e in edges:
        u, v = e["u"], e["v"]
        g.add_edge(u, v, distance_km=e["distance_km"], speed_kmh=e.get("speed_kmh", 60))
    return g

def path_stats(g: nx.Graph, path: list[str]) -> tuple[float, float]:
    d = t = 0.0
    for i in range(len(path) - 1):
        data = g.get_edge_data(path[i], path[i + 1])
        d += data["distance_km"]
        t += data["distance_km"] / data.get("speed_kmh", 60)
    return (round(d, 2), round(t, 2))

def shortest_path(g: nx.Graph, a: str, b: str) -> list[str]:
    return nx.shortest_path(g, a, b, weight=lambda u, v, d: d["distance_km"])


class StationAgent(Agent):
    def __init__(self, jid, password, station_id: str, data_path: str, is_last: bool = False, **kwargs):
        super().__init__(jid, password, **kwargs)
        self.station_id = station_id
        self.data_path = data_path
        self.is_last = is_last

        self.data: Dict[str, Any] = {}
        self.graph: nx.Graph | None = None
        self.stations: list[dict] = []
        self.orders: list[dict] = []
        self.rolling: dict[str, Any] = {}
        self.offers_box: dict[str, list[dict]] = {}  # order_id -> offers

        self.output_dir = "output"

        # Idle control
        self.last_active: float = time.time()

        # Барьер готовности
        self.ready_stations: set[str] = set()
        self.all_ready_event: asyncio.Event = asyncio.Event()

        self._written_ids = set()

        # ресурсы
        self.free_at_loco: dict[str, float] = {}
        self.free_at_wagon: dict[str, float] = {}
        self.turnaround_h: float = 0.5
        self.wagon_repos_cost_per_km: float = 0.5

        # тайминги протокола
        self.CFP_COLLECT_MAX_S = 20.0   # максимум собираем офферы столько
        self.CFP_QUIET_S = 3.0          # тихий период без новых офферов для завершения
        self.WAGONS_COLLECT_S = 5.0     # окно сбора PROPOSE_WAGONS

        print(f"[{self.station_id}] Я родился!")

    # ---------------------- setup ----------------------
    async def setup(self):
        self.add_behaviour(self.Setup())

    class Setup(OneShotBehaviour):
        async def run(self):
            self.agent.data = load_json(self.agent.data_path)
            self.agent.graph = build_graph(self.agent.data["graph"]["edges"])
            self.agent.stations = self.agent.data["stations"]
            self.agent.orders = [o for o in self.agent.data["orders"]]
            self.agent.rolling = {
                s["id"]: {"locomotives": s.get("locomotives", []), "wagons": s.get("wagons", [])}
                for s in self.agent.stations
            }

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

            # 4) если этот агент последний — он инициатор барьера
            if self.agent.is_last:
                print(f"[{self.agent.station_id}] Последний агент: опрашиваю всех PING'ом...")
                for st in self.agent.stations:
                    msg = Message(to=st["jid"])
                    msg.set_metadata("performative", HELLO_PING)
                    msg.body = json.dumps({"station_id": self.agent.station_id}, ensure_ascii=False)
                    await self.send(msg)

                # ждём, пока все ответят (или пока не истечёт таймаут)
                try:
                    await asyncio.wait_for(self.agent.all_ready_event.wait(), timeout=120.0)
                    print(f"[{self.agent.station_id}] Все станции ответили, запускаю торги.")
                except asyncio.TimeoutError:
                    print(
                        f"[{self.agent.station_id}] Не все ответили за 120 секунд, запускаю торги с теми, кто online.")

                # стартуем торги
                self.agent.add_behaviour(self.agent.InitiateCFP())

            # 5) idle killer как раньше
            self.agent.add_behaviour(self.agent.IdleKiller(timeout_s=180))

            # параметры станции
            my = next(s for s in self.agent.stations if s["id"] == self.agent.station_id)
            self.agent.turnaround_h = float(my.get("turnaround_h", 0.5))
            self.agent.wagon_repos_cost_per_km = float(my.get("wagon_repos_cost_per_km", 0.5))

            # инициализация календарей
            for l in my.get("locomotives", []):
                self.agent.free_at_loco[l["id"]] = float(l.get("available_from_h", 0.0))
            for w in my.get("wagons", []):
                self.agent.free_at_wagon[w["id"]] = float(w.get("available_from_h", 0.0))

            print(f"[{self.agent.station_id}] Готов. Оборот={self.agent.turnaround_h} ч, "
                  f"аренда вагонов={self.agent.wagon_repos_cost_per_km} ₽/км")

    # ---------------------- сторожок ----------------------
    class IdleKiller(CyclicBehaviour):
        def __init__(self, timeout_s=120):
            super().__init__()
            self.timeout_s = timeout_s

        async def run(self):
            if time.time() - self.agent.last_active > self.timeout_s:
                print(f"[{self.agent.station_id}] Агент неактивен {self.timeout_s} сек — завершаюсь.")
                await self.agent.stop()
                return
            await asyncio.sleep(5)

    # ---------------------- Барьер готовности ----------------------
    class ListenHello(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return

            self.agent.last_active = time.time()
            meta = msg.metadata or {}
            pf = meta.get("performative")

            if pf not in (HELLO_PING, HELLO_PONG):
                # всё остальное отправим в control, как раньше
                if hasattr(self.agent, "bus") and self.agent.bus:
                    await self.agent.bus.put(msg)
                return

            payload = json.loads(msg.body)
            st_id = payload.get("station_id")
            if not st_id:
                return

            # запоминаем, что видели эту станцию живой
            self.agent.ready_stations.add(st_id)
            print(f"[{self.agent.station_id}] Видит живой станцию: {st_id}")

            # если получили PING → ответим PONG
            if pf == HELLO_PING:
                reply = Message(to=str(msg.sender))
                reply.set_metadata("performative", HELLO_PONG)
                reply.body = json.dumps({"station_id": self.agent.station_id}, ensure_ascii=False)
                await self.send(reply)

            # если теперь количество живых станций = ожидаемому → снимаем барьер
            if len(self.agent.ready_stations) == len(self.agent.stations):
                self.agent.all_ready_event.set()

    # ---------------------- вагонный брокер ----------------------
    class ListenWagons(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return
            self.agent.last_active = time.time()

            meta = msg.metadata or {}
            if meta.get("performative") != CFP_WAGONS:
                if hasattr(self.agent, "bus") and self.agent.bus:
                    await self.agent.bus.put(msg)
                return

            payload = json.loads(msg.body)
            cargo_type = payload["type"]
            need_t = float(payload["need_tonnage"])
            origin = payload["origin"]

            print(f"[{self.agent.station_id}] CFP_WAGONS: тип={cargo_type}, нужно={need_t} т, origin={origin}")

            my = next(s for s in self.agent.stations if s["id"] == self.agent.station_id)
            offers = []
            acc_t = 0.0

            for w in sorted(my.get("wagons", []), key=lambda x: -x["capacity_t"]):
                if w["type"] != cargo_type:
                    continue
                wid = w["id"]
                free = float(self.agent.free_at_wagon.get(wid, float(w.get("available_from_h", 0.0))))
                if self.agent.station_id == origin:
                    path = [origin]; dist = 0.0; time_h = 0.0
                else:
                    path = shortest_path(self.agent.graph, self.agent.station_id, origin)
                    dist, time_h = path_stats(self.agent.graph, path)
                arrive_origin = free + time_h
                repos_cost = dist * float(self.agent.wagon_repos_cost_per_km)

                offers.append({
                    "owner_station": self.agent.station_id,
                    "wagon": {"id": wid, "type": w["type"], "capacity_t": float(w["capacity_t"])},
                    "empty_path": path,
                    "empty_distance_km": dist,
                    "arrive_origin_h": round(arrive_origin, 2),
                    "repos_cost": round(repos_cost, 2)
                })
                acc_t += float(w["capacity_t"])
                if acc_t >= need_t:
                    break

            print(f"[{self.agent.station_id}] PROPOSE_WAGONS: {len(offers)} шт, сумм.ёмкость={acc_t} т")

            reply = Message(to=str(msg.sender))
            reply.set_metadata("performative", PROPOSE_WAGONS)
            reply.set_metadata("for_order_id", payload["order_id"])
            reply.body = json.dumps({"from": self.agent.station_id, "wagons": offers}, ensure_ascii=False)
            await self.send(reply)

    # ---------------------- инициатор ----------------------
    class InitiateCFP(OneShotBehaviour):
        async def run(self):
            my_orders = [o for o in self.agent.orders if o["from"] == self.agent.station_id]
            my_orders.sort(key=lambda o: (-int(o.get("priority", 0)), float(o.get("earliest_depart_h", 0.0))))
            if not my_orders:
                return
            recipients = [s for s in self.agent.stations if s["id"] != "COORD"]

            for od in my_orders:
                print(f"[{self.agent.station_id}] Новый заказ {od['id']}: {od['from']}→{od['to']} "
                      f"{od['type']} {od['tonnage']} т (prio={od.get('priority',0)})")

                for s in recipients:
                    msg = Message(to=s["jid"])
                    msg.set_metadata("performative", CFP)
                    msg.set_metadata("order_id", od["id"])
                    msg.body = json.dumps(od, ensure_ascii=False)
                    await self.send(msg)

                # адаптивный сбор офферов
                self.agent.offers_box[od["id"]] = []
                start = asyncio.get_event_loop().time()
                last_count = 0
                while True:
                    await asyncio.sleep(0.5)
                    elapsed = asyncio.get_event_loop().time() - start
                    offers = self.agent.offers_box.get(od["id"], [])
                    if elapsed >= self.agent.CFP_COLLECT_MAX_S:
                        print(f"[{self.agent.station_id}] Окно сбора офферов {od['id']} закрыто по максимуму "
                              f"({self.agent.CFP_COLLECT_MAX_S}s). Получено: {len(offers)}")
                        break
                    # «тихий период»: если количество не меняется в конце окна — выходим
                    if elapsed >= self.agent.CFP_COLLECT_MAX_S - self.agent.CFP_QUIET_S:
                        if len(offers) == last_count:
                            print(f"[{self.agent.station_id}] Тихий период по {od['id']} — завершаю сбор. "
                                  f"Получено: {len(offers)}")
                            break
                        last_count = len(offers)

                offers = self.agent.offers_box.get(od["id"], [])
                print(f"[{self.agent.station_id}] Получено PROPOSE по {od['id']}: {len(offers)}")

                if not offers:
                    print(f"[{self.agent.station_id}] Нет офферов по {od['id']}.")
                    continue

                best = min(offers, key=lambda x: self._score_offer(x, od))
                print(f"[{self.agent.station_id}] Победитель {best['proposer']}: "
                      f"стоимость={best['total_cost']} T={best['depart_h']}→{best['arrive_h']}")

                # ACCEPT/REJECT
                for of in offers:
                    m = Message(to=of["proposer_jid"])
                    m.set_metadata("order_id", od["id"])
                    if of is best:
                        m.set_metadata("performative", ACCEPT)
                        m.body = json.dumps(best, ensure_ascii=False)
                    else:
                        m.set_metadata("performative", REJECT)
                        m.body = json.dumps({"order_id": od["id"]}, ensure_ascii=False)
                    await self.send(m)

                # INFORM broadcast
                for s in recipients:
                    im = Message(to=s["jid"])
                    im.set_metadata("performative", INFORM)
                    im.set_metadata("order_id", od["id"])
                    im.body = json.dumps(best, ensure_ascii=False)
                    await self.send(im)

                await self.agent._append_schedule(best)

        def _score_offer(self, offer: dict, order: dict) -> float:
            cost = float(offer["total_cost"])
            arrive = float(offer["arrive_h"])
            latest = order.get("latest_arrival_h")
            priority = float(order.get("priority", 0))
            lateness = max(0.0, arrive - float(latest)) if latest is not None else 0.0
            ALPHA = 10000.0  # штраф за опоздание
            BETA = 2.0      # ранние прибытия чуть предпочтительней
            GAMMA = 200.0   # бонус за приоритет
            return cost + ALPHA * lateness + BETA * arrive - GAMMA * priority

    # ---------------------- исполнители: ответы на CFP ----------------------
    class ListenCFP(CyclicBehaviour):
        async def run(self):
            print(f"[{self.agent.station_id}] ListenCFP: жду...")
            msg = await self.receive(timeout=1)
            if not msg:
                return

            # любое сообщение = агент активен
            self.agent.last_active = time.time()

            meta = msg.metadata or {}
            if meta.get("performative") != CFP:
                if hasattr(self.agent, "bus") and self.agent.bus:
                    await self.agent.bus.put(msg)
                return

            od = json.loads(msg.body)
            print(f"[{self.agent.station_id}] CFP по {od['id']} получен.")
            offer = await self.agent._try_build_offer(od, proposer_jid=str(self.agent.jid), beh=self)
            if not offer:
                print(f"[{self.agent.station_id}] Не могу предложить по {od['id']} (нет состава/дедлайн).")
                return

            print(f"[{self.agent.station_id}] PROPOSE по {od['id']}: "
                  f"cost={offer['total_cost']} T={offer['depart_h']}→{offer['arrive_h']}")
            reply = Message(to=str(msg.sender))
            reply.set_metadata("performative", PROPOSE)
            reply.set_metadata("order_id", od["id"])
            reply.body = json.dumps(offer, ensure_ascii=False)
            await self.send(reply)

    # ---------------------- control-поведение ----------------------
    class ListenControl(CyclicBehaviour):
        def __init__(self):
            super().__init__()
            self.bus = asyncio.Queue()

        async def on_start(self):
            self.agent.bus = self.bus

        async def run(self):
            msg = await self.bus.get()
            self.agent.last_active = time.time()
            meta = msg.metadata or {}
            pf = meta.get("performative")

            if pf == PROPOSE:
                order_id = meta.get("order_id")
                offer = json.loads(msg.body)
                self.agent.offers_box.setdefault(order_id, []).append(offer)

            elif pf == INFORM:
                content = json.loads(msg.body)
                print(f"[{self.agent.station_id}] INFORM по {content.get('order_id')} — записываю результат.")
                await self.agent._append_schedule(content)
            # на ACCEPT/REJECT ничего не пишем — избегаем дублей

    # ---------------------- helpers: выбор вагонов/аренда ----------------------
    def _select_own_wagons(self, cargo_type: str, tonnage: float, origin: str):
        """Вернёт (список_moves, суммарная_ёмкость, latest_arrive_origin)."""
        my = next(s for s in self.stations if s["id"] == self.station_id)
        picked = []
        acc = 0.0
        latest = 0.0
        for w in sorted(my.get("wagons", []), key=lambda x: -x["capacity_t"]):
            if w["type"] != cargo_type:
                continue
            wid = w["id"]
            free = float(self.free_at_wagon.get(wid, float(w.get("available_from_h", 0.0))))
            if self.station_id == origin:
                path = [origin]; dist = 0.0; time_h = 0.0
            else:
                path = shortest_path(self.graph, self.station_id, origin)
                dist, time_h = path_stats(self.graph, path)
            arrive_origin = free + time_h
            picked.append({
                "owner_station": self.station_id,
                "wagon": {"id": wid, "type": w["type"], "capacity_t": float(w["capacity_t"])},
                "empty_path": path,
                "empty_distance_km": dist,
                "arrive_origin_h": round(arrive_origin, 2),
                "repos_cost": 0.0  # свои вагоны: не считаем отдельную «арендную» цену
            })
            acc += float(w["capacity_t"])
            latest = max(latest, arrive_origin)
            if acc >= tonnage:
                break
        return picked, acc, latest

    async def _gather_wagons(self, order: dict, need_t: float, beh) -> tuple[list[dict], float, float]:
        """Собрать недостающие вагоны из сети через beh.send/beh.receive."""
        origin = order["from"]; cargo_type = order["type"]
        print(f"[{self.station_id}] Аренда: type={cargo_type}, need={need_t} т, origin={origin}")

        # разослать CFP_WAGONS
        for st in self.stations:
            msg = Message(to=st["jid"])
            msg.set_metadata("performative", CFP_WAGONS)
            msg.body = json.dumps({"order_id": order["id"], "origin": origin, "type": cargo_type,
                                   "need_tonnage": need_t}, ensure_ascii=False)
            await beh.send(msg)

        # собрать PROPOSE_WAGONS
        proposals = []
        deadline = asyncio.get_event_loop().time() + self.WAGONS_COLLECT_S
        while asyncio.get_event_loop().time() < deadline:
            msg = await beh.receive(timeout=0.5)
            if not msg:
                continue
            meta = msg.metadata or {}
            if meta.get("performative") != PROPOSE_WAGONS:
                if hasattr(self, "bus") and self.bus:
                    await self.bus.put(msg)
                continue
            payload = json.loads(msg.body)
            for item in payload.get("wagons", []):
                cap = float(item["wagon"]["capacity_t"])
                price_per_t = (float(item["repos_cost"]) + 1e-6) / cap
                proposals.append((price_per_t, item))

        proposals.sort(key=lambda x: x[0])
        picked = []; acc = 0.0; latest = 0.0; cost = 0.0
        for _, it in proposals:
            picked.append(it)
            acc += float(it["wagon"]["capacity_t"])
            latest = max(latest, float(it["arrive_origin_h"]))
            cost += float(it["repos_cost"])
            if acc >= need_t:
                break

        print(f"[{self.station_id}] Аренда итог: {len(picked)} шт, сумм.ёмкость={acc} т, "
              f"ETA={round(latest,2)} ч, repos_cost={round(cost,2)}")

        if acc < need_t:
            return [], 0.0, 0.0
        return picked, latest, cost

    # ---------------------- построение оффера ----------------------
    async def _try_build_offer(self, od: Dict[str, Any], proposer_jid: str, beh) -> Dict[str, Any] | None:
        origin = od["from"]; dest = od["to"]; cargo_type = od["type"]; tonnage = float(od["tonnage"])
        print(f"[{self.station_id}] Собираю состав под {od['id']} ({cargo_type} {tonnage} т)")

        # свои вагоны
        own, own_cap, own_latest = self._select_own_wagons(cargo_type, tonnage, origin)

        picked = list(own)
        repos_total_cost = 0.0
        latest_wagon_eta = own_latest

        # арендуем недостающее
        if own_cap + 1e-6 < tonnage:
            need = tonnage - own_cap
            extra, latest_extra_eta, extra_cost = await self._gather_wagons(od, need, beh)
            if not extra:
                print(f"[{self.station_id}] Недостаточно вагонов (даже с арендой) для {od['id']}")
                return None
            picked.extend(extra)
            repos_total_cost += extra_cost
            latest_wagon_eta = max(latest_wagon_eta, latest_extra_eta)

        total_cap = sum(x["wagon"]["capacity_t"] for x in picked)

        # локомотив
        st = next(s for s in self.stations if s["id"] == self.station_id)
        locos = st.get("locomotives", [])
        loco = None
        for l in sorted(locos, key=lambda x: float(x["cost_per_km"])):
            if float(l["max_tonnage"]) >= total_cap:
                loco = l; break
        if not loco:
            print(f"[{self.station_id}] Нет подходящей тяги ({total_cap} т) для {od['id']}")
            return None

        # доступность и окно заказа
        free_loco = float(self.free_at_loco.get(loco["id"], float(loco.get("available_from_h", 0.0))))
        earliest_order = float(od.get("earliest_depart_h", 0.0))
        depart_h = max(0.0, free_loco, latest_wagon_eta, earliest_order)

        # маршруты
        if self.station_id == origin:
            empty_path = [origin]; dist_empty = 0.0; time_empty = 0.0
        else:
            empty_path = shortest_path(self.graph, self.station_id, origin)
            dist_empty, time_empty = path_stats(self.graph, empty_path)

        loaded_path = shortest_path(self.graph, origin, dest)
        dist_loaded, time_loaded = path_stats(self.graph, loaded_path)

        arrive_origin_h = depart_h + time_empty
        arrive_h = arrive_origin_h + time_loaded

        # дедлайн
        latest = od.get("latest_arrival_h")
        if latest is not None and arrive_h > float(latest):
            print(f"[{self.station_id}] Опоздание по дедлайну для {od['id']}: ETA={round(arrive_h,2)} > {latest}")
            return None

        # стоимость: локо (пустой+гружёный) + репозиция арендованных вагонов
        loco_cost = (dist_empty + dist_loaded) * float(loco["cost_per_km"])
        cost = loco_cost + repos_total_cost

        offer = {
            "order_id": od["id"],
            "proposer": self.station_id,
            "proposer_jid": str(self.jid),
            "loco_id": loco["id"],
            "wagons": [
                {"id": mv["wagon"]["id"], "capacity_t": mv["wagon"]["capacity_t"],
                 "owner": mv.get("owner_station", self.station_id)}
                for mv in picked
            ],
            "wagon_moves": [
                {"owner": mv.get("owner_station", self.station_id),
                 "wagon_id": mv["wagon"]["id"],
                 "empty_path": mv["empty_path"],
                 "empty_distance_km": mv["empty_distance_km"],
                 "arrive_origin_h": mv["arrive_origin_h"],
                 "repos_cost": mv["repos_cost"]}
                for mv in picked if mv.get("empty_distance_km", 0.0) > 0.0 or mv.get("owner_station") != self.station_id
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

        print(f"[{self.station_id}] Состав {od['id']}: локо={loco['id']}, вагонов={len(offer['wagons'])}, "
              f"cap={total_cap} т, cost={offer['total_cost']}, T={offer['depart_h']}→{offer['arrive_h']}")
        return offer

    # ---------------------- запись результатов ----------------------
    async def _append_schedule(self, plan: Dict[str, Any]):
        print(f"[{self.station_id}] Записываю результат: {plan.get('order_id')}")
        oid = plan.get("order_id")
        if oid in self._written_ids:
            return
        self._written_ids.add(oid)

        # обновление календарей: освобождение после прибытия + оборот
        end = float(plan.get("arrive_h", 0.0)) + float(getattr(self, "turnaround_h", 0.5))

        def _bump(d: dict, key: str, t: float):
            d[key] = max(float(d.get(key, 0.0)), float(t))

        # локомотив — только у исполнителя
        if plan.get("proposer") == self.station_id:
            lid = plan.get("loco_id")
            if lid:
                _bump(self.free_at_loco, lid, end)

        # арендованные вагоны — у владельцев
        for mv in plan.get("wagon_moves", []):
            if mv.get("owner") == self.station_id:
                wid = mv.get("wagon_id")
                if wid:
                    _bump(self.free_at_wagon, wid, end)

        # если исполнитель и владелец части вагонов
        if plan.get("proposer") == self.station_id:
            for w in plan.get("wagons", []):
                wid = w.get("id"); owner = w.get("owner")
                if wid and owner == self.station_id:
                    _bump(self.free_at_wagon, wid, end)

        # JSON
        jpath = f"{self.output_dir}/schedule.json"
        try:
            cur = load_json(jpath); arr = cur.get("schedule", [])
        except Exception:
            arr = []
        if not any(p.get("order_id") == oid and p.get("proposer") == plan.get("proposer") for p in arr):
            arr.append(plan)
            dump_json({"schedule": arr}, jpath)

        # CSV
        cpath = f"{self.output_dir}/schedule.csv"
        os.makedirs(self.output_dir, exist_ok=True)
        header = ["order_id","proposer","loco_id","wagons","empty_distance_km","loaded_distance_km",
                  "total_time_h","total_cost","depart_h","arrive_h","empty_path","loaded_path"]

        existing = set()
        if Path(cpath).exists():
            with open(cpath, "r", encoding="utf-8", newline="") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    existing.add((r.get("order_id"), r.get("proposer")))

        key = (str(oid), str(plan.get("proposer")))
        if key not in existing:
            row = {
                "order_id": plan["order_id"],
                "proposer": plan["proposer"],
                "loco_id": plan["loco_id"],
                "wagons": ";".join(w.get("id") for w in plan["wagons"]),
                "empty_distance_km": plan["empty_distance_km"],
                "loaded_distance_km": plan["loaded_distance_km"],
                "total_time_h": plan["total_time_h"],
                "total_cost": plan["total_cost"],
                "depart_h": plan["depart_h"],
                "arrive_h": plan["arrive_h"],
                "empty_path": " - ".join(plan["empty_path"]),
                "loaded_path": " - ".join(plan["loaded_path"]),
            }
            write_header = not Path(cpath).exists()
            with open(cpath, "a", encoding="utf-8", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=header)
                if write_header:
                    wr.writeheader()
                wr.writerow(row)
