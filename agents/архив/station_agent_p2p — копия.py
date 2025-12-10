# from typing import Dict, Any, List
# import networkx as nx
# from spade.agent import Agent
# from spade.behaviour import OneShotBehaviour, CyclicBehaviour
# from spade.message import Message
# from pathlib import Path
# import os, csv, asyncio, json, time
# from utils.io_utils import load_json, dump_json
#
#
# CFP="CFP"; PROPOSE="PROPOSE"; ACCEPT="ACCEPT"; REJECT="REJECT"; INFORM="INFORM"
# CFP_WAGONS="CFP_WAGONS"; PROPOSE_WAGONS="PROPOSE_WAGONS"
#
# def build_graph(edges: list[dict]) -> nx.Graph:
#     g = nx.Graph()
#     for e in edges:
#         u, v = e["u"], e["v"]
#         g.add_edge(u, v, distance_km=e["distance_km"], speed_kmh=e.get("speed_kmh", 60))
#     return g
#
# def path_stats(g: nx.Graph, path: list[str]) -> tuple[float, float]:
#     d=t=0.0
#     for i in range(len(path)-1):
#         data = g.get_edge_data(path[i], path[i+1])
#         d += data["distance_km"]
#         t += data["distance_km"]/data.get("speed_kmh",60)
#     return (round(d,2), round(t,2))
#
# def shortest_path(g: nx.Graph, a: str, b: str) -> list[str]:
#     return nx.shortest_path(g, a, b, weight=lambda u, v, d: d["distance_km"])
#
# class StationAgent(Agent):
#     def __init__(self, jid, password, station_id: str, data_path: str, **kwargs):
#         super().__init__(jid, password, **kwargs)
#         self.station_id = station_id
#         self.data_path = data_path
#         self.data: Dict[str, Any] = {}
#         self.graph = None
#         self.stations: list[dict] = []
#         self.orders: list[dict] = []
#         self.rolling: dict[str, Any] = {}
#         self.offers_box: dict[str, list[dict]] = {}  # order_id -> list offers
#         self.output_dir = "output"
#         self.last_active = time.time()
#         self._written_ids = set()  # уже записанные order_id (в этой сессии)
#         self.free_at_loco = {}  # dict[loco_id] -> float (часы)
#         self.free_at_wagon = {}  # dict[wagon_id] -> float (часы)
#         self.turnaround_h = 0.5  # значение по умолчанию, можно читать из станции минимальное время оборота после рейса
#         self.wagon_repos_cost_per_km = 0.5  # ₽/км за перегон одного вагона (дефолт)
#
#         print("Я родился! - " + station_id)
#
#     class Setup(OneShotBehaviour):
#         async def run(self):
#             self.agent.data = load_json(self.agent.data_path)
#             self.agent.graph = build_graph(self.agent.data["graph"]["edges"])
#             self.agent.stations = self.agent.data["stations"]
#             self.agent.orders = [o for o in self.agent.data["orders"] if True]  # everyone sees all
#             self.agent.rolling = {
#                 s["id"]: {
#                     "locomotives": s.get("locomotives", []),
#                     "wagons": s.get("wagons", []),
#                 }
#                 for s in self.agent.stations
#             }
#
#             # ✅ 1) сначала контрольная очередь (создаст self.agent.bus)
#             lc = self.agent.ListenControl()
#             self.agent.add_behaviour(lc)
#             await asyncio.sleep(0.1)
#
#             # ✅ 2) затем приёмник CFP
#             self.agent.add_behaviour(self.agent.ListenWagons())
#             self.agent.add_behaviour(self.agent.ListenCFP())
#             await asyncio.sleep(0.1)
#
#             # ✅ 3) потом инициируем торги
#             self.agent.add_behaviour(self.agent.InitiateCFP())
#
#             self.agent.add_behaviour(self.agent.IdleKiller(timeout_s=180))
#
#             my = next(s for s in self.agent.stations if s["id"] == self.agent.station_id)
#             self.agent.turnaround_h = float(my.get("turnaround_h", 0.5))
#             self.agent.wagon_repos_cost_per_km = float(my.get("wagon_repos_cost_per_km", 0.5))
#
#             # инициализируем free_at_* из available_from_h
#             for l in my.get("locomotives", []):
#                 self.agent.free_at_loco[l["id"]] = float(l.get("available_from_h", 0.0))
#             for w in my.get("wagons", []):
#                 self.agent.free_at_wagon[w["id"]] = float(w.get("available_from_h", 0.0))
#
#     class IdleKiller(CyclicBehaviour):
#         def __init__(self, timeout_s=180):
#             super().__init__()
#             self.timeout_s = timeout_s
#
#         async def run(self):
#             import time
#             if time.time() - self.agent.last_active > self.timeout_s:
#                 print(f"[{self.agent.station_id}] Агент неактивен 3 минуты — завершаюсь.")
#                 await self.agent.stop()
#                 return
#             await asyncio.sleep(5)
#
#     class ListenWagons(CyclicBehaviour):
#         async def run(self):
#             msg = await self.receive(timeout=1)
#             if not msg:
#                 return
#             meta = msg.metadata or {}
#             if meta.get("performative") != CFP_WAGONS:
#                 # пробросим в control, если это не наш тип
#                 if hasattr(self.agent, "bus") and self.agent.bus:
#                     await self.agent.bus.put(msg)
#                 return
#
#             payload = json.loads(msg.body)
#             cargo_type = payload["type"]
#             need_t = float(payload["need_tonnage"])
#             origin = payload["origin"]
#
#             # Возьмем доступные вагоны нужного типа на этой станции по свободности
#             my = next(s for s in self.agent.stations if s["id"] == self.agent.station_id)
#             offers = []
#             acc_t = 0.0
#             for w in sorted(my.get("wagons", []), key=lambda x: -x["capacity_t"]):
#                 if w["type"] != cargo_type:
#                     continue
#                 wid = w["id"]
#                 # оценим когда вагон свободен
#                 free = float(self.agent.free_at_wagon.get(wid, float(w.get("available_from_h", 0.0))))
#                 # маршрут перегона этого вагона до origin
#                 if self.agent.station_id == origin:
#                     path = [origin];
#                     dist = 0.0;
#                     time_h = 0.0
#                 else:
#                     path = shortest_path(self.agent.graph, self.agent.station_id, origin)
#                     dist, time_h = path_stats(self.agent.graph, path)
#                 arrive_origin = free + time_h
#                 repos_cost = dist * float(self.agent.wagon_repos_cost_per_km)
#                 offers.append({
#                     "owner_station": self.agent.station_id,
#                     "wagon": {"id": wid, "type": w["type"], "capacity_t": float(w["capacity_t"])},
#                     "empty_path": path,
#                     "empty_distance_km": dist,
#                     "arrive_origin_h": round(arrive_origin, 2),
#                     "repos_cost": round(repos_cost, 2)
#                 })
#                 acc_t += float(w["capacity_t"])
#                 if acc_t >= need_t:
#                     break
#
#             reply = Message(to=str(msg.sender))
#             reply.set_metadata("performative", PROPOSE_WAGONS)
#             reply.set_metadata("for_order_id", payload["order_id"])
#             reply.body = json.dumps({"from": self.agent.station_id, "wagons": offers}, ensure_ascii=False)
#             await self.send(reply)
#
#     class InitiateCFP(OneShotBehaviour):
#         async def run(self):
#             my_orders = [o for o in self.agent.orders if o["from"] == self.agent.station_id]
#
#             # Сортируем: выше priority — раньше; затем по более раннему окну отправления
#             my_orders.sort(key=lambda o: (-int(o.get("priority", 0)),float(o.get("earliest_depart_h", 0.0))))
#
#             if not my_orders:
#                 return
#             recipients = [s for s in self.agent.stations if s["id"] != "COORD"]  # ignore any coord entry
#             for od in my_orders:
#                 # broadcast CFP
#                 print("Новый заказ от - "+ self.agent.station_id +"!")
#                 for s in recipients:
#                     msg = Message(to=s["jid"])
#                     msg.set_metadata("performative", CFP)
#                     msg.set_metadata("order_id", od["id"])
#                     msg.body = json.dumps(od, ensure_ascii=False)
#                     await self.send(msg)
#                 # collect offers for a few seconds
#                 self.agent.offers_box[od["id"]] = []
#                 await asyncio.sleep(5)
#                 offers = self.agent.offers_box.get(od["id"], [])
#                 if not offers:
#                     continue
#                 # choose best by cost
#                 best = min(offers, key=lambda x: self._score_offer(x, od))
#                 # send ACCEPT to winner, REJECT others
#                 print("Корль принял решение...")
#                 for of in offers:
#                     m = Message(to=of["proposer_jid"])
#                     m.set_metadata("order_id", od["id"])
#                     if of is best:
#                         m.set_metadata("performative", ACCEPT)
#                         m.body = json.dumps(best, ensure_ascii=False)
#                     else:
#                         m.set_metadata("performative", REJECT)
#                         m.body = json.dumps({"order_id": od["id"]}, ensure_ascii=False)
#                     await self.send(m)
#                 # publish INFORM (broadcast result)
#                 print("Корль объявил решение.")
#                 for s in recipients:
#                     im = Message(to=s["jid"])
#                     im.set_metadata("performative", INFORM)
#                     im.set_metadata("order_id", od["id"])
#                     im.body = json.dumps(best, ensure_ascii=False)
#                     await self.send(im)
#                 # write to local file
#                 await self.agent._append_schedule(best)
#
#         def _score_offer(self, offer: dict, order: dict) -> float:
#             cost = float(offer["total_cost"])
#             arrive = float(offer["arrive_h"])
#             latest = order.get("latest_arrival_h")
#             priority = float(order.get("priority", 0))
#
#             lateness = max(0.0, arrive - float(latest)) if latest is not None else 0.0
#
#             ALPHA = 10000.0  # штраф за 1 час опоздания
#             BETA = 2.0  # вес «своевременности»
#             GAMMA = 200.0  # бонус за единицу приоритета
#
#             return cost + ALPHA * lateness + BETA * arrive - GAMMA * priority
#
#     class ListenCFP(CyclicBehaviour):
#         def __init__(self, idle_timeout_s: int = 60):
#             super().__init__()
#             self.idle_timeout_s = idle_timeout_s
#             self.last_msg_time = time.time()
#         async def run(self):
#             print("Я хочу только спросить! - " + self.agent.station_id)
#             self.agent.last_active = time.time()
#             msg = await self.receive(timeout=1)
#             now = time.time()
#             # Проверяем — если нет активности дольше idle_timeout_s → завершить поведение
#             if now - self.last_msg_time > self.idle_timeout_s:
#                 print(f"[{self.agent.station_id}] ListenCFP завершает работу: 3 минуты без сообщений.")
#                 # await self.kill()
#                 await self.agent.stop()
#                 return
#
#             if not msg:
#                 # если сообщений не было, просто подождём следующую итерацию
#                 return
#             # Сброс таймера простоя
#             self.last_msg_time = now
#             self.agent.last_active = now  # синхронизируем с агентом (если используете IdleKiller)
#
#             meta = msg.metadata or {}
#             if meta.get("performative") != CFP:
#                 # requeue to control behaviour
#                 if hasattr(self.agent, "bus") and self.agent.bus:
#                     await self.agent.bus.put(msg)
#                 return
#             od = json.loads(msg.body)
#             offer = self.agent._try_build_offer(od, proposer_jid=str(self.agent.jid))
#             if not offer:
#                 return
#             reply = Message(to=str(msg.sender))
#             reply.set_metadata("performative", PROPOSE)
#             reply.set_metadata("order_id", od["id"])
#             reply.body = json.dumps(offer, ensure_ascii=False)
#             await self.send(reply)
#
#     class ListenControl(CyclicBehaviour):
#         def __init__(self):
#             super().__init__()
#             self.bus = asyncio.Queue()
#         async def on_start(self):
#             self.agent.bus = self.bus
#         async def run(self):
#             print("А можно помедленее, я записываю - " + self.agent.station_id)
#             self.agent.last_active = time.time()
#             msg = await self.bus.get()
#             meta = msg.metadata or {}
#             pf = meta.get("performative")
#
#             if pf == PROPOSE:
#                 order_id = meta.get("order_id")
#                 offer = json.loads(msg.body)
#                 self.agent.offers_box.setdefault(order_id, []).append(offer)
#
#             elif pf in (ACCEPT, REJECT, INFORM):
#                 # On ACCEPT or INFORM, append to schedule if we are involved
#                 print("я что-то записал... - " + self.agent.station_id)
#                 content = json.loads(msg.body)
#                 if pf in (ACCEPT, INFORM):
#                     await self.agent._append_schedule(content)
#
#     # Helpers
#     def _select_own_wagons(self, cargo_type: str, tonnage: float):
#         my = next(s for s in self.stations if s["id"] == self.station_id)
#         sel = [];
#         remaining = tonnage
#         for w in sorted(my.get("wagons", []), key=lambda x: -x["capacity_t"]):
#             if w["type"] != cargo_type: continue
#             sel.append({"owner_station": self.station_id,
#                         "wagon": {"id": w["id"], "type": w["type"], "capacity_t": float(w["capacity_t"])},
#                         "empty_path": [self.station_id], "empty_distance_km": 0.0,
#                         "arrive_origin_h": float(
#                             self.free_at_wagon.get(w["id"], float(w.get("available_from_h", 0.0)))),
#                         "repos_cost": 0.0})
#             remaining -= float(w["capacity_t"])
#             if remaining <= 0: break
#         return sel, max(0.0, remaining)
#
#     async def _gather_wagons(self, order: dict, need_t: float) -> tuple[list[dict], float, float]:
#         """Запросить вагоны у сети. Возвращает (список_вагонов, latest_arrive_origin, total_repos_cost)."""
#         origin = order["from"];
#         cargo_type = order["type"]
#         # broadcast CFP_WAGONS
#         for st in self.stations:
#             msg = Message(to=st["jid"])
#             msg.set_metadata("performative", CFP_WAGONS)
#             msg.body = json.dumps({"order_id": order["id"], "origin": origin, "type": cargo_type,
#                                    "need_tonnage": need_t}, ensure_ascii=False)
#             await self.send(msg)
#
#         # collect PROPOSE_WAGONS
#         proposals = []
#         deadline = asyncio.get_event_loop().time() + 5.0
#         while asyncio.get_event_loop().time() < deadline:
#             msg = await self.receive(timeout=0.5)
#             if not msg:
#                 continue
#             meta = msg.metadata or {}
#             if meta.get("performative") != PROPOSE_WAGONS:
#                 # пусть остальная почта уйдет в control
#                 if hasattr(self, "bus") and self.bus:
#                     await self.bus.put(msg)
#                 continue
#             payload = json.loads(msg.body)
#             for item in payload.get("wagons", []):
#                 # метрика «цена за тонну»
#                 cap = float(item["wagon"]["capacity_t"])
#                 price_per_t = (float(item["repos_cost"]) + 1e-6) / cap
#                 proposals.append((price_per_t, item))
#
#         # набираем корзину
#         proposals.sort(key=lambda x: x[0])
#         picked = [];
#         acc = 0.0;
#         latest = 0.0;
#         cost = 0.0
#         for _, it in proposals:
#             picked.append(it)
#             acc += float(it["wagon"]["capacity_t"])
#             latest = max(latest, float(it["arrive_origin_h"]))
#             cost += float(it["repos_cost"])
#             if acc >= need_t:
#                 break
#         if acc < need_t:
#             return [], 0.0, 0.0  # недостаточно вагонов в сети
#         return picked, latest, cost
#
#     def _try_build_offer(self, od: Dict[str, Any], proposer_jid: str) -> Dict[str, Any] | None:
#         """
#         Формирует предложение перевозки одним составом со СВОЕЙ станции:
#         - выбирает вагоны нужного типа и суммарной вместимости >= tonnage
#         - выбирает локомотив по тяге
#         - учитывает доступность ресурсов (free_at_loco/free_at_wagon) и earliest_depart_h заказа
#         - считает маршруты: пустой (до origin) и гружёный (origin -> dest)
#         - возвращает оффер или None, если состав собрать нельзя
#         """
#         origin = od["from"]
#         dest = od["to"]
#         cargo_type = od["type"]
#         tonnage = float(od["tonnage"])
#
#         # --- Вагоны на ЭТОЙ станции ---
#         st = next(s for s in self.stations if s["id"] == self.station_id)
#         wagons = st.get("wagons", [])
#         locos = st.get("locomotives", [])
#
#         sel_w: list[dict] = []
#         remaining = tonnage
#         for w in sorted(wagons, key=lambda x: -float(x["capacity_t"])):
#             if w["type"] != cargo_type:
#                 continue
#             sel_w.append({"id": w["id"], "capacity_t": float(w["capacity_t"])})
#             remaining -= float(w["capacity_t"])
#             if remaining <= 1e-6:
#                 break
#
#         if remaining > 1e-6:
#             # На одной станции не хватает вагонов — в текущей версии отказываемся
#             return None
#
#         total_cap = sum(w["capacity_t"] for w in sel_w)
#
#         # --- Локомотив по тяге ---
#         loco = None
#         for l in sorted(locos, key=lambda x: float(x["cost_per_km"])):
#             if float(l["max_tonnage"]) >= total_cap:
#                 loco = l
#                 break
#         if not loco:
#             return None
#
#         # --- Доступность ресурсов и earliest_depart_h заказа ---
#         free_loco = float(self.free_at_loco.get(loco["id"], 0.0))
#         free_wagons = max(float(self.free_at_wagon.get(w["id"], 0.0)) for w in sel_w) if sel_w else 0.0
#         earliest_order = float(od.get("earliest_depart_h", 0.0))
#         depart_h = max(0.0, free_loco, free_wagons, earliest_order)
#
#         # --- Пустой перегон (наш поезд без груза до origin) ---
#         if self.station_id == origin:
#             empty_path = [origin]
#             dist_empty, time_empty = 0.0, 0.0
#         else:
#             empty_path = shortest_path(self.graph, self.station_id, origin)
#             dist_empty, time_empty = path_stats(self.graph, empty_path)
#
#         # --- Гружёный перегон (origin -> dest) ---
#         loaded_path = shortest_path(self.graph, origin, dest)
#         dist_loaded, time_loaded = path_stats(self.graph, loaded_path)
#
#         # --- Время прибытия ---
#         arrive_origin_h = depart_h + time_empty
#         arrive_h = arrive_origin_h + time_loaded
#
#         # --- (опц.) жёсткий дедлайн: если есть latest_arrival_h и мы опаздываем, можно отказаться ---
#         latest = od.get("latest_arrival_h", None)
#         if latest is not None and arrive_h > float(latest):
#             # Можно вернуть None (отказаться), либо оставить предложение, если инициатор будет сам штрафовать.
#             # Здесь выберем строгий вариант — откажемся:
#             return None
#
#         # --- Стоимость (наивно: локо * (пустой + гружёный)) ---
#         cost = (dist_empty + dist_loaded) * float(loco["cost_per_km"])
#
#         offer = {
#             "order_id": od["id"],
#             "proposer": self.station_id,
#             "proposer_jid": str(self.jid),
#             "loco_id": loco["id"],
#             "wagons": sel_w,  # [{id, capacity_t}]
#             "empty_path": empty_path,
#             "empty_distance_km": round(dist_empty, 2),
#             "loaded_path": loaded_path,
#             "loaded_distance_km": round(dist_loaded, 2),
#             "total_time_h": round(time_empty + time_loaded, 2),
#             "total_cost": round(cost, 2),
#             "depart_h": round(depart_h, 2),
#             "arrive_h": round(arrive_h, 2),
#         }
#         return offer
#
#     async def _append_schedule(self, plan: Dict[str, Any]):
#         # append to CSV/JSON locally
#         print("Записал в книжечку - " + self.station_id)
#         oid = plan.get("order_id")
#         if oid in self._written_ids:
#             return  # уже записали в этой сессии
#         self._written_ids.add(oid)
#
#         # ---- ОБНОВЛЕНИЕ КАЛЕНДАРЕЙ (вариант с арендой вагонов) ----
#         # ресурсы освобождаются в момент прибытия + оборот
#         end = float(plan.get("arrive_h", 0.0)) + float(getattr(self, "turnaround_h", 0.5))
#
#         # на всякий случай — гарантируем словари
#         if not hasattr(self, "free_at_loco"):
#             self.free_at_loco = {}
#         if not hasattr(self, "free_at_wagon"):
#             self.free_at_wagon = {}
#
#         def _bump(d: dict, key: str, t: float):
#             d[key] = max(float(d.get(key, 0.0)), float(t))
#
#         # 1) Исполнитель (перевозчик) освобождает свой локомотив после рейса
#         if plan.get("proposer") == self.station_id:
#             lid = plan.get("loco_id")
#             if lid:
#                 _bump(self.free_at_loco, lid, end)
#
#         # 2) Владельцы вагонов: обновляем занятость СВОИХ вагонов по wagon_moves
#         #    (этот блок сработает у каждого агента-владельца при получении INFORM)
#         for mv in plan.get("wagon_moves", []):
#             if mv.get("owner") == self.station_id:
#                 wid = mv.get("wagon_id")
#                 if wid:
#                     _bump(self.free_at_wagon, wid, end)
#
#         # 3) Доп. защита: если в plan["wagons"] проставлены owner,
#         #    и мы одновременно являемся исполнителем и владельцем части вагонов,
#         #    обновим их тоже (без дублей).
#         if plan.get("proposer") == self.station_id:
#             for w in plan.get("wagons", []):
#                 wid = w.get("id")
#                 owner = w.get("owner")
#                 if wid and owner == self.station_id:
#                     _bump(self.free_at_wagon, wid, end)
#
#         # --- JSON (надёжная проверка по order_id + proposer) ---
#         jpath = f"{self.output_dir}/schedule.json"
#         try:
#             cur = load_json(jpath)
#             arr = cur.get("schedule", [])
#         except Exception:
#             arr = []
#
#         if not any(p.get("order_id") == oid and p.get("proposer") == plan.get("proposer") for p in arr):
#             arr.append(plan)
#             dump_json({"schedule": arr}, jpath)
#
#         # --- CSV (проверим, нет ли уже order_id в файле) ---
#         cpath = f"{self.output_dir}/schedule.csv"
#         os.makedirs(self.output_dir, exist_ok=True)
#         header = ["order_id", "proposer", "loco_id", "wagons", "empty_distance_km",
#                   "loaded_distance_km", "total_time_h", "total_cost", "depart_h",
#                   "arrive_h", "empty_path", "loaded_path"]
#
#         existing = set()
#         if Path(cpath).exists():
#             with open(cpath, "r", encoding="utf-8", newline="") as f:
#                 rd = csv.DictReader(f)
#                 for r in rd:
#                     existing.add((r.get("order_id"), r.get("proposer")))
#
#         key = (str(oid), str(plan.get("proposer")))
#         if key not in existing:
#             row = {
#                 "order_id": plan["order_id"],
#                 "proposer": plan["proposer"],
#                 "loco_id": plan["loco_id"],
#                 "wagons": ";".join(w["id"] for w in plan["wagons"]),
#                 "empty_distance_km": plan["empty_distance_km"],
#                 "loaded_distance_km": plan["loaded_distance_km"],
#                 "total_time_h": plan["total_time_h"],
#                 "total_cost": plan["total_cost"],
#                 "depart_h": plan["depart_h"],
#                 "arrive_h": plan["arrive_h"],
#                 "empty_path": " - ".join(plan["empty_path"]),
#                 "loaded_path": " - ".join(plan["loaded_path"]),
#             }
#             write_header = not Path(cpath).exists()
#             with open(cpath, "a", encoding="utf-8", newline="") as f:
#                 wr = csv.DictWriter(f, fieldnames=header)
#                 if write_header:
#                     wr.writeheader()
#                 wr.writerow(row)
#
#     async def setup(self):
#         self.add_behaviour(self.Setup())


from typing import Any, Dict, List, Optional

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
    def __init__(self, jid, password, station_id: str, data_path: str, **kwargs):
        super().__init__(jid, password, **kwargs)
        self.station_id = station_id
        self.data_path = data_path

        self.data: Dict[str, Any] = {}
        self.graph: nx.Graph | None = None
        self.stations: list[dict] = []
        self.orders: list[dict] = []
        self.rolling: dict[str, Any] = {}
        self.offers_box: dict[str, list[dict]] = {}  # order_id -> offers

        self.output_dir = "output"
        self.last_active = time.time()
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

        self.wagon_location: dict[str, str] = {}  # wagon_id -> station_id
        self.loco_location: dict[str, str] = {}  # loco_id  -> station_id

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
            self.agent.add_behaviour(self.agent.ListenWagons())
            self.agent.add_behaviour(self.agent.ListenCFP(idle_timeout_s=120))
            await asyncio.sleep(0.1)

            # 3) инициатор торгов (по своим заказам)
            self.agent.add_behaviour(self.agent.InitiateCFP())

            # 4) сторожок неактивности агента
            self.agent.add_behaviour(self.agent.IdleKiller(timeout_s=120))

            # параметры станции
            my = next(s for s in self.agent.stations if s["id"] == self.agent.station_id)
            self.agent.turnaround_h = float(my.get("turnaround_h", 0.5))
            self.agent.wagon_repos_cost_per_km = float(my.get("wagon_repos_cost_per_km", 0.5))

            # инициализация календарей
            for l in my.get("locomotives", []):
                self.agent.free_at_loco[l["id"]] = float(l.get("available_from_h", 0.0))
            for w in my.get("wagons", []):
                self.agent.free_at_wagon[w["id"]] = float(w.get("available_from_h", 0.0))

            # начальные локации: что в данных у станции — стоит на этой станции
            for s in self.agent.stations:
                sid = s["id"]
                for w in s.get("wagons", []):
                    self.agent.wagon_location[w["id"]] = sid
                for l in s.get("locomotives", []):
                    self.agent.loco_location[l["id"]] = sid

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

                cur_loc = self.agent.wagon_location.get(wid, self.agent.station_id)
                # предлагаем только вагоны, которые физически у нас
                # if cur_loc != self.agent.station_id:
                #     continue

                free = float(self.agent.free_at_wagon.get(wid, float(w.get("available_from_h", 0.0))))
                if cur_loc == origin:
                    path = [origin];
                    dist = 0.0;
                    time_h = 0.0
                else:
                    path = shortest_path(self.agent.graph, cur_loc, origin)
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
                print(f"[{self.agent.station_id}] Рассылаю CFP по {od['id']} получателям: "
                      f"{', '.join(s['id'] for s in recipients)}")

                print(f"[{self.agent.station_id}] Новый заказ {od['id']}: {od['from']}→{od['to']} "
                      f"{od['type']} {od['tonnage']} т (prio={od.get('priority',0)})")

                for s in recipients:
                    msg = Message(to=s["jid"])
                    msg.set_metadata("performative", CFP)
                    msg.set_metadata("order_id", od["id"])
                    msg.body = json.dumps(od, ensure_ascii=False)
                    print(f"[{self.agent.station_id}] → CFP {od['id']} → {s['id']} ({s['jid']})")
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
                    print(f"[{self.agent.station_id}] Нет офферов по {od['id']} — пробую локальный fallback.")
                    try:
                        local_offer = await self.agent._try_build_offer(od, proposer_jid=str(self.agent.jid), beh=self)
                    except Exception as e:
                        local_offer = None
                        print(f"[{self.agent.station_id}] Fallback для {od['id']} упал: {e!r}")

                    if local_offer:
                        print(f"[{self.agent.station_id}] Fallback-оффер готов: cost={local_offer['total_cost']} "
                              f"T={local_offer['depart_h']}→{local_offer['arrive_h']}")
                        offers = [local_offer]
                        self.agent.offers_box[od["id"]] = [local_offer]
                    else:
                        print(f"[{self.agent.station_id}] Не удалось выполнить {od['id']} даже с fallback — пропускаю.")
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
        def __init__(self, idle_timeout_s: int = 120):
            super().__init__()
            self.idle_timeout_s = idle_timeout_s
            self.last_msg_time = time.time()

        async def run(self):
            print(f"[{self.agent.station_id}] ListenCFP: жду...")
            msg = await self.receive(timeout=1)
            now = time.time()

            if now - self.last_msg_time > self.idle_timeout_s:
                print(f"[{self.agent.station_id}] ListenCFP: {self.idle_timeout_s} сек тишины — завершаю поведение.")
                self.kill()
                return

            if not msg:
                return

            self.last_msg_time = now
            self.agent.last_active = now

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
        """Вернёт (список_moves, суммарная_ёмкость, latest_arrive_origin). Берём ТОЛЬКО вагоны,
        которыми владеет станция и которые физически стоят у нас (по wagon_location)."""
        my = next(s for s in self.stations if s["id"] == self.station_id)
        picked = []
        acc = 0.0
        latest = 0.0

        for w in sorted(my.get("wagons", []), key=lambda x: -x["capacity_t"]):
            if w["type"] != cargo_type:
                continue
            wid = w["id"]

            # берем только те, что реально у нас находятся
            cur_loc = self.wagon_location.get(wid, self.station_id)
            if cur_loc != self.station_id:
                continue

            free = float(self.free_at_wagon.get(wid, float(w.get("available_from_h", 0.0))))

            # путь от current_location вагона до origin (а не от self.station_id — вдруг он уже не здесь)
            if cur_loc == origin:
                path = [origin];
                dist = 0.0;
                time_h = 0.0
            else:
                path = shortest_path(self.graph, cur_loc, origin)
                dist, time_h = path_stats(self.graph, path)

            arrive_origin = free + time_h
            picked.append({
                "owner_station": self.station_id,
                "wagon": {"id": wid, "type": w["type"], "capacity_t": float(w["capacity_t"])},
                "empty_path": path,
                "empty_distance_km": dist,
                "arrive_origin_h": round(arrive_origin, 2),
                "repos_cost": 0.0
            })
            acc += float(w["capacity_t"])
            latest = max(latest, arrive_origin)
            if acc >= tonnage:
                break

        return picked, acc, latest

    async def _gather_wagons(self, order: dict, need_t: float, beh) -> tuple[list[dict], float, float]:
        """
        Собрать недостающие вагоны из сети через beh.send/beh.receive.

        Возвращает:
          picked: список нормализованных словарей по каждому арендованному вагону:
            {
              "id", "capacity_t", "type", "owner",
              "cur_loc", "empty_path", "empty_distance_km",
              "arrive_origin_h", "repos_cost"
            }
          latest_eta: максимальное время прибытия арендованных вагонов в origin
          total_repos_cost: суммарная стоимость перегона/аренды
        """
        origin = order["from"];
        cargo_type = order["type"]
        print(f"[{self.station_id}] Аренда: type={cargo_type}, need={float(need_t)} т, origin={origin}")

        # ── разослать CFP_WAGONS всем станциям ───────────────────────────────────
        for st in self.stations:
            try:
                msg = Message(to=st["jid"])
                msg.set_metadata("performative", CFP_WAGONS)
                msg.body = json.dumps(
                    {"order_id": order["id"], "origin": origin, "type": cargo_type, "need_tonnage": float(need_t)},
                    ensure_ascii=False
                )
                await beh.send(msg)
            except Exception as e:
                print(f"[{self.station_id}] WARN: не смог отправить CFP_WAGONS → {st.get('id')}: {e!r}")

        # ── собрать PROPOSE_WAGONS и нормализовать элементы ──────────────────────
        proposals: list[dict] = []
        loop = asyncio.get_event_loop()
        deadline = loop.time() + float(getattr(self, "WAGONS_COLLECT_S", 3.0))

        while loop.time() < deadline:
            msg = await beh.receive(timeout=0.5)
            if not msg:
                continue

            meta = msg.metadata or {}
            if meta.get("performative") != PROPOSE_WAGONS:
                # чужие сообщения не теряем
                if hasattr(self, "bus") and self.bus:
                    await self.bus.put(msg)
                continue

            try:
                payload = json.loads(msg.body)
            except Exception as e:
                print(f"[{self.station_id}] WARN: не смог распарсить PROPOSE_WAGONS: {e!r}")
                continue

            for raw in payload.get("wagons", []):
                # допускаем 2 формата: item = {"wagon": {...}, ...} ИЛИ item = {...} сразу вагон
                w = raw.get("wagon", raw)

                wid = w.get("id")
                if not wid:
                    continue

                # тип берём из сообщения, иначе из индекса
                wtype = w.get("type") or self.wagon_index.get(wid, {}).get("type")
                if wtype != cargo_type:
                    print(f"[{self.station_id}] SKIP {wid}: тип {wtype} != {cargo_type}")
                    continue

                cap = float(w.get("capacity_t", 0.0))
                if cap <= 0.0:
                    print(f"[{self.station_id}] SKIP {wid}: нулевая ёмкость")
                    continue

                # попытка взять cur_loc/путь/стоимость из ответа; если нет — досчитаем
                cur_loc = (raw.get("cur_loc") or w.get("location") or
                           self.wagon_location.get(wid) or
                           self.wagon_index.get(wid, {}).get("location") or
                           self.wagon_index.get(wid, {}).get("owner_station") or
                           w.get("owner") or self.station_id)

                empty_path = raw.get("empty_path")
                empty_dist = raw.get("empty_distance_km")
                arrive_origin_h = raw.get("arrive_origin_h")
                repos_cost = raw.get("repos_cost")

                # если каких-то полей нет, аккуратно досчитываем
                if empty_path is None or empty_dist is None or arrive_origin_h is None or repos_cost is None:
                    try:
                        if cur_loc == origin:
                            calc_path = [origin]
                            dist, time_h = 0.0, 0.0
                        else:
                            calc_path = shortest_path(self.graph, cur_loc, origin)
                            dist, time_h = path_stats(self.graph, calc_path)

                        free_at = float(self.free_at_wagon.get(wid, 0.0))
                        now_h = float(getattr(self, "now_h", 0.0))
                        eta = max(free_at, now_h) + time_h

                        # цена перегона — по тарифу аренды за км
                        rent_km = float(getattr(self, "rent_cost_per_km",
                                                getattr(self, "rent_wagon_cost_per_km", 0.5)))
                        rcost = dist * rent_km

                        empty_path = empty_path or calc_path
                        empty_dist = float(empty_dist) if empty_dist is not None else float(dist)
                        arrive_origin_h = float(arrive_origin_h) if arrive_origin_h is not None else float(eta)
                        repos_cost = float(repos_cost) if repos_cost is not None else float(rcost)
                    except Exception as e:
                        print(
                            f"[{self.station_id}] SKIP {wid}: не удалось вычислить перегон {cur_loc}->{origin}: {e!r}")
                        continue

                # метрика для ранжирования: стоимость репозиционирования на тонну + лёгкий приоритет более раннего ETA
                price_per_t = (float(repos_cost) + 1e-6) / cap + float(arrive_origin_h) * 1e-6

                proposals.append({
                    "id": wid,
                    "capacity_t": cap,
                    "type": wtype,
                    "owner": w.get("owner"),
                    "cur_loc": cur_loc,
                    "empty_path": empty_path,
                    "empty_distance_km": float(empty_dist),
                    "arrive_origin_h": float(arrive_origin_h),
                    "repos_cost": float(repos_cost),
                    "_rank": price_per_t,
                })

        # ── дедупликация по вагону: оставляем лучший вариант ─────────────────────
        best_by_wid: dict[str, dict] = {}
        for it in proposals:
            wid = it["id"]
            prev = best_by_wid.get(wid)
            if (prev is None) or (it["_rank"] < prev["_rank"]):
                best_by_wid[wid] = it

        deduped = list(best_by_wid.values())
        # более дёшево и чуть раньше → выше
        deduped.sort(key=lambda x: (x["_rank"], x["arrive_origin_h"]))

        # ── выбираем до покрытия потребности need_t ───────────────────────────────
        picked: list[dict] = []
        acc = 0.0
        latest = 0.0
        total_cost = 0.0

        for it in deduped:
            picked.append({
                "id": it["id"],
                "capacity_t": it["capacity_t"],
                "type": it["type"],
                "owner": it["owner"],
                "cur_loc": it["cur_loc"],
                "empty_path": it["empty_path"],
                "empty_distance_km": it["empty_distance_km"],
                "arrive_origin_h": round(it["arrive_origin_h"], 2),
                "repos_cost": round(it["repos_cost"], 2),
            })
            acc += float(it["capacity_t"])
            latest = max(latest, float(it["arrive_origin_h"]))
            total_cost += float(it["repos_cost"])

            if acc >= need_t:
                break

        print(f"[{self.station_id}] Аренда итог: {len(picked)} шт, сумм.ёмкость={acc} т, "
              f"ETA={round(latest, 2)} ч, repos_cost={round(total_cost, 2)}")

        if acc < need_t:
            # не удалось покрыть потребность — возвращаем пусто
            return [], 0.0, 0.0

        return picked, latest, total_cost

    # ---------------------- построение оффера ----------------------
    async def _try_build_offer(self, od: Dict[str, Any], proposer_jid: str, beh: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        origin = od["from"]; dest = od["to"]; cargo_type = od["type"]; tonnage = float(od["tonnage"])
        # найдём «свою» станцию
        st = next(s for s in self.stations if s["id"] == self.station_id)
        wagons = st.get("wagons", []); locos = st.get("locomotives", [])

        print(f"[{self.station_id}] Собираю состав под {od['id']} ({cargo_type} {tonnage} т)")

        # 0) локальный выбор своих вагонов нужного типа (крупные сначала)
        sel_w: List[Dict[str, Any]] = []
        remaining = tonnage
        for w in sorted(wagons, key=lambda x: -float(x["capacity_t"])):
            if w["type"] != cargo_type:
                continue
            sel_w.append({"id": w["id"], "capacity_t": float(w["capacity_t"])})
            remaining -= float(w["capacity_t"])
            if remaining <= 0:
                break

        total_cap = sum(float(w["capacity_t"]) for w in sel_w)
        if total_cap < tonnage:
            print(f"[{self.station_id}] Не хватает своих вагонов: есть {total_cap}, нужно {tonnage}")

        # 1) локомотив по минимальной стоимости, но с достаточной тягой
        loco = None
        for l in sorted(locos, key=lambda x: float(x["cost_per_km"])):
            if float(l["max_tonnage"]) >= max(total_cap, tonnage):
                loco = l; break
        if not loco:
            print(f"[{self.station_id}] Нет локо под {od['id']} (недостаточная тяга)")
            return None

        # 2) доступность ресурсов (free_at) и ранний старт
        free_loco = float(self.free_at_loco.get(loco["id"], 0.0))
        free_wagons = max([float(self.free_at_wagon.get(w["id"], 0.0)) for w in sel_w], default=0.0)
        earliest_depart = max(0.0, free_loco, free_wagons, float(od.get("earliest_depart_h", 0.0)))

        # 3) пустой перегон от текущей станции до origin
        if self.station_id == origin:
            empty_path = [origin]; dist_empty, time_empty = 0.0, 0.0
        else:
            empty_path = shortest_path(self.graph, self.station_id, origin)
            dist_empty, time_empty = path_stats(self.graph, empty_path)

        # 4) гружёный путь origin → dest
        loaded_path = shortest_path(self.graph, origin, dest)
        dist_loaded, time_loaded = path_stats(self.graph, loaded_path)

        # 5) недостающее — арендой
        missing = max(0.0, tonnage - total_cap)
        wagon_moves: List[dict] = []
        extra_eta = 0.0
        extra_cost = 0.0
        if missing > 0.0:
            extra, extra_eta, extra_cost = await self._gather_wagons(od, missing, beh)
            if sum(float(x["capacity_t"]) for x in extra) + total_cap < tonnage:
                print(f"[{self.station_id}] Недостаточно вагонов (даже с арендой) для {od['id']}")
                return None
            # добавляем арендованные вагоны (тип проверим ниже)
            for x in extra:
                sel_w.append({"id": x["id"], "capacity_t": float(x["capacity_t"])})
            wagon_moves = extra

        # 6) финальные времена (нельзя стартовать до прибытия арендованных в origin)
        depart_h = max(earliest_depart, float(extra_eta) if wagon_moves else earliest_depart)
        arrive_origin_h = depart_h + time_empty
        arrive_h = arrive_origin_h + time_loaded

        # 7) стоимость: пробеги локомотива + репозиционирование арендованных вагонов
        cost = dist_empty*float(loco["cost_per_km"]) + dist_loaded*float(loco["cost_per_km"]) + float(extra_cost)

        # 8) ВАЛИДАЦИИ
        # 8.1) типы всех вагонов должны совпадать с типом заказа
        bad_types = [w["id"] for w in sel_w if (self.wagon_index.get(w["id"], {}).get("type") or cargo_type) != cargo_type]
        if bad_types:
            print(f"[{self.station_id}] ОТКЛОНЯЮ план {od['id']}: несоответствие типа вагонов {bad_types} требуемому {cargo_type}.")
            return None

        # 8.2) дедлайн
        latest = od.get("latest_arrival_h")
        if latest is not None and arrive_h > float(latest):
            print(f"[{self.station_id}] ОТКЛОНЯЮ {od['id']}: дедлайн {latest} < ETA {arrive_h:.2f}.")
            return None

        # 9) финальный оффер
        offer = {
            "order_id": od["id"],
            "proposer": self.station_id,
            "proposer_jid": str(self.jid),
            "loco_id": loco["id"],
            "wagons": sel_w,
            "empty_path": empty_path,
            "empty_distance_km": dist_empty,
            "loaded_path": loaded_path,
            "loaded_distance_km": dist_loaded,
            "total_time_h": round(time_empty + time_loaded, 2),
            "total_cost": round(cost, 2),
            "depart_h": round(depart_h, 2),
            "arrive_h": round(arrive_h, 2),
            "wagon_moves": wagon_moves,     # важно для _append_schedule и восстановления состояний
        }

        print(f"[{self.station_id}] Состав {od['id']}: локо={loco['id']}, вагонов={len(sel_w)}, "
              f"cap={sum(w['capacity_t'] for w in sel_w)} т, cost={offer['total_cost']}, "
              f"T={offer['depart_h']}→{offer['arrive_h']}")
        return offer

    # async def _try_build_offer(self, od: Dict[str, Any], proposer_jid: str, beh) -> Dict[str, Any] | None:
    #     origin = od["from"]; dest = od["to"]; cargo_type = od["type"]; tonnage = float(od["tonnage"])
    #     print(f"[{self.station_id}] Собираю состав под {od['id']} ({cargo_type} {tonnage} т)")
    #
    #     # свои вагоны
    #     own, own_cap, own_latest = self._select_own_wagons(cargo_type, tonnage, origin)
    #
    #     picked = list(own)
    #     repos_total_cost = 0.0
    #     latest_wagon_eta = own_latest
    #
    #     # арендуем недостающее
    #     if own_cap + 1e-6 < tonnage:
    #         need = tonnage - own_cap
    #         extra, latest_extra_eta, extra_cost = await self._gather_wagons(od, need, beh)
    #         if not extra:
    #             print(f"[{self.station_id}] Недостаточно вагонов (даже с арендой) для {od['id']}")
    #             return None
    #         picked.extend(extra)
    #         repos_total_cost += extra_cost
    #         latest_wagon_eta = max(latest_wagon_eta, latest_extra_eta)
    #
    #     total_cap = sum(x["wagon"]["capacity_t"] for x in picked)
    #
    #     # локомотив
    #     st = next(s for s in self.stations if s["id"] == self.station_id)
    #     locos = st.get("locomotives", [])
    #     loco = None
    #     for l in sorted(locos, key=lambda x: float(x["cost_per_km"])):
    #         if float(l["max_tonnage"]) >= total_cap:
    #             loco = l; break
    #     if not loco:
    #         print(f"[{self.station_id}] Нет подходящей тяги ({total_cap} т) для {od['id']}")
    #         return None
    #
    #     # доступность и окно заказа
    #     free_loco = float(self.free_at_loco.get(loco["id"], float(loco.get("available_from_h", 0.0))))
    #     earliest_order = float(od.get("earliest_depart_h", 0.0))
    #     depart_h = max(0.0, free_loco, latest_wagon_eta, earliest_order)
    #
    #     # маршруты
    #     if self.station_id == origin:
    #         empty_path = [origin]; dist_empty = 0.0; time_empty = 0.0
    #     else:
    #         empty_path = shortest_path(self.graph, self.station_id, origin)
    #         dist_empty, time_empty = path_stats(self.graph, empty_path)
    #
    #     loaded_path = shortest_path(self.graph, origin, dest)
    #     dist_loaded, time_loaded = path_stats(self.graph, loaded_path)
    #
    #     arrive_origin_h = depart_h + time_empty
    #     arrive_h = arrive_origin_h + time_loaded
    #
    #     # дедлайн
    #     latest = od.get("latest_arrival_h")
    #     if latest is not None and arrive_h > float(latest):
    #         print(f"[{self.station_id}] Опоздание по дедлайну для {od['id']}: ETA={round(arrive_h,2)} > {latest}")
    #         return None
    #
    #     # стоимость: локо (пустой+гружёный) + репозиция арендованных вагонов
    #     loco_cost = (dist_empty + dist_loaded) * float(loco["cost_per_km"])
    #     cost = loco_cost + repos_total_cost
    #
    #     offer = {
    #         "order_id": od["id"],
    #         "proposer": self.station_id,
    #         "proposer_jid": str(self.jid),
    #         "loco_id": loco["id"],
    #         "wagons": [
    #             {"id": mv["wagon"]["id"], "capacity_t": mv["wagon"]["capacity_t"],
    #              "owner": mv.get("owner_station", self.station_id)}
    #             for mv in picked
    #         ],
    #         "wagon_moves": [
    #             {"owner": mv.get("owner_station", self.station_id),
    #              "wagon_id": mv["wagon"]["id"],
    #              "empty_path": mv["empty_path"],
    #              "empty_distance_km": mv["empty_distance_km"],
    #              "arrive_origin_h": mv["arrive_origin_h"],
    #              "repos_cost": mv["repos_cost"]}
    #             for mv in picked if mv.get("empty_distance_km", 0.0) > 0.0 or mv.get("owner_station") != self.station_id
    #         ],
    #         "empty_path": empty_path,
    #         "empty_distance_km": round(dist_empty, 2),
    #         "loaded_path": loaded_path,
    #         "loaded_distance_km": round(dist_loaded, 2),
    #         "total_time_h": round(time_empty + time_loaded, 2),
    #         "total_cost": round(cost, 2),
    #         "depart_h": round(depart_h, 2),
    #         "arrive_h": round(arrive_h, 2),
    #     }
    #
    #     print(f"[{self.station_id}] Состав {od['id']}: локо={loco['id']}, вагонов={len(offer['wagons'])}, "
    #           f"cap={total_cap} т, cost={offer['total_cost']}, T={offer['depart_h']}→{offer['arrive_h']}")
    #     return offer

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

          # --- RUNTIME-ВАЛИДАЦИЯ ПЕРЕД ЗАПИСЬЮ ---
        try:
            order_id = plan["order_id"]
            order_type = (self.order_index[order_id]["type"]
            if hasattr(self, "order_index") and order_id in self.order_index else None)
            bad = []
            if order_type:
                for w in plan.get("wagons", []):
                    wid = w["id"]
                    wtype = self.wagon_index.get(wid, {}).get("type")

                    if wtype and wtype != order_type:
                        bad.append((wid, wtype))
            if bad:
                print(f"[{self.station_id}] BUG: несоответствие типа вагона заказу {order_id}: {bad}. Не записываю план.")
                return
        except Exception as e:
            print(f"[{self.station_id}] WARN: не удалось выполнить runtime-проверку типов перед записью: {e!r}")

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

        # ---- ОБНОВЛЯЕМ ФАКТИЧЕСКИЕ ЛОКАЦИИ ПОСЛЕ РЕЙСА ----
        try:
            dest_station = plan["loaded_path"][-1]  # последняя станция маршрута с грузом
        except Exception:
            dest_station = None

        if dest_station:
            # локомотив исполнителя переезжает в точку прибытия
            if plan.get("proposer") == self.station_id:
                lid = plan.get("loco_id")
                if lid:
                    self.loco_location[lid] = dest_station

            # все вагоны из плана оказываются на станции назначения (владельцы сохраняются)
            for w in plan.get("wagons", []):
                wid = w.get("id")
                if wid:
                    self.wagon_location[wid] = dest_station
