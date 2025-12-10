# offer_utils.py
import asyncio
import json
from typing import Any, Dict, List, Tuple, Optional

from spade.behaviour import CyclicBehaviour
from spade.message import Message

from agents.module.graph_utils import shortest_path, path_stats
from agents.module.message_types import (
    TYPE_WAGON_CFP,
    TYPE_LOCO_CFP,
    TYPE_ORDER_FAIL,
    TYPE_ORDER_CONFIRM,
)


def order_not_expired(agent: Any, order: Dict[str, Any]) -> bool:
    """
    Грубая проверка "срок не закончился": берём текущее логическое время станции
    как максимум занятости её ресурсов и сравниваем с latest_arrival_h (если он задан).
    """
    latest = order.get("latest_arrival_h")
    if latest is None:
        return True

    cur_loco = max(agent.free_at_loco.values(), default=0.0)
    cur_wagon = max(agent.free_at_wagon.values(), default=0.0)
    current_time_h = max(cur_loco, cur_wagon, 0.0)

    still_ok = current_time_h <= float(latest)
    if not still_ok:
        print(
            f"[{agent.station_id}] Заказ {order.get('id')} уже просрочен по дедлайну: "
            f"now≈{round(current_time_h, 2)} > latest_arrival={latest}"
        )
    return still_ok


async def handle_order_accept(
    agent: Any,
    msg: Message,
    order_id: str,
    chosen_offer: Dict[str, Any],
    ctx: CyclicBehaviour,
):
    """
    Исполнитель получил ACCEPT по заказу.
    Пересобираем оффер с учётом текущих календарей.
    Если можем выполнить — шлём CONFIRM с новым планом.
    Если нет — шлём FAIL.
    """
    try:
        od = next(o for o in agent.orders if o["id"] == order_id)
    except StopIteration:
        print(f"[{agent.station_id}] ACCEPT по неизвестному заказу {order_id}")
        reply = Message(to=str(msg.sender))
        reply.set_metadata("type", TYPE_ORDER_FAIL)
        reply.body = json.dumps({"order_id": order_id}, ensure_ascii=False)
        await ctx.send(reply)
        return

    print(
        f"[{agent.station_id}] Пересчитываю оффер для подтверждения заказа {order_id} "
        f"(оптимистичная проверка)..."
    )

    new_offer = await try_build_offer(
        agent,
        od,
        proposer_jid=str(agent.jid),
        ctx=ctx,
    )

    if not new_offer:
        print(
            f"[{agent.station_id}] Не могу подтвердить заказ {order_id} после повторной проверки. "
            f"Отправляю FAIL."
        )
        reply = Message(to=str(msg.sender))
        reply.set_metadata("type", TYPE_ORDER_FAIL)
        reply.body = json.dumps({"order_id": order_id}, ensure_ascii=False)
        await ctx.send(reply)
        return

    print(
        f"[{agent.station_id}] Подтверждаю заказ {order_id}: "
        f"cost={new_offer['total_cost']} T={new_offer['depart_h']}→{new_offer['arrive_h']}"
    )
    reply = Message(to=str(msg.sender))
    reply.set_metadata("type", TYPE_ORDER_CONFIRM)
    reply.body = json.dumps(
        {"order_id": order_id, "plan": new_offer},
        ensure_ascii=False,
    )
    await ctx.send(reply)


async def try_build_offer(
    agent: Any,
    od: Dict[str, Any],
    proposer_jid: str,
    ctx: CyclicBehaviour,
) -> Optional[Dict[str, Any]]:
    """
    Собрать состав (вагоны + тяга: свои/арендные), посчитать маршруты/стоимость/время
    и вернуть оффер или None.
    """
    origin = od["from"]
    dest = od["to"]
    cargo_type = od["type"]
    tonnage = float(od["tonnage"])

    print(
        f"[{agent.station_id}] Собираю состав под {od['id']} "
        f"({cargo_type} {tonnage} т)"
    )

    # 1. Свои вагоны: сначала уже на origin, потом удалённые
    own, remaining = select_own_wagons(agent, cargo_type, tonnage, origin)
    picked = list(own)
    latest_wagon_eta = max((w["arrive_origin_h"] for w in picked), default=0.0)
    repos_total_cost = sum(float(w["repos_cost"]) for w in picked)

    # 2. Аренда вагонов, если не хватает
    if remaining > 1e-6:
        extra, latest_extra_eta, extra_cost = await gather_wagons(agent, od, remaining, ctx)
        if not extra:
            print(
                f"[{agent.station_id}] Недостаточно вагонов "
                f"(даже с арендой) для {od['id']}"
            )
            return None
        picked.extend(extra)
        latest_wagon_eta = max(latest_wagon_eta, latest_extra_eta)
        repos_total_cost += extra_cost

    total_capacity = sum(float(w["wagon"]["capacity_t"]) for w in picked)

    # 3. Гружёный рейс origin -> dest
    loaded_path = shortest_path(agent.graph, origin, dest)
    dist_loaded, time_loaded = path_stats(agent.graph, loaded_path)

    # 4. Выбор локомотива:
    #    1) свои на origin, 2) свои удалённые, 3) аренда
    my = next(s for s in agent.stations if s["id"] == agent.station_id)
    locos = my.get("locomotives", [])

    local_own: List[dict] = []
    remote_own: List[dict] = []
    rented_list: List[dict] = []

    for l in sorted(locos, key=lambda x: float(x["cost_per_km"])):
        if float(l["max_tonnage"]) < total_capacity:
            continue

        lid = l["id"]
        free = float(
            agent.free_at_loco.get(lid, float(l.get("available_from_h", 0.0)))
        )
        start_station = agent.loco_pos.get(lid, agent.station_id)
        if start_station == origin:
            path = [origin]
            dist = 0.0
            time_empty = 0.0
        else:
            path = shortest_path(agent.graph, start_station, origin)
            dist, time_empty = path_stats(agent.graph, path)

        arrive_origin = free + time_empty
        repos_cost_loco = dist * float(l["cost_per_km"])

        cand = {
            "owner_station": agent.station_id,
            "loco": {
                "id": lid,
                "max_tonnage": float(l["max_tonnage"]),
                "cost_per_km": float(l["cost_per_km"]),
            },
            "empty_path": path,
            "empty_distance_km": round(dist, 2),
            "arrive_origin_h": round(arrive_origin, 2),
            "repos_cost": round(repos_cost_loco, 2),
        }

        if start_station == origin:
            local_own.append(cand)
        else:
            remote_own.append(cand)

    # Аренда (только чужие станции)
    rented = await gather_locos(agent, od, total_capacity, ctx)
    if rented:
        rented_list.append(rented)

    if not local_own and not remote_own and not rented_list:
        print(
            f"[{agent.station_id}] Нет подходящей тяги "
            f"({total_capacity} т) для {od['id']} (ни своей, ни арендной)."
        )
        return None

    earliest_order = float(od.get("earliest_depart_h", 0.0))
    latest = od.get("latest_arrival_h")

    def choose_best_from_group(group: List[dict]) -> Optional[tuple]:
        best_loco = None
        best_total_cost = None
        best_depart = None
        best_arrive = None

        for cand in group:
            c_loco = cand["loco"]
            cpk = float(c_loco["cost_per_km"])
            dist_empty = float(cand["empty_distance_km"])
            arrive_origin_h = float(cand["arrive_origin_h"])

            depart_h = max(0.0, earliest_order, latest_wagon_eta, arrive_origin_h)
            arrive_h = depart_h + time_loaded  # пустой ход уже в arrive_origin_h

            if latest is not None and arrive_h > float(latest):
                continue

            # Скидка 30% на пустой ход СВОЕГО локомотива
            if cand["owner_station"] == agent.station_id:
                reposition_factor = 0.7
            else:
                reposition_factor = 1.0

            cost_loco = dist_empty * cpk * reposition_factor + dist_loaded * cpk
            total_cost = cost_loco + repos_total_cost

            if best_total_cost is None or total_cost < best_total_cost:
                best_total_cost = total_cost
                best_loco = cand
                best_depart = depart_h
                best_arrive = arrive_h

        if not best_loco:
            return None
        return best_loco, best_total_cost, best_depart, best_arrive

    # 1) свои локальные
    result = choose_best_from_group(local_own)
    # 2) свои удалённые
    if result is None:
        result = choose_best_from_group(remote_own)
    # 3) аренда
    if result is None:
        result = choose_best_from_group(rented_list)

    if result is None:
        print(
            f"[{agent.station_id}] Все варианты тяги опаздывают по дедлайну "
            f"для {od['id']}."
        )
        return None

    best_loco, best_total_cost, best_depart, best_arrive = result

    loco_owner = best_loco["owner_station"]
    loco = best_loco["loco"]
    empty_path = best_loco["empty_path"]
    dist_empty = round(float(best_loco["empty_distance_km"]), 2)
    depart_h = round(best_depart, 2)
    arrive_h = round(best_arrive, 2)
    cost = round(best_total_cost, 2)
    total_time_h = round(arrive_h - depart_h, 2)

    offer = {
        "order_id": od["id"],
        "proposer": agent.station_id,
        "proposer_jid": proposer_jid,
        "loco_id": loco["id"],
        "loco_owner": loco_owner,
        "loco_empty_path": empty_path,
        "loco_empty_distance_km": dist_empty,
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
        "empty_distance_km": dist_empty,
        "loaded_path": loaded_path,
        "loaded_distance_km": round(dist_loaded, 2),
        "total_time_h": total_time_h,
        "total_cost": cost,
        "depart_h": depart_h,
        "arrive_h": arrive_h,
    }
    print(
        f"[{agent.station_id}] Состав {od['id']}: локо={loco['id']} (owner={loco_owner}), "
        f"вагонов={len(offer['wagons'])}, cap={total_capacity} т, "
        f"cost={offer['total_cost']}, T={offer['depart_h']}→{offer['arrive_h']}"
    )
    return offer


def select_own_wagons(
    agent: Any,
    cargo_type: str,
    tonnage: float,
    origin: str,
) -> Tuple[List[dict], float]:
    """
    Подбор собственных вагонов станции под груз/тоннаж:
    1) сначала те, что уже на origin
    2) затем с других станций (с перегоном).
    """
    my = next(s for s in agent.stations if s["id"] == agent.station_id)
    sel: List[dict] = []
    remaining = tonnage

    def make_entry(w, start_station: str) -> dict:
        wid = w["id"]
        free = float(
            agent.free_at_wagon.get(wid, float(w.get("available_from_h", 0.0)))
        )
        if start_station == origin:
            path = [origin]
            dist = 0.0
            time_h = 0.0
        else:
            path = shortest_path(agent.graph, start_station, origin)
            dist, time_h = path_stats(agent.graph, path)

        arrive_origin = free + time_h
        # перегон СВОЕГО вагона — на 30% дешевле аренды
        repos_cost = dist * float(agent.wagon_repos_cost_per_km) * 0.7

        return {
            "owner_station": agent.station_id,
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

    # 1-я фаза: свои вагоны уже на origin
    for w in sorted(my.get("wagons", []), key=lambda x: -float(x["capacity_t"])):
        if w["type"] != cargo_type:
            continue
        wid = w["id"]
        start_station = agent.wagon_pos.get(wid, agent.station_id)
        if start_station != origin:
            continue

        entry = make_entry(w, start_station)
        sel.append(entry)
        remaining -= float(w["capacity_t"])
        if remaining <= 1e-6:
            return sel, 0.0

    # 2-я фаза: свои вагоны с других станций (с перегоном)
    for w in sorted(my.get("wagons", []), key=lambda x: -float(x["capacity_t"])):
        if w["type"] != cargo_type:
            continue
        wid = w["id"]
        start_station = agent.wagon_pos.get(wid, agent.station_id)
        if start_station == origin:
            continue  # уже учли

        entry = make_entry(w, start_station)
        sel.append(entry)
        remaining -= float(w["capacity_t"])
        if remaining <= 1e-6:
            break

    return sel, max(0.0, remaining)


async def gather_wagons(
    agent: Any,
    order: dict,
    need_t: float,
    ctx: CyclicBehaviour,
) -> Tuple[List[dict], float, float]:
    """
    Аренда вагонов у других станций.
    Возвращает: (список выбранных вагонов, ETA на origin, суммарный repos_cost).
    """
    origin = order["from"]
    cargo_type = order["type"]
    print(f"[{agent.station_id}] Аренда: type={cargo_type}, need={need_t} т, origin={origin}")

    order_id = order["id"]

    # очищаем старые предложения
    agent.wagon_offers.pop(order_id, None)

    # Broadcast wagon_cfp
    for st in agent.stations:
        if st["id"] == agent.station_id:
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
    deadline = start + agent.WAGONS_COLLECT_S

    # ждём предложения, которые ListenControl складывает в agent.wagon_offers
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.2)
        batch = agent.wagon_offers.pop(order_id, [])
        if not batch:
            continue
        for item in batch:
            cap = float(item["wagon"]["capacity_t"])
            ppt = (float(item["repos_cost"]) + 1e-6) / max(cap, 1e-6)
            proposals.append((ppt, item))

    proposals.sort(key=lambda x: x[0])
    picked: List[dict] = []
    acc = 0.0
    latest = 0.0
    cost = 0.0

    for _, it in proposals:
        picked.append(it)
        acc += float(it["wagon"]["capacity_t"])
        latest = max(latest, float(it["arrive_origin_h"]))
        cost += float(it["repos_cost"])
        if acc >= need_t:
            break

    print(
        f"[{agent.station_id}] Аренда итог: {len(picked)} шт, "
        f"сумм.ёмкость={acc} т, ETA={round(latest,2)} ч, "
        f"repos_cost={round(cost,2)}"
    )

    if acc < need_t:
        return [], 0.0, 0.0
    return picked, latest, cost


async def gather_locos(
    agent: Any,
    order: dict,
    need_t: float,
    ctx: CyclicBehaviour,
) -> Optional[dict]:
    """
    Аренда локомотива: спрашиваем все станции (кроме себя), кто готов дать тягу.
    Возвращаем один лучший вариант или None.
    """
    origin = order["from"]
    order_id = order["id"]

    print(
        f"[{agent.station_id}] Аренда тяги: нужно тянуть {need_t} т, origin={origin}"
    )

    agent.loco_offers.pop(order_id, None)

    for st in agent.stations:
        if st["id"] == agent.station_id:
            continue
        msg = Message(to=st["jid"])
        msg.set_metadata("type", TYPE_LOCO_CFP)
        msg.body = json.dumps(
            {
                "order_id": order_id,
                "origin": origin,
                "need_tonnage": need_t,
            },
            ensure_ascii=False,
        )
        await ctx.send(msg)

    start = asyncio.get_event_loop().time()
    deadline = start + agent.WAGONS_COLLECT_S
    proposals: List[dict] = []

    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.2)
        batch = agent.loco_offers.pop(order_id, [])
        if not batch:
            continue
        proposals.extend(batch)

    if not proposals:
        print(f"[{agent.station_id}] Аренда тяги: нет предложений.")
        return None

    def loco_key(it: dict) -> tuple:
        cpk = float(it["loco"]["cost_per_km"])
        rcost = float(it.get("repos_cost", 0.0))
        return (cpk, rcost)

    proposals.sort(key=loco_key)
    best = proposals[0]

    print(
        f"[{agent.station_id}] Аренда тяги итог: loco={best['loco']['id']} "
        f"owner={best['owner_station']}, "
        f"cpk={best['loco']['cost_per_km']}, "
        f"ETA_origin={best['arrive_origin_h']}"
    )

    return best

def score_offer(offer: dict, order: dict) -> float:
    """
    Скоринг оффера для выбора победителя.
    """
    cost = float(offer["total_cost"])
    arrive = float(offer["arrive_h"])
    latest = order.get("latest_arrival_h")
    priority = float(order.get("priority", 0))

    lateness = max(0.0, arrive - float(latest)) if latest is not None else 0.0

    ALPHA = 10000.0  # penalty per hour late
    BETA = 2.0       # prefer earlier arrival
    GAMMA = 200.0    # bonus per priority unit

    return cost + ALPHA * lateness + BETA * arrive - GAMMA * priority