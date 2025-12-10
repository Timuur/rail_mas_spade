import os
import csv
from pathlib import Path
from typing import Any, Dict

from utils.io_utils import load_json, dump_json

async def append_schedule(agent: Any, plan: Dict[str, Any], persist: bool = True):
    """
    persist=True  -> писать schedule.json/csv + обновлять календари (инициатор).
    persist=False -> только обновлять календари (участники/владельцы ресурсов).
    Также обновляет позиции локомотивов и вагонов.
    """
    key_mem = (str(plan.get("order_id")), str(plan.get("proposer")))

    if persist:
        print(f"[{agent.station_id}] Записываю результат: {plan.get('order_id')}")

        if not hasattr(agent, "_written_ids"):
            agent._written_ids = set()

        if key_mem in agent._written_ids:
            return
        agent._written_ids.add(key_mem)

        os.makedirs(agent.output_dir, exist_ok=True)

        # JSON with de-dupe
        jpath = f"{agent.output_dir}/schedule.json"
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
        cpath = f"{agent.output_dir}/schedule.csv"
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
    arrive_h = float(plan.get("arrive_h", 0.0))
    end = arrive_h + float(getattr(agent, "turnaround_h", 0.5))

    if not hasattr(agent, "free_at_loco"):
        agent.free_at_loco = {}
    if not hasattr(agent, "free_at_wagon"):
        agent.free_at_wagon = {}

    def bump(d: dict, key: str, t: float):
        d[key] = max(float(d.get(key, 0.0)), float(t))

    loco_id = plan.get("loco_id")
    loco_owner = plan.get("loco_owner")

    # Владелец локомотива обновляет календарь
    if loco_id and loco_owner == agent.station_id:
        bump(agent.free_at_loco, loco_id, end)

    # Владелец вагонов обновляет календари
    for mv in plan.get("wagon_moves", []):
        if mv.get("owner") == agent.station_id and mv.get("wagon_id"):
            bump(agent.free_at_wagon, mv["wagon_id"], end)

    for w in plan.get("wagons", []):
        wid = w.get("id")
        if wid and w.get("owner") == agent.station_id:
            bump(agent.free_at_wagon, wid, end)

    # ---- Обновление позиций: локомотив и вагоны остаются на конечной станции ----
    loaded_path = plan.get("loaded_path") or []
    dest_station = loaded_path[-1] if loaded_path else None

    if dest_station:
        if not hasattr(agent, "loco_pos"):
            agent.loco_pos = {}
        if not hasattr(agent, "wagon_pos"):
            agent.wagon_pos = {}

        # локомотив
        if loco_id and loco_owner == agent.station_id:
            agent.loco_pos[loco_id] = dest_station

        # вагонные перемещения по wagon_moves
        for mv in plan.get("wagon_moves", []):
            if mv.get("owner") == agent.station_id and mv.get("wagon_id"):
                agent.wagon_pos[mv["wagon_id"]] = dest_station

        # явный список wagons
        for w in plan.get("wagons", []):
            wid = w.get("id")
            if wid and w.get("owner") == agent.station_id:
                agent.wagon_pos[wid] = dest_station