import asyncio
import json
import time

from spade.behaviour import CyclicBehaviour
from spade.message import Message

from agents.module.message_types import (
    TYPE_HELLO_PING,
    TYPE_HELLO_PONG,
    TYPE_HELLO_START,
)

class ListenHello(CyclicBehaviour):
    """
    Обработка PING/PONG и сигнала старта торгов.
    """

    async def run(self):
        msg = await self.receive(timeout=1)
        if not msg:
            return

        self.agent.last_active = time.time()
        meta = msg.metadata or {}
        mtype = meta.get("type")

        # Старт торгов
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

        # PING/PONG
        if mtype not in (TYPE_HELLO_PING, TYPE_HELLO_PONG):
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
                print(
                    f"[{self.agent.station_id}] (последний) увидел живого агента: {st_id}. "
                    f"Всего вижу: {len(self.agent.ready_stations)}/{len(self.agent.stations)}"
                )
            else:
                print(
                    f"[{self.agent.station_id}] (не последний) увидел агента: {st_id}. "
                    f"Сейчас вижу: {len(self.agent.ready_stations)} агента(ов)."
                )

        # ответ PONG на PING
        if mtype == TYPE_HELLO_PING:
            reply = Message(to=str(msg.sender))
            reply.set_metadata("type", TYPE_HELLO_PONG)
            reply.body = json.dumps(
                {"station_id": self.agent.station_id}, ensure_ascii=False
            )
            await self.send(reply)

        # Барьер только у последнего
        if self.agent.is_last and len(self.agent.ready_stations) == len(self.agent.stations):
            if not self.agent.all_ready_event.is_set():
                print(
                    f"[{self.agent.station_id}] (последний) увидел всех "
                    f"{len(self.agent.stations)} агентов. Барьер готов."
                )
                self.agent.all_ready_event.set()


class IdleKiller(CyclicBehaviour):
    """
    Останавливает агента, если он слишком долго неактивен.
    """

    def __init__(self, timeout_s: float = 180):
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