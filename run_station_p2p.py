# import argparse
# from agents.station_agent_p2p import StationAgent
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--jid", required=True)
#     parser.add_argument("--password", required=True)
#     #parser.add_argument("--server", required=True)
#     parser.add_argument("--station_id", required=True)
#     parser.add_argument("--data", default="data/input.json")
#     args = parser.parse_args()
#
#     #agent = StationAgent(args.jid, args.password, station_id=args.station_id, data_path=args.data, server=args.server)
#     agent = StationAgent(args.jid, args.password, station_id=args.station_id, data_path=args.data)
#     agent.start().result()
#
#     try:
#         agent.web.start(hostname="127.0.0.1", port=0)
#     except Exception:
#         pass
#
#     try:
#         agent.join()
#     finally:
#         agent.stop()
#
# if __name__ == "__main__":
#     main()


import argparse
import asyncio
import sys
from agents.station_agent_p2p import StationAgent

# На Windows многим XMPP-стекам комфортнее selector-луп
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jid", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--station_id", required=True)
    parser.add_argument("--data", default="data/input.json")
    parser.add_argument("--is_last", action="store_true",help="Отметить этого агента как последнего, он начнёт торги после подключения всех остальных")
    return parser.parse_args()

async def amain():
    args = parse_args()

    agent = StationAgent(
        args.jid,
        args.password,
        station_id=args.station_id,
        data_path=args.data,
        is_last=args.is_last,
    )

    # ВАЖНО: start() — корутина, её нужно await
    await agent.start()

    # web.start обычно синхронный; не критичен
    try:
        agent.web.start(hostname="127.0.0.1", port=0)
    except Exception:
        pass

    try:
        # "Живём", пока агент активен
        while agent.is_alive():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(amain())
