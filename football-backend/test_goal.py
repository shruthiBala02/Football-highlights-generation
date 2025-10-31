import asyncio
import websockets
import json

async def send_goal():
    uri = "ws://127.0.0.1:8000/ws/goals"
    async with websockets.connect(uri) as websocket:
        for i in range(1, 4):
            data = {
                "id": i,
                "title": f"Goal #{i}",
                "url": f"http://127.0.0.1:8000/outputs/sample_goal.mp4"
            }
            await websocket.send(json.dumps(data))
            print(f"Sent Goal {i}")
            await asyncio.sleep(3)

asyncio.run(send_goal())
