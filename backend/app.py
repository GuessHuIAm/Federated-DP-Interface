from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dp_fl_simulation import run_dp_federated_learning
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can lock this down to your React URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_stream(epsilon, clip, num_clients, mechanism, rounds):
    for round_num, (global_acc, client_acc) in enumerate(
        run_dp_federated_learning(
            epsilon, clip, num_clients, mechanism, rounds,
        )
    ):
        # Prepare data
        data = {
            "round_num": round_num,
            "global_accuracy": global_acc,
            "client_accuracy": client_acc
        }
        # Yield as a server-sent event (SSE) format
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.01)  # tiny sleep to cooperate with async server

@app.get("/stream_training")
async def stream_training(epsilon: float, clip: float, num_clients: int, mechanism: str, rounds: int):
    return StreamingResponse(
        generate_stream(epsilon, clip, num_clients, mechanism, rounds),
        media_type="text/event-stream"
    )
