from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dp_fl_simulation import run_dp_federated_learning
import json
import asyncio
import time
from pydantic import BaseModel
from typing import Literal


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Add this back
async def generate_stream(epsilon, clip, num_clients, mechanism, rounds):
    training_start_time = time.time()
    round_start_time = training_start_time

    sim = run_dp_federated_learning(epsilon, clip, num_clients, mechanism, rounds)

    # Initial evaluation
    start_eval_time = time.time()
    global_acc, client_acc = next(sim)
    end_eval_time = time.time()

    initial_round_duration = end_eval_time - start_eval_time
    total_training_time = end_eval_time - training_start_time

    data = {
        "round_num": 0,
        "global_accuracy": global_acc,
        "client_accuracy": client_acc,
        "round_duration": initial_round_duration,
        "total_training_time": total_training_time,
    }
    yield f"data: {json.dumps(data)}\n\n"

    round_start_time = time.time()

    for round_num, (global_acc, client_acc) in enumerate(sim, start=1):
        now = time.time()
        round_duration = now - round_start_time
        total_training_time = now - training_start_time

        data = {
            "round_num": round_num,
            "global_accuracy": global_acc,
            "client_accuracy": client_acc,
            "round_duration": round_duration,
            "total_training_time": total_training_time
        }
        yield f"data: {json.dumps(data)}\n\n"

        round_start_time = time.time()
        await asyncio.sleep(0.01)

# SSE endpoint
@app.get("/stream_training")
async def stream_training(epsilon: float, clip: float, num_clients: int, mechanism: str, rounds: int):
    return StreamingResponse(
        generate_stream(epsilon, clip, num_clients, mechanism, rounds),
        media_type="text/event-stream"
    )

# Batch (non-streaming) endpoint
class RunOnceConfig(BaseModel):
    epsilon: float
    clip: float
    numClients: int
    mechanism: Literal["Gaussian", "Laplace"]
    rounds: int

@app.post("/run_once")
async def run_once(config: RunOnceConfig):
    print("[DEBUG] /run_once hit with:", config)

    sim = run_dp_federated_learning(
        epsilon=config.epsilon,
        clip=config.clip,
        num_clients=config.numClients,
        mechanism=config.mechanism,
        rounds=config.rounds,
    )

    final_global_acc = None
    for global_acc, _ in sim:
        final_global_acc = global_acc

    print(f"[DEBUG] Final Accuracy: {final_global_acc[-1]}")
    return {"final_accuracy": final_global_acc[-1]}
