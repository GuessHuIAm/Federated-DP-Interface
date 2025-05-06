from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dp_fl_simulation import run_dp_federated_learning
import json
import asyncio
import time
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_stream(epsilon, num_clients, rounds):
    sim = run_dp_federated_learning(epsilon, clip=1.0, num_clients=num_clients, mechanism="Gaussian", rounds=rounds)
    training_start_time = time.time()

    global_acc, client_acc = next(sim)
    initial_time = time.time()

    data = {
        "round_num": 0,
        "global_accuracy": global_acc,
        "client_accuracy": client_acc,
        "round_duration": initial_time - training_start_time,
        "total_training_time": initial_time - training_start_time
    }
    yield f"data: {json.dumps(data)}\n\n"

    for round_num, (global_acc, client_acc) in enumerate(sim, start=1):
        now = time.time()
        data = {
            "round_num": round_num,
            "global_accuracy": global_acc,
            "client_accuracy": client_acc,
            "round_duration": 0.0,
            "total_training_time": now - training_start_time
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.01)

@app.get("/stream_training")
async def stream_training(epsilon: float, num_clients: int, rounds: int):
    return StreamingResponse(
        generate_stream(epsilon, num_clients, rounds),
        media_type="text/event-stream"
    )

class RunOnceConfig(BaseModel):
    epsilon: float
    numClients: int
    rounds: int

@app.post("/run_once")
async def run_once(config: RunOnceConfig):
    print("[DEBUG] /run_once hit with:", config)
    
    global_acc, _ = run_dp_federated_learning(
        epsilon=config.epsilon,
        clip=1,
        num_clients=config.numClients,
        mechanism="Gaussian",
        rounds=config.rounds,
    )
    final_acc = global_acc[-1]
    print(f"Final Accuracy: {final_acc}")
    return {"final_accuracy": final_acc}
