from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dp_fl_simulation import run_dp_federated_learning
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can lock this down to your React URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_stream(epsilon, clip, num_clients, mechanism, rounds):
    training_start_time = time.time()
    round_start_time = training_start_time

    sim = run_dp_federated_learning(epsilon, clip, num_clients, mechanism, rounds)

    # Initial evaluation timing (round 0)
    start_eval_time = time.time()
    global_acc, client_acc = next(sim)  # initial evaluation happens here
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

    round_start_time = time.time()  # reset after initial evaluation

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

@app.get("/stream_training")
async def stream_training(epsilon: float, clip: float, num_clients: int, mechanism: str, rounds: int):
    return StreamingResponse(
        generate_stream(epsilon, clip, num_clients, mechanism, rounds),
        media_type="text/event-stream"
    )
