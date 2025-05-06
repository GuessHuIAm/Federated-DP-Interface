from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dp_fl_simulation import run_dp_federated_learning
from pydantic import BaseModel
from joblib import Memory

memory = Memory("cache_results")
run_dp_federated_learning = memory.cache(run_dp_federated_learning)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunConfig(BaseModel):
    epsilon: float
    numClients: int
    rounds: int

@app.post("/run")
async def run(config: RunConfig):
    print("[DEBUG] /run hit with:", config)
    
    # With DP noise
    dp_acc, _, noise = run_dp_federated_learning(
        epsilon=config.epsilon,
        clip=1,
        num_clients=config.numClients,
        mechanism="Gaussian",
        rounds=config.rounds,
        dp_noise=True,
    )

    # Without DP noise
    non_dp_acc, _ = run_dp_federated_learning(
        epsilon=config.epsilon,  # still passed but unused
        clip=1,
        num_clients=config.numClients,
        mechanism="Gaussian",
        rounds=config.rounds,
        dp_noise=False,  # disabled
    )

    return {
        "dp_final_accuracy": dp_acc[-1],
        "non_dp_final_accuracy": non_dp_acc[-1],
        "average_noise": noise,
    }
