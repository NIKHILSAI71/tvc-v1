import time
import numpy as np
from src.tvc.env import TvcEnv

def run_benchmark(env, num_steps=1000):
    start_time = time.time()
    env.reset()

    # We want to measure the performance of `env.step()`. We pass random actions.
    for _ in range(num_steps):
        # Action space: tvc_pitch, tvc_yaw, thrust_control
        action = np.random.uniform(-1.0, 1.0, size=(3,))
        env.step(action)

    duration = time.time() - start_time
    print(f"Time taken for {num_steps} steps: {duration:.4f} seconds")
    print(f"Steps per second: {num_steps / duration:.2f}")

if __name__ == "__main__":
    env = TvcEnv()
    run_benchmark(env, num_steps=10000)
