import sys

with open('src/tvc/training.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip() == 'LOGGER.info("PHASE 1: Pure PPO Starting (Neuroevo Warmup Complete)")':
        print(f"Line {i+1}: {repr(line)}")
