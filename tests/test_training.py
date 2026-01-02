import logging
import dataclasses
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tvc.training import train_controller, TrainingConfig, LOGGER

def test_training_loop():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOGGER.info("Starting verification run...")
    
    # Run a short training session for verification
    config = TrainingConfig()
    # Override for speed
    config = dataclasses.replace(config, rollout_length=128, minibatch_size=32, num_epochs=2, sequence_length=32, eval_max_steps=100)
    
    try:
        train_controller(total_episodes=2, config=config, visualize=False)
        LOGGER.info("Verification run completed successfully.")
    except Exception as e:
        LOGGER.error(f"Verification failed: {e}")
        raise

if __name__ == "__main__":
    test_training_loop()
