import re

with open('src/tvc/training.py', 'r') as f:
    code = f.read()

# Replace missing metrics logging

logging_str = """        phase_str = f"P{state.learning_phase}"

        # Enhanced logging with detailed metrics
        success_emoji = "✓" if stats['episode_success'] else "✗"

        # Log every episode with basic info
        LOGGER.info(f"Ep {episode:4d} | {phase_str} | R: {stats['episode_return']:8.1f} | Steps: {stats.get('episode_length', 0):3.0f} | SR: {rolling_sr*100:4.1f}% | {stage.name} | {success_emoji}")

        # Detailed metrics every 5 episodes
        if episode % 5 == 0:
            if 'metrics' in locals() and metrics is not None:
                actor_loss = metrics.get('actor_loss', 0.0)
                value_loss = metrics.get('value_loss', 0.0)
                entropy = metrics.get('entropy', 0.0)
                kl = metrics.get('kl', 0.0)
                pos_err = stats.get('pos_error', 0.0)
                vel_err = stats.get('vel_error', 0.0)
                LOGGER.info(f"         └─ Loss: {metrics.get('loss', 0.0):7.3f} | Actor: {actor_loss:6.3f} | Value: {value_loss:6.3f} | Ent: {entropy:5.2f} | KL: {kl:.4f}")
                updates_this_ep = metrics.get('updates_this_ep', 0)
                LOGGER.info(f"         └─ PosErr: {pos_err:5.2f}m | VelErr: {vel_err:5.2f}m/s | Best: {state.best_return:8.1f} | Updates: {state.update_step} (+{updates_this_ep})")

        # ============================================================
        # STAGED LEARNING: Automatic Progression Based on Success Rate"""

code = code.replace("        # ============================================================\n        # STAGED LEARNING: Automatic Progression Based on Success Rate", logging_str)

with open('src/tvc/training.py', 'w') as f:
    f.write(code)
