import re

with open('src/tvc/training.py', 'r') as f:
    code = f.read()

# Replace the viewer configuration with the helper function
viewer_pattern = r"""        viewer = mujoco\.viewer\.launch_passive\(env\.model, env\.data\)

        # ============================================================
        # VIEWER CONFIGURATION: Camera, Quality, Debug Overlays
        # ============================================================.*?
        LOGGER\.info\("Viewer: Camera tracking rocket, debug overlays enabled"\)"""
code = re.sub(viewer_pattern, "        viewer = _setup_viewer(env)", code, flags=re.DOTALL)


# Now, replace the warmup phase
warmup_pattern = r"""        # ============================================================
        # PHASE 0: NEUROEVOLUTION WARMUP \(AS Rocketry-inspired\)
        # Pure evolution for first N episodes to find good starting weights
        # This is much faster than PPO for initial exploration
        # ============================================================
        if episode < config\.neuroevo_warmup_episodes and not state\.neuroevo_warmup_complete:.*?LOGGER\.info\("=" \* 60\)"""

warmup_replacement = """        # ============================================================
        # PHASE 0: NEUROEVOLUTION WARMUP (AS Rocketry-inspired)
        # ============================================================
        if episode < config.neuroevo_warmup_episodes and not state.neuroevo_warmup_complete:
            _run_neuroevo_warmup(episode, state, batch, funcs, config, env, stage, stats)"""

code = re.sub(warmup_pattern, warmup_replacement, code, flags=re.DOTALL)


# Now, replace the PPO logic
ppo_pattern = r"""        # ============================================================
        # VALUE PRETRAINING: First N updates train value function only
        # This stabilizes advantage estimates before actor learning
        # ============================================================
        elif state\.update_step < config\.value_pretrain_updates:.*?# Normal PPO update
            state, metrics = _ppo_update\(state, batch, optimizer, funcs, config, entropy_coef\)"""

ppo_replacement = """        # ============================================================
        # VALUE PRETRAINING & PPO UPDATES
        # ============================================================
        else:
            state, metrics = _perform_ppo_updates(episode, state, batch, optimizer, funcs, config, entropy_coef)"""

code = re.sub(ppo_pattern, ppo_replacement, code, flags=re.DOTALL)

with open('src/tvc/training.py', 'w') as f:
    f.write(code)
