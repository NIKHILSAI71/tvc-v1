import re

with open('src/tvc/training.py', 'r') as f:
    code = f.read()

# Refactor progression handling
progression_pattern = r"""        # ============================================================
        # STAGED LEARNING: Automatic Progression Based on Success Rate
        # ============================================================
        # Phase 1 → Phase 2: Enable RAJS at 40% success rate
        if not state\.rajs_enabled and rolling_sr >= config\.rajs_enable_threshold and state\.stage_attempts >= 50:.*?# Decay entropy \(only if stage didn't just advance\)
        if state\.stage_index == previous_stage_index:
            entropy_coef \*= config\.entropy_coef_decay"""

progression_replacement = """        # ============================================================
        # STAGED LEARNING: Automatic Progression Based on Success Rate
        # ============================================================
        entropy_coef = _handle_stage_progression(
            episode, state, rolling_sr, curriculum, stage, config, entropy_coef
        )"""

code = re.sub(progression_pattern, progression_replacement, code, flags=re.DOTALL)


# Refactor Evolution
evolution_pattern = r"""        # ============================================================
        # EVOLUTION: Safe Mutation through Gradients \(SM-G\) for GRU
        # ============================================================.*?state\.update_parent_fitness\(stats\['episode_return'\]\)

        # Decrement cooldown counter each episode
        if state\.post_evolution_cooldown_counter > 0:
            state\.post_evolution_cooldown_counter -= 1"""

evolution_replacement = """        # ============================================================
        # EVOLUTION: Safe Mutation through Gradients (SM-G) for GRU
        # ============================================================
        stage_episodes_since_advance = state.stage_attempts
        evolution_ready = (
            (config.use_evolution or state.evolution_enabled)
            and episode > config.evolution_warmup_episodes
            and episode % config.evolution_interval == 0
            and stage_episodes_since_advance > config.evolution_stage_lockout
        )
        if evolution_ready:
            _run_evolution_step(
                episode, total_episodes, state, batch, stats,
                funcs, config, env, stage, optimizer
            )

        # Update rolling parent fitness for next comparison
        state.update_parent_fitness(stats['episode_return'])

        # Decrement cooldown counter each episode
        if state.post_evolution_cooldown_counter > 0:
            state.post_evolution_cooldown_counter -= 1"""

code = re.sub(evolution_pattern, evolution_replacement, code, flags=re.DOTALL)


with open('src/tvc/training.py', 'w') as f:
    f.write(code)
