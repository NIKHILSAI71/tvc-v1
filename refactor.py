import re

with open('src/tvc/training.py', 'r') as f:
    code = f.read()

# Helper block string
helpers = """
def _setup_viewer(env: TvcEnv) -> mujoco.viewer.Viewer:
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    # Camera: Track the rocket body with good viewing angle
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = env.model.body('vehicle').id
    viewer.cam.distance = 25.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -15
    viewer.cam.lookat[:] = [0, 0, 10]

    # Visual Options: Minimal overlays (clean view)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False

    # Rendering quality
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = True

    LOGGER.info("Viewer: Camera tracking rocket, debug overlays enabled")
    return viewer

def _run_neuroevo_warmup(
    episode: int,
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    funcs: PolicyFunctions,
    config: TrainingConfig,
    env: TvcEnv,
    stage: CurriculumStage,
    stats: Dict[str, float]
) -> None:
    from .policies import safe_mutate_parameters_smg, crossover_parameters

    sample_obs = batch["observations"][:config.sequence_length]
    sample_obs = sample_obs.reshape(1, sample_obs.shape[0], sample_obs.shape[1])
    gru_dim = config.policy_config.gru_hidden_dim
    sample_hidden = jnp.zeros((1, gru_dim))

    if len(state.recent_parent_fitness) >= 3:
        current_fitness = state.get_rolling_parent_fitness()
    else:
        current_fitness = stats['episode_return']

    best_params = state.params
    best_fitness = current_fitness

    for _ in range(config.neuroevo_warmup_mutations):
        state.rng, mut_key, cross_key, select_key = jax.random.split(state.rng, 4)

        parent = state.params
        if config.use_elite_crossover and len(state.elite_archive) > 0:
            if float(jax.random.uniform(select_key)) < config.crossover_probability:
                elite_idx = int(jax.random.randint(select_key, (), 0, len(state.elite_archive)))
                _, elite_params = state.elite_archive[elite_idx]
                parent = crossover_parameters(cross_key, state.params, elite_params, 0.5)

        warmup_mutation_scale = config.mutation_scale * 3.0
        mutant_params = safe_mutate_parameters_smg(
            mut_key, parent, funcs, sample_obs, sample_hidden,
            scale=warmup_mutation_scale,
        )

        mutant_fitness = _evaluate_candidate(
            env, stage, mutant_params, funcs, state.obs_rms, config,
            num_episodes=1
        )

        if mutant_fitness > best_fitness:
            best_fitness = mutant_fitness
            best_params = mutant_params

    if best_fitness > current_fitness:
        state.params = best_params
        state.update_elite_archive(best_fitness, best_params)
        LOGGER.info(f"  [Neuroevo Warmup] Ep {episode}: R {current_fitness:.1f} -> {best_fitness:.1f} (+{best_fitness-current_fitness:.1f})")

    state.update_parent_fitness(stats['episode_return'])

    if episode == config.neuroevo_warmup_episodes - 1:
        state.neuroevo_warmup_complete = True
        state.learning_phase = 1
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("PHASE 1: Pure PPO Starting (Neuroevo Warmup Complete)")
        LOGGER.info(f"   Best warmup fitness: {state.best_return:.1f}")
        LOGGER.info(f"   Elite archive size: {len(state.elite_archive)}")
        LOGGER.info("=" * 60)

def _perform_ppo_updates(
    episode: int,
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    optimizer: optax.GradientTransformation,
    funcs: PolicyFunctions,
    config: TrainingConfig,
    entropy_coef: float
) -> Tuple[TrainingState, Dict[str, float]]:
    if state.update_step < config.value_pretrain_updates:
        value_pretrain_config = dataclass_replace(
            config,
            policy_loss_coef=0.1,
        )
        state, metrics = _ppo_update(
            state, batch, optimizer, funcs, value_pretrain_config,
            entropy_coef * 0.1
        )
        if episode % 10 == 0:
            LOGGER.info(f"  [Value Pretrain {state.update_step}/{config.value_pretrain_updates}]")
    elif state.post_evolution_cooldown_counter > 0:
        cooldown_config = dataclass_replace(
            config,
            num_epochs=config.post_evolution_epochs,
            learning_rate=config.learning_rate * config.post_evolution_lr_scale
        )
        state, metrics = _ppo_update(state, batch, optimizer, funcs, cooldown_config, entropy_coef * 0.5)
        if episode % 5 == 0:
            LOGGER.info(f"  [Post-Evolution Cooldown: {state.post_evolution_cooldown_counter} eps remaining, using {config.post_evolution_epochs} PPO epochs]")
    else:
        state, metrics = _ppo_update(state, batch, optimizer, funcs, config, entropy_coef)

    return state, metrics

def _handle_stage_progression(
    episode: int,
    state: TrainingState,
    rolling_sr: float,
    curriculum: List[CurriculumStage],
    stage: CurriculumStage,
    config: TrainingConfig,
    entropy_coef: float
) -> float:
    if not state.rajs_enabled and rolling_sr >= config.rajs_enable_threshold and state.stage_attempts >= 50:
        state.rajs_enabled = True
        state.learning_phase = 2
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("PHASE 2 UNLOCKED: RAJS Enabled (40% SR achieved)")
        LOGGER.info("   Heuristic guidance will now assist learning")
        LOGGER.info("=" * 60)

    if not state.evolution_enabled and rolling_sr >= config.evolution_enable_threshold and state.stage_attempts >= 100:
        state.evolution_enabled = True
        state.learning_phase = 3
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("PHASE 3 UNLOCKED: Evolution Enabled (70% SR achieved)")
        LOGGER.info("   Neuroevolution will now explore policy mutations")
        LOGGER.info("=" * 60)

    previous_stage_index = state.stage_index
    consecutive_success_req = stage.success_episodes
    recent_streak = sum(1 for s in state.recent_successes[-consecutive_success_req:] if s) if state.recent_successes else 0
    has_streak = recent_streak >= consecutive_success_req

    sr_threshold = 0.60 if state.stage_index == 0 else 0.70

    if rolling_sr >= sr_threshold and state.stage_attempts >= stage.min_episodes and has_streak:
        if state.stage_index < len(curriculum) - 1:
            old_stage_name = curriculum[state.stage_index].name
            state.stage_index += 1
            state.stage_attempts = 0
            state.stage_successes = 0
            state.recent_successes = []
            new_stage = curriculum[state.stage_index]
            LOGGER.info(f"")
            LOGGER.info(f"STAGE UP! {old_stage_name} -> {new_stage.name}")
            LOGGER.info(f"   New altitude: {new_stage.initial_position[2]:.0f}m | Target SR: {sr_threshold*100:.0f}%")

            if config.staged_exploration_reset > 0:
                entropy_coef = config.entropy_coef * config.staged_exploration_reset

    if state.stage_index == previous_stage_index:
        entropy_coef *= config.entropy_coef_decay

    return entropy_coef

def _run_evolution_step(
    episode: int,
    total_episodes: int,
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    stats: Dict[str, float],
    funcs: PolicyFunctions,
    config: TrainingConfig,
    env: TvcEnv,
    stage: CurriculumStage,
    optimizer: optax.GradientTransformation
) -> None:
    from .policies import safe_mutate_parameters_smg, crossover_parameters

    sample_obs = batch["observations"][:config.sequence_length]
    sample_obs = sample_obs.reshape(1, sample_obs.shape[0], sample_obs.shape[1])
    gru_dim = config.policy_config.gru_hidden_dim
    sample_hidden = jnp.zeros((1, gru_dim))

    rolling_parent_fitness = state.get_rolling_parent_fitness()
    current_episode_fitness = stats['episode_return']

    best_params = state.params
    best_fitness = current_episode_fitness

    episode_progress = min(episode / total_episodes, 1.0)
    adaptive_mutation_scale = config.mutation_scale * (0.3 + 0.7 * episode_progress)

    for candidate_idx in range(config.evolution_candidates):
        state.rng, mut_key, cross_key, select_key = jax.random.split(state.rng, 4)

        parent = state.params
        crossover_used = False
        if config.use_elite_crossover and len(state.elite_archive) > 0:
            if float(jax.random.uniform(select_key)) < config.crossover_probability:
                elite_idx = int(jax.random.randint(select_key, (), 0, len(state.elite_archive)))
                _, elite_params = state.elite_archive[elite_idx]
                parent = crossover_parameters(cross_key, state.params, elite_params, 0.5)
                crossover_used = True

        mutant_params = safe_mutate_parameters_smg(
            mut_key,
            parent,
            funcs,
            sample_obs,
            sample_hidden,
            scale=adaptive_mutation_scale,
        )

        mutant_fitness = _evaluate_candidate(
            env, stage, mutant_params, funcs, state.obs_rms, config,
            num_episodes=config.evolution_eval_episodes
        )

        if mutant_fitness > best_fitness:
            best_fitness = mutant_fitness
            best_params = mutant_params
            cross_str = " (crossover)" if crossover_used else ""
            LOGGER.info(f"  Evolution: Candidate {candidate_idx}{cross_str} improved! R: {mutant_fitness:.1f}")

    comparison_fitness = rolling_parent_fitness if rolling_parent_fitness > -float('inf') else current_episode_fitness
    improvement_ratio = best_fitness / max(abs(comparison_fitness), 1.0)
    meets_threshold = improvement_ratio >= config.fitness_improvement_threshold

    if best_params is not state.params and meets_threshold:
        state.params = best_params

        added_to_elite = state.update_elite_archive(best_fitness, best_params)
        elite_str = " [added to elite archive]" if added_to_elite else ""

        LOGGER.info(f"  Evolution: Accepted mutant with R: {best_fitness:.1f} ({improvement_ratio:.1%} vs rolling avg){elite_str}")

        state.post_evolution_cooldown_counter = config.post_evolution_cooldown
        LOGGER.info(f"  Evolution: Cooldown activated for {config.post_evolution_cooldown} episodes")

        LOGGER.info(f"  Evolution: Soft optimizer reset (keeping {config.soft_reset_momentum_keep*100:.0f}% momentum)")
        new_opt_state = optimizer.init(state.params)

        def blend_opt_states(old, new):
            if isinstance(old, jnp.ndarray) and isinstance(new, jnp.ndarray):
                if old.shape == new.shape:
                    return config.soft_reset_momentum_keep * old + (1 - config.soft_reset_momentum_keep) * new
            return new

        try:
            state.opt_state = jax.tree_util.tree_map(blend_opt_states, state.opt_state, new_opt_state)
        except:
            state.opt_state = new_opt_state
    elif best_params is not state.params:
        LOGGER.info(f"  Evolution: Rejected mutant (improvement {improvement_ratio:.1%} < threshold {config.fitness_improvement_threshold:.0%})")
"""

# Find `def train_controller`
train_controller_idx = code.find('def train_controller(')

new_code = code[:train_controller_idx] + helpers + "\n\n" + code[train_controller_idx:]

with open('src/tvc/training.py', 'w') as f:
    f.write(new_code)
