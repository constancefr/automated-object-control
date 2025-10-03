from gymnasium.envs.registration import register

register(
    id="versaille_env/acc-discrete-v0",
    entry_point="versaille_env.envs:ACCEnv",
)
