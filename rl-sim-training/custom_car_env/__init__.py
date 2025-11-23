from gymnasium.envs.registration import register

register(
    id="custom_car_env/acc-continuous-v0",
    entry_point="custom_car_env.envs:ACCEnv",
)
