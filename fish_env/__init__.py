from gym.envs.registration import register

register(
    id='FishEnv-v0',
    entry_point='fish_env.fish_env:FishEnv',
)