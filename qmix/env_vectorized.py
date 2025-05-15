class VectorizedEnv:
    def __init__(self, env_cls, num_envs, **env_kwargs):
        # Assign a unique seed to each environment if 'seed' is in env_kwargs
        base_seed = env_kwargs.get('seed', None)
        self.envs = []
        for idx in range(num_envs):
            env_args = env_kwargs.copy()
            if base_seed is not None:
                env_args['seed'] = base_seed + idx
            self.envs.append(env_cls(**env_args))
        self.num_envs = num_envs

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        # actions: list of actions for each env
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        next_states, rewards, dones, infos = zip(*results)
        return list(next_states), list(rewards), list(dones), list(infos)

    def render(self):
        for env in self.envs:
            env.render_pygame() 