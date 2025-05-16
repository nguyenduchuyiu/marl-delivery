# QMIX/vectorized_env.py
class VectorizedEnv:
    def __init__(self, env_cls, num_envs, **env_kwargs):
        base_seed = env_kwargs.get('seed', None)
        self.envs = []
        for idx in range(num_envs):
            env_args = env_kwargs.copy()
            if base_seed is not None:
                env_args['seed'] = base_seed + idx
            self.envs.append(env_cls(**env_args))
        self.num_envs = num_envs

    def reset(self, indices=None):
        """
        Reset all environments or a subset specified by indices.
        Returns a list of observations (for all or selected envs).
        """
        if indices is None:
            return [env.reset() for env in self.envs]
        else:
            return [self.envs[i].reset() for i in indices]

    def step(self, actions, indices=None):
        """
        Step all environments or a subset specified by indices.
        - actions: list of actions (for all envs or for selected indices)
        - indices: list of indices to step (optional)
        Returns: list of (next_state, reward, done, info) for each stepped env.
        """
        if indices is None:
            # Step all envs
            results = [env.step(action) for env, action in zip(self.envs, actions)]
        else:
            # Step only selected envs
            results = [self.envs[i].step(action) for i, action in zip(indices, actions)]
        next_states, rewards, dones, infos = zip(*results)
        return list(next_states), list(rewards), list(dones), list(infos)

    def render(self, indices=None):
        """
        Render all environments or a subset specified by indices.
        """
        if indices is None:
            for env in self.envs:
                env.render_pygame()
        else:
            for i in indices:
                self.envs[i].render_pygame()

