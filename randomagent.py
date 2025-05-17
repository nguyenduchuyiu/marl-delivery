import numpy as np

class RandomAgents:
    def __init__(self):
        self.n_robots = 0
        self.is_init = False

    def init_agents(self, state):
        self.n_robots = len(state['robots'])
        self.is_init = True

    def get_actions(self, state, deterministic=True):
        # Mỗi robot chọn ngẫu nhiên một hành động: 
        # move: 'U', 'D', 'L', 'R', 'S' (up, down, left, right, stay)
        # action: '0' (no-op), '1' (pickup), '2' (drop)
        moves = ['U', 'D', 'L', 'R', 'S']
        actions = ['0', '1', '2']
        result = []
        for _ in range(self.n_robots):
            move = np.random.choice(moves)
            act = np.random.choice(actions)
            result.append((move, act))
        return result
