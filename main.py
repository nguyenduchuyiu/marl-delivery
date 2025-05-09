from env import Environment
#from agent import Agents
from agent import Agents as Agents

import numpy as np
import time
import pygame

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning for Delivery")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Maximum time steps for the environment")
    parser.add_argument("--map", type=str, default="map.txt", help="Map name")

    args = parser.parse_args()
    np.random.seed(args.seed)

    env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                      n_robots=args.num_agents, n_packages=args.n_packages,
                      seed = args.seed)
    
    state = env.reset()
    agents = Agents(weights_path="ppo_delivery.zip")
    agents.init_agents(state)
    # print(state)
    env.render()
    done = False
    t = 0
    while not done:
        actions = agents.get_actions(state)
        print("Actions passed to env:", actions)
        next_state, reward, done, infos = env.step(actions)
        state = next_state
        env.render_pygame()
        pygame.time.wait(100)  # 100 ms pause for visibility
        t += 1

        delivered = sum(1 for p in env.packages if p.status == 'delivered')
        waiting = sum(1 for p in env.packages if p.status == 'waiting')
        in_transit = sum(1 for p in env.packages if p.status == 'in_transit')
        total = len(env.packages)

        print(f"Time step: {env.t}")
        print(f"Total Reward: {env.total_reward}")
        print(f"Packages: {delivered}/{total} delivered, {waiting} waiting, {in_transit} in transit")
        for i, robot in enumerate(env.robots):
            carrying = f"carrying package {robot.carrying}" if robot.carrying else "not carrying"
            print(f"Robot {i}: position={robot.position}, {carrying}")

        # Show upcoming packages (to be released in the next 3 timesteps)
        upcoming = [p for p in env.packages if p.status == 'None' and env.t < p.start_time <= env.t+3]
        if upcoming:
            print("Upcoming packages in next 3 steps:")
            for p in upcoming:
                print(f"  Package {p.package_id} at {p.start} -> {p.target} (release at t={p.start_time}, deadline={p.deadline})")

    print("Episode finished")
    print("Total reward:", infos['total_reward'])
    print("Total time steps:", infos['total_time_steps'])
    pygame.quit()
