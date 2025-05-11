from env import Environment
from agent import Agents as Agents
from greedyagent import GreedyAgents as GreedyAgents
from dqn.networks import convert_state

import numpy as np
import time
import pygame
import matplotlib.pyplot as plt

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
    observation_shape = (6, env.n_rows, env.n_cols)
    agents = Agents(observation_shape, "models/dqn_agent_final.pt", "cuda")
    # agents = GreedyAgents()
    agents.init_agents(state)
    # print(state)
    env.render()
    done = False
    t = 0
    # `infos` will be from the last call to env.step().
    # It's only guaranteed to be fully populated by env.step if `done` is True from that step.
    # We initialize it here; it will be updated during the loop.
    infos = {}

    while not done:
        actions = agents.get_actions(state)
        print("Actions passed to env:", actions)
        # `current_done` and `current_infos` are from this specific step
        next_state, reward, current_done, current_infos = env.step(actions)
        state = next_state
        infos = current_infos # Always update infos with the latest from env.step
                              # It will contain total_reward etc. if current_done is True.
        
        if current_done:
            done = True
            # `infos` is already set correctly from `current_infos` by env.step
        
        # Only attempt to render if the simulation isn't already marked as done
        if not done:
            try:
                env.render_pygame()
                pygame.time.wait(100)  # 100 ms pause for visibility
            except pygame.error as e:
                print(f"Pygame window closed or error during rendering: {e}. Ending simulation.")
                done = True # Signal to terminate the loop
                # If the environment hadn't already signaled 'done' via current_done,
                # we need to populate `infos` with the current environment state for the final print.
                if not current_done:
                    infos['total_reward'] = env.total_reward
                    infos['total_time_steps'] = env.t
        
        if done:
            # If done is true, `infos` should now be correctly populated either by
            # env.step (if `current_done` was true) or by the `except` block.
            break # Exit the loop

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
    # `infos` should now hold the correct final information.
    # Using .get() with a fallback to current env values for robustness,
    # though the logic above aims to ensure `infos` is correctly populated.
    print("Total reward:", infos.get('total_reward', env.total_reward))
    print("Total time steps:", infos.get('total_time_steps', env.t))

    # --- Plotting the final state (agent 0's perspective) ---
    # IMPORTANT: Ensure you have the 'convert_state' function (from your notebook)
    # defined or imported in this script for the following code to work.
    # You'll also need 'get_persistent_packages'.

    # Example definition for get_persistent_packages (if not already available):
    def get_persistent_packages(current_env):
        return {
            p.package_id: {
                'id': p.package_id,
                'start_pos': p.start,
                'target_pos': p.target,
                'start_time': p.start_time,
                'deadline': p.deadline,
                'status': p.status
            }
            for p in current_env.packages
        }
        
    try:
        final_pkgs = get_persistent_packages(env)
        # print(final_pkgs) # Optional: for debugging
        
        # Assuming you want to visualize from the perspective of agent 1 (as in original code)
        # And that your convert_state function is now updated and imported/defined.
        # The 'state' here is the final state of the environment.
        plot_robot_idx = 1 # Or 0, or any agent you want to visualize
        if plot_robot_idx >= args.num_agents:
            print(f"Warning: plot_robot_idx {plot_robot_idx} is out of bounds for {args.num_agents} agents. Defaulting to 0.")
            plot_robot_idx = 0
            
        final_tensor_observation = convert_state(state, final_pkgs, current_robot_idx=plot_robot_idx)

        # Updated channel names for the 6-channel representation
        channel_names_plot = [
            "Map (1=obstacle, 0=empty)",
            "Urgency (Waiting Pkgs, if not carrying)",
            "Start Pos (Waiting Pkgs, if not carrying)",
            "Other Robots' Positions",
            f"Robot {plot_robot_idx}'s Position",
            f"Robot {plot_robot_idx}'s Carried Pkg Target (if carrying)",
        ]

        n_channels_plot = final_tensor_observation.shape[0]
        # Adjust figsize as needed; (width, height)
        fig, axes_plot = plt.subplots(1, n_channels_plot, figsize=(3 * n_channels_plot, 3.5))

        if n_channels_plot == 1 and not isinstance(axes_plot, np.ndarray):
            axes_plot = [axes_plot] # Ensure axes_plot is iterable for a single channel

        for i in range(n_channels_plot):
            ax = axes_plot[i]
            im = ax.imshow(final_tensor_observation[i], cmap='viridis', interpolation='nearest')
            if i < len(channel_names_plot):
                ax.set_title(channel_names_plot[i], fontsize=8) # Adjusted fontsize
            else:
                ax.set_title(f"Channel {i+1}", fontsize=8) # Adjusted fontsize
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f"Final State Observation (Agent {plot_robot_idx}, t={env.t})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        plt.show()
    except NotImplementedError as e:
        print(f"Plotting skipped: {e}")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

    pygame.quit()
