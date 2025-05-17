from MAPPO.helper import generate_vector_features, convert_observation
from env import Environment
from agent import Agents
from greedyagent import GreedyAgents
from randomagent import RandomAgents
import numpy as np
import argparse

def run_eval(agent_type, mappo_params, num_episodes, env_config):
    rewards, delivered, delivery_rate = [], [], []
    base_seed = env_config['seed']
    for ep in range(num_episodes):
        env = Environment(
            map_file=env_config['map'],
            max_time_steps=env_config['max_time_steps'],
            n_robots=env_config['n_agents'],
            n_packages=env_config['n_packages'],
            seed=base_seed + ep
        )
        state = env.reset()
        if agent_type == "mappo":
            agent = Agents(
                mappo_params['observation_shape'],
                mappo_params['vector_obs_dim'],
                env_config['max_time_steps'],
                mappo_params['model_path'],
                mappo_params['device']
            )
        elif agent_type == "greedy":
            agent = GreedyAgents()
        elif agent_type == "random":
            agent = RandomAgents()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        agent.init_agents(state)
        done = False
        infos = {}
        current_total_reward_debug = 0
        while not done:
            use_deterministic_action = True
            if agent_type == "mappo":
                use_deterministic_action = False
            actions = agent.get_actions(state, deterministic=use_deterministic_action)
            state, reward, current_done, current_infos = env.step(actions)
            infos = current_infos
            if current_done:
                done = True
        final_reward = infos.get('total_reward', env.total_reward)
        rewards.append(final_reward)
        n_delivered = sum(1 for p in env.packages if p.status == 'delivered')
        delivered.append(n_delivered)
        current_delivery_rate = 0
        if env_config['n_packages'] > 0:
            current_delivery_rate = (n_delivered / env_config['n_packages']) * 100
        delivery_rate.append(current_delivery_rate)
    return {
        "rewards": rewards,
        "delivered": delivered,
        "delivery_rate": delivery_rate,
        "mean_reward": np.mean(rewards) if rewards else 0,
        "std_reward": np.std(rewards) if rewards else 0,
        "mean_delivered": np.mean(delivered) if delivered else 0,
        "std_delivered": np.std(delivered) if delivered else 0,
        "mean_delivery_rate": np.mean(delivery_rate) if delivery_rate else 0,
        "std_delivery_rate": np.std(delivery_rate) if delivery_rate else 0,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=5)
    parser.add_argument("--n_packages", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max_time_steps", type=int, default=1000)
    parser.add_argument("--map", type=str, default="map.txt")
    parser.add_argument("--num_test_episodes", type=int, default=10, 
                        help="Số episode để chạy cho mỗi agent. Mỗi episode sẽ dùng cùng seed môi trường.")
    parser.add_argument("--mappo_model_path", type=str, default="MAPPO/models/mappo_final_actor.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    np.random.seed(args.seed)

    temp_env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                           n_robots=args.num_agents, n_packages=args.n_packages,
                           seed=args.seed)
    temp_state = temp_env.reset()
    obs_shape = (6, temp_env.n_rows, temp_env.n_cols)
    vec_obs_dim = 0
    try:
        vec_obs_dim = generate_vector_features(temp_state, {}, 0, args.max_time_steps).shape[0]
    except Exception as e:
        print(f"Lỗi khi tạo vector_obs_dim: {e}. Đặt tạm vec_obs_dim = 0.")
    del temp_env

    env_config = {
        'max_time_steps': args.max_time_steps,
        'n_packages': args.n_packages,
        'n_agents': args.num_agents,
        'seed': args.seed,
        'map': args.map,
    }
    mappo_params = {
        'observation_shape': obs_shape,
        'vector_obs_dim': vec_obs_dim,
        'model_path': args.mappo_model_path,
        'device': args.device
    }

    print(f"Testing each agent on {args.num_test_episodes} episodes.")
    print(f"Environment seed for each episode: {args.seed}")
    print(f"MAPPO agent will use deterministic=False for actions.")
    print(f"Greedy and Random agents will use deterministic=True for actions.")

    print("\nEvaluating MAPPO agent...")
    mappo_metrics = run_eval("mappo", mappo_params, args.num_test_episodes, env_config)
    print("Evaluating Greedy agent...")
    greedy_metrics = run_eval("greedy", mappo_params, args.num_test_episodes, env_config)
    print("Evaluating Random agent...")
    random_metrics = run_eval("random", mappo_params, args.num_test_episodes, env_config)

    print("\n=== Evaluation Results ===")
    print(f"MAPPO: mean_reward={mappo_metrics['mean_reward']:.2f}±{mappo_metrics['std_reward']:.2f}, "
          f"mean_delivered={mappo_metrics['mean_delivered']:.2f}±{mappo_metrics['std_delivered']:.2f}, "
          f"mean_delivery_rate={mappo_metrics['mean_delivery_rate']:.2f}%±{mappo_metrics['std_delivery_rate']:.2f}%")
    print(f"Greedy: mean_reward={greedy_metrics['mean_reward']:.2f}±{greedy_metrics['std_reward']:.2f}, "
          f"mean_delivered={greedy_metrics['mean_delivered']:.2f}±{greedy_metrics['std_delivered']:.2f}, "
          f"mean_delivery_rate={greedy_metrics['mean_delivery_rate']:.2f}%±{greedy_metrics['std_delivery_rate']:.2f}%")
    print(f"Random: mean_reward={random_metrics['mean_reward']:.2f}±{random_metrics['std_reward']:.2f}, "
          f"mean_delivered={random_metrics['mean_delivered']:.2f}±{random_metrics['std_delivered']:.2f}, "
          f"mean_delivery_rate={random_metrics['mean_delivery_rate']:.2f}%±{random_metrics['std_delivery_rate']:.2f}%")

    # Plot nhiều metric hơn
    try:
        import matplotlib.pyplot as plt
        labels = ['Reward', 'Delivered', 'Delivery Rate (%)']
        
        mappo_means = [mappo_metrics['mean_reward'], mappo_metrics['mean_delivered'], mappo_metrics['mean_delivery_rate']]
        mappo_stds = [mappo_metrics['std_reward'], mappo_metrics['std_delivered'], mappo_metrics['std_delivery_rate']]
        
        greedy_means = [greedy_metrics['mean_reward'], greedy_metrics['mean_delivered'], greedy_metrics['mean_delivery_rate']]
        greedy_stds = [greedy_metrics['std_reward'], greedy_metrics['std_delivered'], greedy_metrics['std_delivery_rate']]
        
        random_means = [random_metrics['mean_reward'], random_metrics['mean_delivered'], random_metrics['mean_delivery_rate']]
        random_stds = [random_metrics['std_reward'], random_metrics['std_delivered'], random_metrics['std_delivery_rate']]
        
        x = np.arange(len(labels))
        width = 0.25 
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width, mappo_means, width, yerr=mappo_stds, label='MAPPO', color='skyblue', capsize=5)
        rects2 = ax.bar(x, greedy_means, width, yerr=greedy_stds, label='Greedy', color='salmon', capsize=5)
        rects3 = ax.bar(x + width, random_means, width, yerr=random_stds, label='Random', color='gray', capsize=5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Score')
        ax.set_title(
            f'MAPPO vs Greedy vs Random Agent Evaluation ({args.num_test_episodes} episodes each)\n'
            f'Map: {args.map} | Agents: {args.num_agents} | Packages: {args.n_packages} | Seed: {args.seed}',
            fontsize=12
        )
        ax.legend()
        
        # Thêm số và sai số lên trên cột
        def autolabel(rects, stds):
            for rect, std in zip(rects, stds):
                height = rect.get_height()
                ax.annotate(f'{height:.2f}\n±{std:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        autolabel(rects1, mappo_stds)
        autolabel(rects2, greedy_stds)
        autolabel(rects3, random_stds)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed, skipping plot.")
    except Exception as e:
        print(f"Error during plotting: {e}")
