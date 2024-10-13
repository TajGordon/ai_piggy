import numpy as np
from numba import njit
from collections import defaultdict
from typing import Tuple, Dict, List
import gymnasium as gym
import pigenv
from multiprocessing import Pool
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

@njit
def update_q_value(q_value: float, lr: float, reward: float, discount_factor: float, next_max_q: float, current_q: float) -> float:
    """
    Optimized Q-value update function.
    """
    return q_value + lr * (reward + discount_factor * next_max_q - current_q)

def default_q_value():
    return np.zeros(2, dtype=np.float32)

class PigAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(default_q_value)
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # For monitoring performance
        self.episode_rewards = []
        self.average_q_values = []
        self.win_rates = []

    def get_action(self, obs: Tuple[int, int, int, int, int]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[obs]))

    def update(self, obs: Tuple[int, int, int, int, int], action: int, reward: float, terminated: bool, next_obs: Tuple[int, int, int, int, int]):
        future_q_value = 0 if terminated else np.max(self.q_values[next_obs])
        self.q_values[obs][action] = update_q_value(
            self.q_values[obs][action],
            self.lr,
            reward,
            self.discount_factor,
            future_q_value,
            self.q_values[obs][action]
        )

    def decay_epsilon(self, episode: int):
        self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * np.exp(-1. * episode / self.epsilon_decay)

    def decay_learning_rate(self, episode: int):
        self.lr = self.initial_lr * (1 / (1 + 0.001 * episode))

    def record_performance(self, episode_reward: float, win: bool):
        self.episode_rewards.append(episode_reward)
        self.average_q_values.append(np.mean([np.max(q) for q in self.q_values.values()]))
        if not self.win_rates:
            self.win_rates.append(1 if win else 0)
        else:
            self.win_rates.append(self.win_rates[-1] * 0.99 + (1 if win else 0) * 0.01)

def train_agent(args: Tuple[gym.Env, PigAgent, int]) -> PigAgent:
    env, agent, n_episodes = args
    for episode in trange(n_episodes, desc="Training", leave=False):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                reward = 1 if reward > 0 else -1
            else:
                reward = 0

            agent.update(obs, action, reward, terminated, next_obs)
            episode_reward += reward
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon(episode)
        agent.decay_learning_rate(episode)
        agent.record_performance(episode_reward, reward > 0)

    return agent

def parallel_train(n_processes: int, total_episodes: int, env_id: str, agent_params: Dict) -> List[PigAgent]:
    envs = [gym.make(env_id) for _ in range(n_processes)]
    agents = [PigAgent(env, **agent_params) for env in envs]
    episodes_per_process = total_episodes // n_processes

    with Pool(n_processes) as p:
        trained_agents = list(tqdm(
            p.imap(train_agent, [(env, agent, episodes_per_process) for env, agent in zip(envs, agents)]),
            total=n_processes,
            desc="Overall Progress"
        ))

    return trained_agents

def evaluate_agent(agent: PigAgent, env: gym.Env, n_episodes: int = 1000) -> float:
    wins = 0
    for _ in trange(n_episodes, desc="Evaluating", leave=False):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward > 0:
            wins += 1
    return wins / n_episodes

def plot_performance(agent: PigAgent):
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.plot(agent.episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(3, 1, 2)
    plt.plot(agent.average_q_values)
    plt.title('Average Q-Values')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-Value')

    plt.subplot(3, 1, 3)
    plt.plot(agent.win_rates)
    plt.title('Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_processes = 4
    total_episodes = 1_000_000
    env_id = "PigGame-v0"
    agent_params = {
        "learning_rate": 0.1,
        "initial_epsilon": 1.0,
        "epsilon_decay": total_episodes / 5,
        "final_epsilon": 0.01,
        "discount_factor": 0.99
    }

    print("Starting training...")
    trained_agents = parallel_train(n_processes, total_episodes, env_id, agent_params)
    print("Training completed.")

    print("Evaluating agents...")
    eval_env = gym.make(env_id)
    best_agent = max(trained_agents, key=lambda agent: evaluate_agent(agent, eval_env))
    average_win_rate = evaluate_agent(best_agent, eval_env)
    print(f"Best agent average win rate: {average_win_rate:.2f}")

    plot_performance(best_agent)

    print("Sample Q-values:")
    for obs, q_values in list(best_agent.q_values.items())[:5]:
        print(f"State: {obs}, Q-values: {q_values}")
