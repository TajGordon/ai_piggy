from collections import defaultdict, deque
from typing import Optional
from gymnasium.spaces import Tuple, Discrete
import gymnasium as gym
import numpy as np
import pigenv
import matplotlib.pyplot as plt

class PigAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        # logging stuff
        self.episode_rewards = []
        self.moving_avg_rewards = deque(maxlen=100)
        self.wins = 0
        self.games_played = 0
        self.episode_lengths = []
        self.max_q_values = []
        self.avg_scores = []

    def get_action(self, obs: tuple[int, int, int, int, int]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # probability epsilon allows to explore environment by making random moves
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, int, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, int, int, int],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * np.exp(-self.epsilon_decay))








# Hyperparameteres
learning_rate = 0.002
n_episodes = 10_000_000
log_window = 10_000
start_epsilon = 1.0
epislon_decay = start_epsilon / (n_episodes * 0.5) # reduce the exploration over time
final_epsilon = 0.1

from tqdm import tqdm

env = gym.make("PigGame-v0")
env = gym.wrappers.RecordEpisodeStatistics(env)

agent = PigAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epislon_decay,
    final_epsilon=final_epsilon,
)


#logging
episode_rewards = []
episode_lengths = []
win_rates = []
epsilons = []

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    #logging
    episode_reward = 0
    episode_length = 0

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, float(reward), terminated, next_obs)

        done = terminated or truncated
        obs = next_obs
        #logging
        episode_reward += float(reward)
        episode_length += 1

    agent.decay_epsilon()

    # Logging
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    agent.games_played += 1
    win_rates.append(agent.wins / agent.games_played)
    epsilons.append(agent.epsilon)

    # Every 1000 episodes, print some stats
    if episode % log_window == 0:
        avg_reward = np.mean(episode_rewards[-log_window:])
        avg_length = np.mean(episode_lengths[-log_window:])
        win_rate = win_rates[-1]
        print("\033[2J\033[H")
        print(f"Episode {episode}")
        print(f"Average Reward (last {log_window}): {avg_reward:.2f}")
        print(f"Average Length (last {log_window}): {avg_length:.2f}")
        print(f"Win Rate: {win_rate:.2f}")
        print(f"Epsilon: {agent.epsilon:.4f}")
        print(f"Max Q-value: {np.max(list(agent.q_values.values())):.2f}")
        print("-------------------------")

# After training, plot the results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.subplot(2, 2, 2)
plt.plot(win_rates)
plt.title('Win Rate')
plt.subplot(2, 2, 3)
plt.plot(episode_lengths)
plt.title('Episode Lengths')
plt.subplot(2, 2, 4)
plt.plot(epsilons)
plt.title('Epsilon Decay')
plt.tight_layout()
plt.show()
