from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import penv as penv

class PigGameAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.96,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def get_action(self, obs: tuple[int, int, int, int, int, int]) -> int:
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update(
            self,
            obs: tuple[int, int, int, int, int, int],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, int, int, int, int],
    ):
        future_q_value = (not terminated) * np.map(self.q_values[obs][action])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon ** -self.epislon_decay)


# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2) # reduce exloration over time
final_epsilon = 0.1

env = gym.make("PigGame-v0")
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

agent = PigGameAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=0.99,
)

# logging
# episode_rewards = []
# episode_lengths = []
games_won = 0
games_lost = 0
win_rates = []
# epsilons  = []

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    episode_length = 0

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs=obs, action=action, reward=reward, terminted=terminated, next_obs=next_obs)

        # udpate if the environment is done and the current obs # tf does this mean
        done = terminated or truncated 
        obs = next_obs

        # Logging
        if reward == 1.0 and terminated:
            games_won += 1
        elif reward == -1.0 and terminated:
            games_lost += 1
        episode_length += 1
    
    agent.decay_epsilon()
    win_rates.append(games_won / (games_won + games_lost))



# Visualing, code yoinked from tutorial
rolling_length = 500
fig, axs = plt.subplots(ncols=4, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()

# me adding stuff
axs[3].set_title("Win Rate")
axs[3].plot(win_rates)

plt.show()