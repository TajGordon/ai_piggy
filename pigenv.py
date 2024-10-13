from typing import Optional
import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete

class PigGameEnv(gym.Env):
    def __init__(self):
        self._banked_money = 0
        self._unbanked_money = 0

        # right now opponent is a bank at 15 guy
        # will implement adding custom players later
        self._op_banked_money = 0
        self._op_unbanked_money = 0
        self._op_is_banked = False

        # tuple of 4 elements, our coins, unbanked coins, their coins, their unbanked coins
        self.observation_space = Tuple((Discrete(n=8, start=0), Discrete(n=8, start=0), Discrete(n=8, start=0), Discrete(n=8, start=0), Discrete(n=2, start=0)))
        self.action_space = gym.spaces.Discrete(2) # bank (1) or continue (0)
        self.action_to_word = ['continue', 'bank']

    def _banked_money_to_bucket(self, value):
        if (value < 20):
            return 0
        elif (value < 30):
            return 1
        elif (value < 42):
            return 2
        elif (value < 55):
            return 3
        elif (value < 62):
            return 4
        elif (value < 70):
            return 5
        elif (value < 80):
            return 6
        else:
            return 7

    def _unbanked_money_to_bucket(self, value):
        if (value < 10):
            return 0
        elif (value < 15):
            return 1
        elif (value < 21):
            return 2
        elif (value < 25):
            return 3
        elif (value < 30):
            return 4
        elif (value < 35):
            return 5
        elif (value < 45):
            return 6
        else:
            return 7

    def _get_obs(self):
        return (self._banked_money_to_bucket(self._banked_money), self._unbanked_money_to_bucket(self._unbanked_money),
            self._banked_money_to_bucket(self._op_banked_money), self._unbanked_money_to_bucket(self._op_unbanked_money),
            1 if (self._banked_money + self._unbanked_money >= 100) else 0)

    def _get_info(self):
        return {
            "distance_banked": self._banked_money - self._op_banked_money,
            "distance_unbanked": self._unbanked_money - self._op_unbanked_money,
            "distance_total": self._banked_money + self._unbanked_money - self._op_banked_money - self._op_unbanked_money,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self._banked_money = 0
        self._unbanked_money = 0
        self._op_banked_money = 0
        self._op_unbanked_money = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        # the opponent's decision (just bank at 15 for now)
        if (self._op_unbanked_money >= 15 or self._op_unbanked_money + self._op_banked_money >= 100):
            self._op_banked_money += self._op_unbanked_money
            self._op_unbanked_money = 0
            self._op_is_banked = True

        roll = 0

        if (self.action_to_word[action] == 'bank'):
            self._banked_money += self._unbanked_money
            self._unbanked_money = 0
            if not (self._op_is_banked):
                # emulate the other player rolling a bunch
                rolled_a_one = False
                while self._op_unbanked_money < 15 and self._op_unbanked_money + self._op_banked_money < 100 and not rolled_a_one:
                    roll = random.randint(1, 6)
                    if roll == 1:
                        rolled_a_one = True
                        self._op_unbanked_money = 0
                    else:
                        self._op_unbanked_money += roll
                self._op_banked_money += self._op_unbanked_money
                self._op_unbanked_money = 0
            # reset their banking status, no notion of rounds really...
            self._op_is_banked = False
        else:
            roll = random.randint(1, 6)
            # if roll is a one, then we just set the unbanked money to 0 for both players,
            if roll == 1:
                self._unbanked_money = 0
                self._op_unbanked_money = 0
                self._op_is_banked = False
            else:
                self._unbanked_money += roll
                # only add money if they aren't banked, reset their bank status when we bank or when a 1 is rolled
                if not self._op_is_banked:
                    self._op_unbanked_money += roll


        terminated = self._banked_money >= 100 or self._op_banked_money >= 100
        truncated = False

        if terminated:
            if self._banked_money >= 100:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = (self._banked_money - self._op_banked_money * 0.8)/1000.0 + self._unbanked_money / 1000.0
        if roll == 1:
            reward -= 0.05
        else:
            reward += roll / 100.0

        observation = self._get_obs()
        info = self._get_info()



        return observation, reward, terminated, truncated, info

gym.register(
    id="PigGame-v0",
    entry_point="pigenv:PigGameEnv",
)
