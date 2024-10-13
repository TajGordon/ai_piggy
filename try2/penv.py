import random
from typing import Optional
import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete

def roll_dice():
    return random.randint(1,6)

class PigGameEnv:
    def __init__(self):

        self.my_banked_money = 0
        self.my_ubanked_money = 0
        self.my_wins = 0 # opponents wins = total_rounds - self.my_wins

        self.opp_banked_money = 0
        self.opp_ubanked_money = 0
        self.opp_is_banked = False

        # we'll do 10 boxes, 0-10,10-20,...,90-100, maybe slightly different
        self.observation_space = Tuple(Discrete(10, 0), Discrete(8, 0), Discrete(10, 0), Discrete(8, 0), Discrete(2, 0), Discrete(2, 0))
        self.action_space = gym.spaces.Discrete(2)
    
    def _ubank_to_box(value: int):
        if (value < 10):
            return 0
        if (value < 15):
            return 1
        if (value < 20):
            return 2
        if (value < 25):
            return 3
        if (value < 30):
            return 4
        if (value < 35):
            return 5
        if (value < 40):
            return 6
        else:
            return 7

    def _bank_to_box(value: int):
        if (value < 15):
            return 0
        if (value < 20):
            return 1
        if (value < 30):
            return 2
        if (value < 40):
            return 3
        if (value < 50):
            return 4
        if (value < 55):
            return 5
        if (value < 60):
            return 6
        if (value < 70):
            return 7
        if (value < 80):
            return 8
        else:
            return 9

    def _get_obs(self):
        return {
            self._bank_to_box(self.my_banked_money), self._ubank_to_box(self.my_ubanked_money),
            self._bank_to_box(self.opp_banked_money), self._ubank_to_box(self.opp_ubanked_money),
            1 if (self.my_banked_money + self.my_ubanked_money >= 100) else 0, # little flag the AI hopefully picks up on that tells it that it wins if it banks now
            1 if (self.opp_is_banked) else 0, # just telling the ai whether or not the opponent is banked
        }

    def _get_info(self): # idk what this function really does, might use it later tho
        return {
            "my_banked_money": self.my_banked_money,
            "my_unbanked_money": self.my_ubanked_money,
            "opp_banked_money": self.opp_banked_money,
            "opp_unbanked_money": self.opp_ubanked_money,
        }

    # Resets the game/episode/environment to run again
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # we need the following line to seed self.np_random
        super().reset(seed=seed)

        self.my_banked_money = 0
        self.my_ubanked_money = 0

        self.opp_banked_money = 0
        self.opp_ubanked_money = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # step function is basically just an iteration of the loop, kinda, its a step
    # in step function we compute the agents reward for their action
    def step(self, action):
        
        if action == 1: # bank is 1, continue is 0
            self.my_banked_money += self.my_ubanked_money
            self.my_ubanked_money = 0
            if not self.opp_is_banked:
                rolled_a_one = False
                while not rolled_a_one and self.opp_ubanked_money < 15:
                    roll = roll_dice()
                    if (roll == 1):
                        rolled_a_one = True
                        self.opp_ubanked_money = 0
                    else:
                        self.opp_ubanked_money += roll
                # No longer rolling, either rolled a one or opp banked
                self.opp_banked_money += self.opp_ubanked_money # 0 if rolled a one
                self.opp_ubanked_money = 0
        else: # if continuing
            roll = roll_dice()
            if (roll == 1):
                self.my_ubanked_money = 0
                if not self.opp_is_banked:
                    self.opp_ubanked_money = 0
            else:
                self.my_ubanked_money += roll
                if not self.opp_is_banked:
                    self.opp_ubanked_money += roll
                    if (self.opp_ubanked_money >= 15):
                        self.opp_is_banked = True
                        self.opp_banked_money += self.opp_ubanked_money 
                        self.opp_ubanked_money = 0

        terminated = self.my_banked_money >= 100 or self.opp_banked_money >= 100
        truncated = False
        # wonderful python syntax (this might be my fault)
        reward = 0 if not terminated else 1 if self.my_banked_money >= 100 else -1
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

# Registering the bot
gym.register(
    id="PigGame-v0",
    entry_point=PigGameEnv,
)