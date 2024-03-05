"""
implemented from:
"""

from __future__ import division # '/' always means non-truncating division
from stable_baselines3 import PPO
import copy
import myosuite
import numpy as np
import gym
import gymnasium
from gymnasium import spaces

import time

# Note: Both gym and gymnasium packages are installed, be careful to differentiate them
import numpy as np
import skvideo.io
import os

class ElbowFlex(gymnasium.Env):

    ACT_RANGE = np.array([-1, 1])
    ##DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        #"pose": 1.0,
        #"bonus": 4.0,
        #"act_reg": 1.0,
        #"penalty": 50,
   #}
    global weight
    global target
    global ctrlp
    def __init__(self, episode_limit=1000, seed=0, trainMode =True, weight = 0, target = -1, ctrlp=10):
        self.seed=seed
        self.time_step=1
        self.episode_limit=episode_limit
        #curr_dir = os.getcwd()
        #pathAndModel = 'CodeColab/UE/ElbowFixTarget/Models/'

        #pathAndModel= '/../assets/elbow/myoelbow_1dof6muscles_1dofexo.xml'
        #pathAndModel='myosuite/'

        env_name = 'myoElbowPose1D6MExoRandom-v0'
        """
        Initialize the Myosuite environment with the parameters you want
        """
        self.MyoEnv = gym.make(env_name,normalize_act=False,joint_random_range=(0, 0))
        
        # Joint_random_range set to 0 to prevent initial position randomization.
        # Will causes failures when replaying actions, because initial states are different the env policy was evaluated on
        
        #self.MyoEnv.seed(self.seed)
        self.rng_gen = np.random
        self.rng_gen.seed(self.seed)
        
        """
        Custom Environment initialization for StableBaseline3
        """

        # Define action space
        self.action_space = spaces.Box(low=self.ACT_RANGE[0], high=self.ACT_RANGE[1], shape=(6,), dtype=np.float32)

        # Define your Observation space:
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.train = trainMode
        """
        Target angle for testing
        """

    """
    Stable Baseline 3 required functions
    """
    def reset(self, seed=None, options=None):
        # PUT IN CODE
        if seed is not None:
            self.seed = seed
        self.MyoEnv.seed(seed=self.seed)
        self.MyoEnv.reset() 
        info = {}
        self.MyoEnv.env.sim.data.qpos[0] = 0
        self.MyoEnv.env.sim.data.qvel[0]  = 0
        if self.train:
            self.target_angle= np.random.uniform(high=self.MyoEnv.sim.model.jnt_range[:,1], low=self.MyoEnv.sim.model.jnt_range[:,0])[0]
        else:
            self.target_angle = target
        observation=self.get_obs_vec(self.get_obs_dict(np.zeros(self.action_space.shape[0])))
        self.time_step=1
        return observation, info

    def step(self, action): 
        action = (action+1)/2
        mus_action = np.append(0, action) # exo action is zero for hlthy
        self.MyoEnv.step(mus_action)
        self.obs_dict=self.get_obs_dict(mus_action)
        observation=self.get_obs_vec(self.obs_dict.copy()) 
        terminated = False
        truncated = False
        if self.time_step > self.episode_limit-1: # Alive for episode, less than ep limit -1 because time_step is incremented at the beginning of the step
            truncated = True
        if self.train:
            if (self.obs_dict['error']<np.deg2rad(2)):
                terminated = True
        reward=self.get_reward(terminated, truncated, mus_action)
        self.time_step += 1
         
        info = {}
        #'timestep': self.time_step, 'musc_stim': mus_action
        return observation, reward, terminated, truncated, info


    def get_obs_dict(self, current_action):
        #state = self.MyoEnv.reset
        obs_dict = {}
        obs_dict['elb_angle'] = self.MyoEnv.env.sim.data.qpos[0].copy()
        obs_dict['elb_vel'] = self.MyoEnv.env.sim.data.qvel[0].copy()
        obs_dict['error']=self.MyoEnv.env.sim.data.qpos[0].copy()-self.target_angle
        obs_dict['target']=self.target_angle
        obs_dict['musc_acts']=self.MyoEnv.env.sim.data.act.copy()
        #print(obs_dict['elb_angle'],obs_dict['error'],self.target_angle)
        return obs_dict
    
    def get_obs_vec(self, obs_dict):
        obsvec = np.zeros(0)
        for key in obs_dict.keys():
            obsvec = np.concatenate([obsvec, obs_dict[key].ravel()])
        return np.array(obsvec, dtype=np.float32)
    
     ##CALCULATE REWARD FUNCTION

    """
    Reward functions
    """
    def get_reward_dict(self, terminated, truncated, current_action):
       
        return rew_dict

    def get_reward(self, terminated, truncated, current_action):
        current_angle= self.MyoEnv.env.sim.data.qpos[0].copy()
        reward = np.exp(-1*np.square(current_angle-self.target_angle))
        if terminated:
            reward=reward+100
        return reward

    def set_reward_weights(self, reward_weights):
        # Make sure it is a dictionary of values
        self.reward_wt = reward_weights


