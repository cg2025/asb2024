# Authors: Chun Kwang Tan <cktan.neumove@gmail.com>
"""
implemented from:
"""

from __future__ import division # '/' always means non-truncating division

import copy
import myosuite
import numpy as np
import gym
import gymnasium
from gymnasium import spaces

# Note: Both gym and gymnasium packages are installed, be careful to differentiate them
import numpy as np
import skvideo.io
import os

class ElbowFlex(gymnasium.Env):

    ACT_RANGE = np.array([-1, 1])
    REF_GAIN_RANGE = np.array([-1, 1])

    CONTROL_PARAM = []
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        'alive_rew': 1.0,
        'action_penalty_zero': 1.0,
        #'action_penalty': 1.0,
        #'terminated': 1,
        'footstep': 1,
        'effort': 1,
        'v_tgt': 1,
    }

    def __init__(self, init_pose='walk', dt=0.01, mode='2D', episode_limit=2000, seed=0, 
                 target_vel=np.ones(2,)*-1,):
        """
        sine_vel_args : A dictionary of paramters to generate the a sine target velocity curve
            min vel, max_vel: Used to calcualte the center velocity of the wave. 

        """

        pathAndModel = '/myosuite/simhive/myo_sim/elbow/assets/myoelbow_1dof6muscles_1dofexo_body.xml'
        env_name = 'myoElbowPose1D6MExoRandom-v0'
        
        """
        Initialize the Myosuite environment with the parameters you want
        """
        self.MyoEnv = gym.make(env_name, 
                            normalize_act=False,
                            muscle_condition='sarcopenia',
                            reset_type='init',
                            )
        

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        """
        Target angle for testing
        """
        ####self.tgt_vel_mode = tgt_vel_mode


    """
    Stable Baseline 3 required functions
    """
    def reset(self, seed=None, options=None):

        # Randomizing linear target angles for training
        #If a target angle is passed in, then use the input target
        if self.tgt_agnle_mode == 'train':
            self.target_angle_randomizer = self.rng_gen.randint(low=1, high=5) # low (inclusive) to high (exclusive)
            self.randomize_target_vel_params()

        self.prev_action = np.zeros(self.action_space.shape)
        
        if seed is not None:
            self.seed = seed

        # Reset the Myosuite environment and Reflex Controller
        self.MyoEnv.reset(seed=self.seed)
        self.ElbowFlex.reset()

        self.set_init_pose(self.init_pose) # Internall calls a forward() after setting initial pose
        self.reset_control_params()
        self.time_step = 0

        self.vtgt.reset(version=self.tgt_field_ver, seed=self.seed)

        if self.tgt_vel_mode == 'train':
            self.randomize_init_state()

        body_xpos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()
        pelvis_euler = self._get_pel_angle()

        pose = np.array([body_xpos[0], body_xpos[1], pelvis_euler[2]]) # No negative yaw, local velocity field is internally inverted
        self.v_tgt_field, self.flag_new_v_tgt_field = self.vtgt.update(pose)

        self.obs_dict = self.get_obs_dict(np.zeros(self.action_space.shape))
        observation = self.get_obs_vec(self.obs_dict.copy())
        info = {}

        self.init_reward_1()
        self.footstep['new'] = False
        self.footstep['r_contact'] = True
        self.footstep['l_contact'] = True

        # Debug lines
        self.avg_vel = 0
        self.step_vel = 0

        return observation, info

    def step(self, action):
        
        mus_action = np.append(0, action) # exo action is zero for hlthy
        state, _, _, _ = self.MyoEnv.step(mus_action)
        terminated = False
        truncated = False
        self.time_step += 1
        error = self.MyoEnv.env.sim.data.joint('r_elbow_flex').qpos.item(0)-self.get_target_angle()
        reward = utils.get_reward(obs[2],error) # obs[2] is the old error
        obs = np.concatenate((state[:2],error,state[3:]),axis=None)#np.append(state,target_angle) # this is 9 dimension does not include exo

        # ----- Checking for ending conditions -----
        if self.time_step > self.episode_limit-1: # Alive for episode, less than ep limit -1 because time_step is incremented at the beginning of the step
            terminated = True

        # ----- Obs and rewards after all updates -----
        self.obs_dict = self.get_obs_dict(action)
        observation = self.get_obs_vec(self.obs_dict.copy()) # Added the action as part of observation

        info = {'timestep': self.time_step, 'musc_stim': mus_action}

        return observation, reward, terminated, truncated, info

    def get_obs_dict(self, current_action):

        obs_dict = {}
        
        #obs_dict['v_tgt_field'] = np.ndarray.flatten(self.v_tgt_field.copy())
        obs_dict['v_tgt'] = self.get_target_vel()

        obs_dict['r_elbow_flex'] = self.MyoEnv.env.sim.data.joint('r_elbow_flex').qpos.copy()
        obs_dict['r_elbow_flex_qvel'] = self.MyoEnv.env.env.sim.data.qvel[0].copy()

        if self.obs_param is not None:
            if 'mus_act' in self.obs_param:
                obs_dict['mus_act'] = self._get_muscle_act()

        return obs_dict

    def get_obs_vec(self, obs_dict):
        obsvec = np.zeros(0)
        for key in obs_dict.keys():
            #print(f"key : {key}")
            if key == 'r_leg' or key == 'l_leg':
                for subkey in obs_dict[key]:
                    #print(f"key : {key}_{subkey}")
                    obsvec = np.concatenate([obsvec, obs_dict[key][subkey].ravel()])
            else:
                obsvec = np.concatenate([obsvec, obs_dict[key].ravel()]) # ravel helps with images
        return np.array(obsvec, dtype=np.float32)

    """
    Reward functions
    """

    def get_reward_dict(self, terminated, truncated, current_action):

        rew_dict = {}
        rew_dict['alive_rew'] = 0.1 # Scale with weights in the reward_dict_weight
        rew_dict['v_tgt'] = self._get_vel_reward(self.get_pel_xvel())
        rew_dict['effort'] = -1*( np.sum(self._get_muscle_act()**2) / self.muscle_space ) # max possible -1
        rew_dict['action_penalty_zero'] = -1*np.mean( np.square(current_action) ) #np.mean(np.exp(-1000 * current_action**2)-0.2) # max possible -0.2
        rew_dict['action_penalty'] = -1*np.mean( np.abs(self.prev_action.copy() - current_action) ) # Max possible -2
        if terminated and self.time_step > self.episode_limit-1:
            rew_dict['terminated'] = 10.0
        elif truncated and self.time_step < self.episode_limit:
            rew_dict['terminated'] = -10.0
        else:
            rew_dict['terminated'] = 0.0

        return rew_dict

    def get_reward(self, reward_dict):
        
        #reward_dict = self.get_reward_dict(truncated, obs_dict)
        reward = np.sum([wt*reward_dict[key] for key, wt in self.reward_wt.items()], axis=0)
        
        return reward

    def set_reward_weights(self, reward_weights):
        # Make sure it is a dictionary of values
        self.reward_wt = reward_weights

    def get_reward_dict_old(self, terminated, truncated, current_action):

        reward = 0
        dt = 0.01
        rew_dict = {}

        # alive reward
        # should be large enough to search for 'success' solutions (alive to the end) first
        rew_dict['alive_rew'] = self.d_reward['alive']
        # effort ~ muscle fatigue ~ (muscle activation)^2 
        # Metabolic cost
        ACT2 = 0
        temp_leg = ['r_leg', 'l_leg']
        temp_act = self._get_muscle_act()
        
        for leg in temp_leg:
            for MUS in self.muscles_dict[leg].keys():
                # np.sum used here because there are multiple muscles in each "bundle"
                ACT2 += np.sum(np.square( temp_act[self.muscles_dict[leg][MUS]] ))
        
        # Accumulates from timesteps, no reward if no new step, so zero
        rew_dict['footstep'] = 0
        rew_dict['v_tgt'] = 0
        rew_dict['effort'] = 0
        rew_dict['terminated'] = 0

        # Accumulator
        self.d_reward['footstep']['effort'] += ACT2*dt
        self.d_reward['footstep']['del_t'] += dt
        # reward from velocity (penalize from deviating from v_tgt)
        self.d_reward['footstep']['del_v'] += self._get_vel_diff(self.get_pel_xvel())*dt
        self.step_vel += self.get_pel_xvel()*dt
        self.avg_vel = 0

        # footstep reward (when made a new step)
        if self.footstep['new']:

            if self.d_reward['footstep']['del_t'] == 0:
               self.avg_vel = 0
            else:
               self.avg_vel = (self.step_vel / self.d_reward['footstep']['del_t'])[0]

            #print(f"Timestep: {self.time_step} , Avg del_v: {self.d_reward['footstep']['del_v']}, average step vel: {self.avg_vel}, Pel Pos: {self.MyoEnv.unwrapped.sim.data.body('pelvis').xpos.copy()[0]}")
            #print(f"Del_t :{self.d_reward['footstep']['del_t']} , Calc avg vel: {self.step_vel / self.d_reward['footstep']['del_t']}, Calc diff : {self.d_reward['footstep']['del_v'] / self.d_reward['footstep']['del_t']}")
            # footstep reward: so that solution does not avoid making footsteps
            # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
            reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

            # deviation from target velocity
            # the average velocity a step (instead of instantaneous velocity) is used
            # as velocity fluctuates within a step in normal human walking
            
            # Scale reward by velocity difference distance to zero
            reward_footstep_v = -1*self.d_reward['weight']['v_tgt']*(np.linalg.norm(self.d_reward['footstep']['del_v']))
            
            #avg_diff = self.d_reward['footstep']['del_v'] / self.d_reward['footstep']['del_t']
            #reward_footstep_v = -1*self.d_reward['weight']['v_tgt']*(np.linalg.norm(avg_diff))
            #reward_footstep_v = self.d_reward['weight']['v_tgt']*( np.sum( np.exp(-np.square(avg_diff))*0.1 ) )

            # panalize effort
            reward_footstep_e = -1*self.d_reward['weight']['effort']*self.d_reward['footstep']['effort']

            self.d_reward['footstep']['del_t'] = 0
            self.d_reward['footstep']['del_v'] = 0
            self.d_reward['footstep']['effort'] = 0

            #reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e
            rew_dict['footstep'] = reward_footstep_0
            rew_dict['v_tgt'] = reward_footstep_v
            rew_dict['effort'] = reward_footstep_e

            self.step_vel = 0
            #print(f"Step made: {self.time_step}, step rew: {rew_dict['footstep']}")
            #print(f"In step: Terminated: {terminated}, trunc: {truncated}")
        # success bonus
        if terminated and (truncated == False): #and self.failure_mode is 'success':
            # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
            #reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
            #reward += reward_footstep_0 + 100
            #reward += self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t'] + 10
            #rew_dict['terminated'] = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t'] + 10
            rew_dict['footstep'] += self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
            #print(f"timestep: {self.time_step},  Terminated_footstep: {rew_dict['footstep']}")
            #print(f"In terminated: Terminated: {terminated}, trunc: {truncated}")
            #print(f"Terminated: {rew_dict['terminated']}")

        # Action delta penalty
        #reward += np.mean(np.exp(-1000 * current_action**2)-0.2)
        # Both are calculated, but only 1 penalty is applied during reward calculation
        rew_dict['action_penalty_zero'] = -0.1*np.mean( np.square(current_action) ) #np.mean(np.exp(-1000 * current_action**2)-0.1)
        rew_dict['action_penalty'] = -1*np.mean( np.abs(self.prev_action.copy() - current_action) )
        
        #self.debug_reward_dict.append(copy.deepcopy(rew_dict))
        self.debug_reward_dict = copy.deepcopy(rew_dict)
        self.debug_actions = current_action.copy()

        return rew_dict

    """
    Environment and state randomization functions
    """
    def randomize_init_state(self):
        rnd_state_idx = self.rng_gen.randint(low=0, high=self.init_rnd_state_len)

        # Randomize the index based on total number
        selected_state = copy.deepcopy(self.init_state_list[rnd_state_idx])
        selected_reflex = copy.deepcopy(self.init_reflex_list[rnd_state_idx])

        self.set_reflex_env_state(selected_state)
        self.ReflexCtrl = selected_reflex

    def randomize_target_vel_params(self):
        # Randomizing target velocities for training
        #If a target velocity is passed in, then use the input target

        """
        Randomize velocities based on percentage
        """
        if self.target_vel_randomizer in np.array([1,2]):
            self.target_x_vel = np.clip( np.round(self.rng_gen.uniform(low=0.8, high=1.81), 2), 0.8, 1.8)
            self.target_y_vel = 0
            self.target_vel_type = 'constant'
        
        elif self.target_vel_randomizer == 3:
            diff = self.rng_gen.uniform(low=0.1, high=0.81)
            min_tgt = self.rng_gen.uniform(low=0.8, high=1.0)
            sorted_tgt = np.array([min_tgt, min_tgt+diff])

            if self.rng_gen.randint(low=0, high=2) == 0:
                sorted_tgt = sorted_tgt[::-1]

            self.single_change_vel = sorted_tgt
            self.single_change_time = self.rng_gen.randint(low=5, high=15) * 100

            self.target_vel_type = 'constant_change'

        elif self.target_vel_randomizer == 4:
            rnd_vel = np.sort(np.clip( np.round(self.rng_gen.uniform(low=0.8, high=1.81, size=2), 2), 0.8, 1.8))
            self.sine_min = rnd_vel[0]
            self.sine_max = rnd_vel[1]
            self.sine_period = self.rng_gen.randint(low=5, high=21) * 100 # simulation timestep is 0.01 sec.
            self.phase_shift = self.rng_gen.randint(low=0, high=11) * 100
            self.target_vel_type == 'sine'

        #print(self.target_vel_type)

    def get_target_angle(self):
        #world_com_xpos = self.MyoEnv.sim.data.body('pelvis').xpos.copy()
        #v_tgt = self.vtgt.get_vtgt(world_com_xpos[0:2]).T

        if self.tgt_vel_mode == 'eval':
            return np.array([self.eval_x_vel, self.eval_y_vel])
            
        elif self.target_vel_type == 'constant':
            return np.array([self.target_x_vel, self.target_y_vel])

        elif self.target_vel_type == 'sine':
            return self.get_sinusoidal_vel(self.time_step)

        elif self.target_vel_type == 'constant_change':

            target_x_vel = self.single_change_vel[0]
            if self.time_step == self.single_change_time:
                target_x_vel = self.single_change_vel[1]

            return np.array([target_x_vel, 0])

    def get_sinusoidal_vel(self, current_time):
        """
        Compute the value of a sine wave at a specific time.
        Current time: Given in milliseconds
        """
        #phase_shift = 0
        
        amplitude = (self.sine_max - self.sine_min) / 2
        offset = (self.sine_min + self.sine_max) / 2

        frequency = 1 / self.sine_period
        value = amplitude * np.sin(2 * np.pi * frequency * current_time + self.phase_shift) + offset

        return np.array([value, 0]) # Currently olny for 2D walking

    """
    Utility functions
    """

    def _get_vel_diff(self, current_pel_vel):

        vel_diff = (current_pel_vel - self.get_target_vel())

        return vel_diff

    def _get_vel_reward(self, current_pel_vel):
    
        #vel_diff = self._get_vel_diff(current_pel_vel)
        tgt_vel = self.get_target_vel()
    
        return np.exp(-np.square(tgt_vel[1] - current_pel_vel[1])) + np.exp(-np.square(tgt_vel[0] - current_pel_vel[0]))

    def _get_pos_reward(self):
        """
        Reward for how close agent is to the goal
        """
        pel_xpos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()
        # np.exp(-np.square(self.target_y_pose - pel_xpos[1])) + np.exp(-np.square(self.target_x_pose - pel_xpos[0]))
        return np.exp(-np.square(self.target_x_pose - pel_xpos[0]))

    def _get_distance_reward(self):
        pel_xpos = self.MyoEnv.env.sim.data.body('pelvis').xpos.copy()
        # Returns distance travelled in x dimension
        return pel_xpos[0] - 0

    def init_reward_1(self):
        self.d_reward = {}

        self.d_reward['weight'] = {}
        self.d_reward['weight']['footstep'] = 10
        self.d_reward['weight']['effort'] = 1
        self.d_reward['weight']['v_tgt'] = 1
        self.d_reward['weight']['v_tgt_R2'] = 3

        self.d_reward['alive'] = 0.1 # Increased it from 0.1 due to addition of action penalty
        self.d_reward['effort'] = 0

        self.d_reward['footstep'] = {}
        self.d_reward['footstep']['effort'] = 0
        self.d_reward['footstep']['del_t'] = 0
        self.d_reward['footstep']['del_v'] = 0

    """
    Initialization functions
    """

    def set_init_pose(self, key_name='stand'):
        self.MyoEnv.env.sim.data.qpos = self.MyoEnv.env.sim.model.keyframe(key_name).qpos
        self.MyoEnv.env.sim.data.qvel = self.MyoEnv.env.sim.model.keyframe(key_name).qvel
        self.MyoEnv.env.forward()

    def adjust_initial_pose(self, joint_dict):
        """
        Function allows for additional adjustment of the joint angles from the pre-defined named poses
        """
        # Values in radians
        for joint_name in joint_dict['joint_angles'].keys():
            self.MyoEnv.env.sim.data.joint(joint_name).qpos[0] = joint_dict['joint_angles'][joint_name]

        self.MyoEnv.env.forward()
