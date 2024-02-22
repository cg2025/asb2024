import numpy as np
import skvideo.io
import os
import pickle
import gym
import matplotlib.pyplot as plt
import myosuite
import mujoco
import glob
import time
from datetime import datetime

from base64 import b64encode
from IPython.display import HTML
from CodeColab.models.PPO import PPO
import CodeColab.Evaluation.UE.ElbowFlexPlot as plot
from myosuite.envs.env_variants import register_env_variant


def show_video(video_path, video_width = 500):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")

def calculate_energy(env,exo):
    a = abs(env.sim.data.act)
    exo_energy = np.zeros((1))
    if exo:
        exo_energy= env.sim.data.actuator('Exo').force
    return np.dot(a,a), np.linalg.norm(exo_energy)

time_step = 0.01 # Simulation timestep, don't change this
env_name = 'myoElbowPose1D6MExoRandom-v0'
env_type = 'UE'
model_nm = 'Exp1'

def create_env(exo,sarc):
    env_name_2 = 'elbow-v0'
    if exo and sarc:
        env_name_2 = 'exosarc'+env_name_2
        register_env_variant( env_id= env_name, variants={'normalize_act':False, 'muscle_condition':'sarcopenia'}, variant_id= env_name_2, silent=False )
        env = gym.make(env_name_2)
    elif exo and not(sarc):
        env_name_2 ='exo'+env_name_2
        register_env_variant( env_id= env_name, variants={'normalize_act':False}, variant_id= env_name_2,silent=False )
        env = gym.make(env_name_2)
    elif not(exo) and sarc:
        env_name_2 ='sarc'+env_name_2
        register_env_variant( env_id= env_name, variants={'normalize_act':False, 'muscle_condition':'sarcopenia'}, variant_id= env_name_2,silent=False)
        env = gym.make(env_name_2)
    else:
        env_name_2  = 'norm' + env_name_2
        register_env_variant( env_id= env_name, variants={'normalize_act':False}, variant_id= env_name_2,silent=False )
        env = gym.make(env_name_2)
    return env, env_name_2 


def evaluate(env_name, train_steps, run_id, weight_value, target_value, ctrl_value, registered):
    env, env_name_2 = create_env(exo,sarc)
    file_suffix = ''
    run_id = str(weight)
    p = utils.init_parameters(train_steps = train_steps)
    if ctrl_adjust:
        file_suffix = '_ctrl_'+str(ctrl_value)
    # Load policies and logs data
    policy_path = 'CodeColab/UE/ElbowFixTarget/Policies/'+ 'PPO_' + env_name_2+'_0_'+run_id+file_suffix+'.pth'
    #policy_path='CodeColab/Policies/UE/Exp1/PPO_1normelbow-v10_0_15.pth'
    log_path = 'CodeColab/UE/ElbowFixTarget/logs/' +env_name+'_'+run_id+file_suffix+'.csv'
    data_f = open('CodeColab/ResultsData/'+path+env_name+'_'+run_id+file_suffix+'.csv',"w")
    #state_dim = 7
    state_dim=9
    action_dim=1
    #action_dim = 2
    # Load existing policy for healthy
    #env_name_2  = 'normelbow-v0'
    #register_env_variant( env_id= env_name, variants={'normalize_act':False}, variant_id= env_name_2,silent=False )
    #hlthy_env= gym.make(env_name_2)
    hlthy_policy_path = 'CodeColab/Policies/UE/Exp1/' + 'PPO_1normelbow-v10_0_15.pth'
    #hlthy_state_dim = hlthy_env.observation_space.shape[0]+1
    hlthy_state_dim=9
    hlthy_action_dim=6
    #hlthy_action_dim = hlthy_env.action_space.shape[0]
    hlthy_agent = PPO(hlthy_state_dim, hlthy_action_dim, 0, 0, 1, 1, 0.2, True, 0.1)
    hlthy_agent.load(hlthy_policy_path)

    ppo_agent = PPO(state_dim, action_dim, 0, 0, 1, 1, 0.2, True, 0.1)
    ppo_agent.load(policy_path)

    o =env.reset()
    if ctrl_adjust:
        env.sim.model.actuator('Exo').ctrlrange = np.array((-1*ctrl_value,ctrl_value))
    frames = []
    # Adjust the episode time as required. You may need a longer time to render when the jumping height increases
    ep_time = 2 # in seconds
    #ep_time/time_step
    num_eps = 1000
    ep_angles = []
    for num_ep in range(num_eps):
        # Resetting the env
        obs = env.reset()
        env.env.sim.model.body_mass[5] = weight *1.0
        env.env.sim_obsd.model.body_mass[5] = weight * 1.0
        env.env.sim.data.qpos[0] =0
        env.env.sim.forward()
        best_angle = -5
        time_to_best = 0
        energy = 0
        exoenergy = 0
        max_energy =0
        for timestep in range(100):      
            #frame = env.sim.renderer.render_offscreen(camera_id=1)
            #frames.append(frame)
            # ----------------------
            # Replace with your policy output
            obs = env.reset()
            mus_act =  hlthy_agent.select_action(obs) # 6 muscle activations
            obs_new = np.append(obs[:3],mus_act)
            exo_act =  ppo_agent.select_action(obs_new)
            action = np.append(exo_act, mus_act) # exo action is zero for hlthy

            #obs = np.append(obs, env.sim.data.joint('r_elbow_flex').qpos[0])
            #hlthy_action =  hlthy_agent.select_action(obs)
            #exo_action = ppo_agent.select_action(hlthy_action)
            #hlthy_action[0] = exo_action[0]
            #  action = hlthy_action
            state, reward, done, _ = env.step(action)                
            obs = state
            m_e, e_e = calculate_energy(env,exo)
            energy+=m_e
            if exo:
                exoenergy+=e_e
            if env.env.sim.data.joint('r_elbow_flex').qpos.item(0) > best_angle:
                best_angle = env.env.sim.data.joint('r_elbow_flex').qpos.item(0)
                time_to_best = timestep
                max_exoenergy = exoenergy
                max_energy = energy

        data_f.write(str(best_angle)+','+
                        str(time_to_best)+','+
                        str(max_energy)+','+
                        str(max_exoenergy)+'\n')
    data_f.close()
    env.close()

#weight = 15
#evaluate(exo = False, sarc = False, weight = weight)