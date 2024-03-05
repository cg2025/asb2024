import numpy as np
import skvideo.io
import os
import pickle
import gym
import matplotlib.pyplot as plt
import myosuite
import mujoco
import glob
import sys
import time
from datetime import datetime

from base64 import b64encode
from IPython.display import HTML
from CodeColab.models.PPO import PPO
#import CodeColab.Evaluation.UE.ElbowFlexPlot as plot
from myosuite.envs.env_variants import register_env_variant
import CodeColab.UE.ElbowFixTarget.Scripts.utils as utils


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

def create_env(exo,sarc):
    env_name_2 = 'elbow-v0'
    env_name_2 = utils.get_env_prefix(exo,sarc)+p['new_model_nm']
    env = gym.make(env_name, normalize_act=False,reset_type='init')
    env.reset()
    return env, env_name_2 

def evaluate2(env_name,  run_id, target_value, weight_value):
    env_name_2 = 'elbow-v0'
    p = utils.init_parameters(train_steps = 1e5)
    env_name_2 = utils.get_env_prefix(False, False)+p['new_model_nm']
    env = gym.make(env_name, normalize_act=False,reset_type='init')
    env.reset()
    file_suffix = ''
    policy_path='CodeColab/UE/ElbowFixTarget/Policies/'+ 'PPO_elbow-v0_tgtrnd.pth' 
    log_path='CodeColab/UE/ElbowFixTarget/logs/'+ 'PPO_elbow-v0_log_tgtrnd.csv'
    #log_path = 'CodeColab/UE/ElbowFixTarget/logs/' 'PPO_' + env_name_2 + "_"+run_id+ weight_value + "_"+ target_value+ "_" +file_suffix+ ".csv" 
    data_f = open('CodeColab/UE/ElbowFixTarget/ResultsData/'+ env_name_2 + "_2"+run_id+ weight_value + "_"+ target_value+ "_" +file_suffix+ ".csv" ,"w")
    state_dim=10
    action_dim=6
    ppo_agent = PPO(state_dim, action_dim, 0, 0, 1, 1, 0.2, True, 0.1)
    count = 0
    t=0
    frames= []
    while count <1:
        obs = env.reset()
        for t in range(1000):
        #frame = vec_env.env.sim.renderer.render_offscreen(camera_id=0)
        #frames.append(frame)
            mus_action =  ppo_agent.select_action(obs) # 6 muscle activations
            mus_action =utils.scaleAction(mus_action,0,1)
            action = np.append(0, mus_action) # exo action is zero for hlthy
            obs, _, _, _ = env.step(action)
            #vec_env.render("human")
            if obs[0][0]!=0:
                data_f.write(str(count)+","+str(t)+","+str(obs[0][0])+","+
                        str(obs[0][3])+ ","+str(np.dot(a,a))+'\n')
            # VecEnv resets automatically
            if done:
                count=count+1
                break   
    data_f.close()
   #plotFlexAngleHealthy()

def plotFlexAngleHealthy():
    a = np.loadtxt('CodeColab/UE/ElbowFixTarget/ResultsData/HealthyResults.csv', delimiter=',')
    plt.figure()
    #print(heights)
    #sprint(energy)
    plt.scatter(a[:,1],a[:,2]) 
    plt.scatter(a[:,1],a[:,3])        
    plt.ylabel("Angle of Flex")
    plt.savefig("CodeColab/UE/ElbowFixTarget/vid_plots/TargetFlexResultsHealthy.png")
    plt.close()



#def evaluate(env_name,  run_id, weight_value, target_value):
def evaluate(env_name,  run_id, target_value):

    env_name_2 = 'elbow-v0'
    p = utils.init_parameters(train_steps = 1e5)
    env_name_2 = utils.get_env_prefix(False, False)+p['new_model_nm']
    env = gym.make(env_name, normalize_act=False,reset_type='init')
    env.reset()
    file_suffix = ''

    # Load policies and logs data
    #policy_path = 'CodeColab/UE/ElbowFixTarget/Policies/'+ 'PPO_' + env_name_2 + "_"+run_id+ weight_value + "_"+ target_value+ "_" +file_suffix+ ".pth"
    policy_path='CodeColab/UE/ElbowFixTarget/Policies/'+ 'PPO_elbow-v0_tgtrnd.pth' 
    log_path='CodeColab/UE/ElbowFixTarget/logs/'+ 'PPO_elbow-v0_log_tgtrnd.csv'
    #log_path = 'CodeColab/UE/ElbowFixTarget/logs/' 'PPO_' + env_name_2 + "_"+run_id+ weight_value + "_"+ target_value+ "_" +file_suffix+ ".csv" 
    data_f = open('CodeColab/UE/ElbowFixTarget/ResultsData/'+ env_name_2 + "_"+run_id+ weight_value + "_"+ target_value+ "_" +file_suffix+ ".csv" ,"w")
    state_dim=10
    action_dim=6

    ppo_agent = PPO(state_dim, action_dim, 0, 0, 1, 1, 0.2, True, 0.1)
    #ppo_agent.load("MyoAssistAgent1")
    frames = []
    ep_time = 2 # in seconds
    num_eps = 50
    ep_angles = []

    if weight_value=="-1":
        weight = np.random.choice(np.arange(1,6),1) # between 1 to 5 kg
    else:
        weight = float(weight_value)
        
    if target_value=="-1":
        target=np.random.uniform(high=env.sim.model.jnt_range[:,1], low=env.sim.model.jnt_range[:,0])
    else:
        target = float(target_value)

    for num_ep in range(num_eps):
        state = env.reset()
        #env.env.sim.model.body_mass[5] = weight *1.0
        #env.env.sim_obsd.model.body_mass[5] = weight * 1.0
        env.env.sim.data.qpos[0] =0
        env.env.sim.forward()
        done = False
        min_error = target
        time_to_min = 0
        energy = 0
        state[0]= env.env.sim.data.qpos[0]
        state[1]= env.env.sim.data.qvel[0]
        obs= np.concatenate((state[:2],state[0]-target,target, state[3:]),axis=None)

        for timestep in range(100):      
            frame = env.sim.renderer.render_offscreen(camera_id=1)
            frames.append(frame)
            # ----------------------
            # Replace with your policy output 
            mus_action =  ppo_agent.select_action(obs) # 6 muscle activations
            mus_action =utils.scaleAction(mus_action,0,1)
            action = np.append(0, mus_action) # exo action is zero for hlthy
            state, _, _, _ = env.step(action)
            error = env.env.sim.data.joint('r_elbow_flex').qpos.item(0)-target
            reward = utils.get_reward(obs[2],error) # obs[2] is the old error
            obs = np.concatenate((state[:2],error,target,state[3:]),axis=None)
            m_e, e_e = calculate_energy(env,False)
            energy+=m_e
            if abs(error) < min_error:
                min_error = abs(error)
                time_to_min = timestep
        data_f.write(str(target)+","+str(weight)+","+str(min_error)+','+
                        str(time_to_min)+','+
                        str(energy)+'\n')
    data_f.close()
    env.close()


def get_frames(env_name, run_id, weight_value, target_value):
    env_name_2 = 'elbow-v0'
    p = utils.init_parameters(train_steps = 1e5)
    env_name_2 = utils.get_env_prefix(False, False)+p['new_model_nm']
    env = gym.make(env_name, normalize_act=False,reset_type='init')
    env.reset()
    file_suffix = '-test'

    # Load policies and logs data
    policy_path='CodeColab/UE/ElbowFixTarget/Policies/'+ 'PPO_elbow-v0_tgtrnd.pth' 
    log_path='CodeColab/UE/ElbowFixTarget/logs/'+ 'PPO_elbow-v0_log_tgtrnd.csv'
    #policy_path = 'CodeColab/UE/ElbowFixTarget/Policies/'+ 'PPO_' + env_name_2 + "_"+run_id+file_suffix+ ".pth" 
    #log_path = 'CodeColab/UE/ElbowFixTarget/logs/' 'PPO_' + env_name_2 + "_"+run_id+file_suffix+ ".csv" 
    data_f = open('CodeColab/UE/ElbowFixTarget/ResultsData/'+ env_name_2 + "_"+run_id+file_suffix+ ".csv" ,"w")
    state_dim=10
    action_dim=6
    print(policy_path)

    ppo_agent = PPO(state_dim, action_dim, 0, 0, 1, 1, 0.2, True, 0.1)
    ppo_agent.load(policy_path)

    frames = []
    ep_time = 4 # in seconds
    #num_eps = 10

    if weight_value=="-1":
        weight = np.random.choice(np.arange(1,6),1) # between 1 to 5 kg
    else:
        weight = float(weight_value)

    target = [float(target_value)]

    for t in target:
        state = env.reset()
        env.env.sim.model.body_mass[5] = weight *1.0
        env.env.sim_obsd.model.body_mass[5] = weight * 1.0
        env.env.sim.data.qpos[0] =0
        env.env.sim.forward()
        done = False
        min_error = t
        time_to_min = 0
        energy = 0
        state[0]= env.env.sim.data.qpos[0]
        state[1]= env.env.sim.data.qvel[0]
        obs= np.concatenate((state[:2],state[0]-t,t,state[3:]),axis=None)

        for timestep in range(1000):      
            frame = env.sim.renderer.render_offscreen(camera_id=0)
            frames.append(frame)
            # ----------------------
            # Replace with your policy output 
            mus_action =  ppo_agent.select_action(obs) # 6 muscle activations
            mus_action = utils.scaleAction(mus_action,0,1)
            action = np.append(0, mus_action) # exo action is zero for hlthy
            state, _, _, _ = env.step(action)
            error = env.env.sim.data.joint('r_elbow_flex').qpos.item(0)-t
            reward = utils.get_reward(obs[2],error) # obs[2] is the old error
            obs = np.concatenate((state[:2],error,t,state[3:]),axis=None)
            m_e, e_e = calculate_energy(env,False)
            print(obs[0])
            energy+=m_e
            if abs(error) < min_error:
                min_error = abs(error)
                time_to_min = timestep
        return frames
    env.close()

def frames2Video(filepath, frames):
    #os.makedirs(dirpath, exist_ok=True)
    # make a local copy
    skvideo.io.vwrite(filepath, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
    #show_video(filepath)


#########################  INPUT Arguments (RUN_ID, WEIGHT, TARGET, TRAIN_STEPS )
run_id = sys.argv[1] # use this to identify the run parameters
weight_value  = sys.argv[2] 
target_value  = sys.argv[3] 

#vid_frames= get_frames(env_name, run_id, weight_value, target_value)
#frames2Video('CodeColab/UE/ElbowFixTarget/vid_plots/healthy'+target_value+'.mp4', vid_frames)

evaluate(env_name,  run_id, weight_value, target_value)