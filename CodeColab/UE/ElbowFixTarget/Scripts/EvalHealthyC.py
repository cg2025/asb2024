import gymnasium
import gym
import argparse
import numpy as np
import skvideo.io
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import CheckpointCallback

def frames2Video(filepath, frames):
    #os.makedirs(dirpath, exist_ok=True)
    # make a local copy
    skvideo.io.vwrite(filepath, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
    #show_video(filepath)

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

parser = argparse.ArgumentParser()
parser.add_argument("--timestep", type=int, help="(int) Number of training timesteps")
parser.add_argument("--chk_freq", type=int, help="(int) Checkpoint Frequency (in timesteps)")
input_args = parser.parse_args()

register(id="myoElbow-v0", entry_point="CodeColab.UE.ElbowFixTarget.Scripts.Interface:ElbowFlex", max_episode_steps=1000)
env=gymnasium.make("myoElbow-v0", weight = 0, trainMode = False)


model = PPO("MlpPolicy", env, verbose=1)
model.load("MyoAssistAgent1")

data_f = open('CodeColab/UE/ElbowFixTarget/ResultsData/HealthyResults.csv' ,"w")
#data_f.write("Episode, time, Angle, Target Angle, Energy\n")
#vec_env = model.get_env()
frames= []
for nEval in range(10):
    obs,_ = env.reset()
    env.unwrapped.target = 1.2+nEval/10
    #env.unwrapped.weight = 2
    for t in range(1000):
        #frame = env.unwrapped.MyoEnv.sim.renderer.render_offscreen(camera_id=0)
        #frames.append(frame)
        action, _state = model.predict(obs, deterministic=True)
        obs,reward,done,truncated,info = env.step(action)
        a = obs[0][4:10]
        #env.render("human")
        data_f.write(str(nEval)+","+str(t)+","+str(obs[0][0])+","+
                    str(obs[0][3])+ ","+str(np.dot(a,a))+'\n')
   
        # VecEnv resets automatically
        #if done:
            #count=count+1
            #break   
data_f.close()
#frames2Video('CodeColab/UE/ElbowFixTarget/vid_plots/' + 'Test.mp4', frames)
#plotFlexAngleHealthy()
#frames2Video('CodeColab/UE/ElbowFixTarget/vid_plots/healthyVideoResults.mp4', frames)

#chk_freq = sys.argv[1] # use this to identify the run parameters
#timestep  = sys.argv[2]] 


