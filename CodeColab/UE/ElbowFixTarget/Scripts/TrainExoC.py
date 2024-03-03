import gymnasium
import gym
import argparse

from stable_baselines3 import PPO

from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import CheckpointCallback

parser = argparse.ArgumentParser()
parser.add_argument("--timestep", type=int, help="(int) Number of training timesteps")
parser.add_argument("--chk_freq", type=int, help="(int) Checkpoint Frequency (in timesteps)")
input_args = parser.parse_args()

register(id="myoElbow-v0", entry_point="CodeColab.UE.ElbowFixTarget.Scripts.ExoInterface:ElbowFlex", max_episode_steps=1000)
env=gymnasium.make("myoElbow-v0")

model = PPO("MlpPolicy", env, verbose=1)
checkpoint_callback = CheckpointCallback(save_freq= input_args.chk_freq, save_path='SB3_logs_Exo', name_prefix=str('MyoAssist'), save_replay_buffer=False, save_vecnormalize=False)
#checkpoint_callback = CheckpointCallback(save_freq=chk_freq, save_path=os.path.join(input_args.save_path, 'logs'), name_prefix=str(input_args.rl_algo), save_replay_buffer=False, save_vecnormalize=False,)
model.learn(total_timesteps=input_args.timestep, callback=checkpoint_callback, tb_log_name="PPO_exo_train", reset_num_timesteps=False)
model.save("MyoAssistAgent2")

#vec_env = model.get_env()
#obs = vec_env.reset()
#for i in range(1000):
#    action, _state = model.predict(obs, deterministic=True)
#    obs, reward, done, info = vec_env.step(action)
#    vec_env.render("human")
#    # VecEnv resets automatically
#    if done:
#        obs = vec_env.reset()


#chk_freq = sys.argv[1] # use this to identify the run parameters
#timestep  = sys.argv[2]] 


