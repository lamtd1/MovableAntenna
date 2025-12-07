import gymnasium as gym

from stable_baselines3 import PPO, A2C, DDPG

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import os
import time
import argparse
import numpy as np

from environment import SecCom_Env
from system_configuration import Config

cfg = Config()

# set up monitor log dir
dir_monitor_log = "monitor_log/"
dir_tensorboard_logs = f"logs/"
file_name = "no_custom_name"

time_steps = 350000  # default time steps for all models

class TensorboardCallBack(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallBack, self).__init__(verbose)
        self.beam_value = 0
        self.secret_rate = 0
        self.users_data_rate = 0

    # def _on_training_start(self) -> None:

    # def _on_rollout_start(self) -> None:

    def _on_step(self) -> bool:
        self.beam_value = self.training_env.get_attr("beam_power")[0]
        self.secret_rate = self.training_env.get_attr("secret_rate")[0]
        self.users_data_rate = self.training_env.get_attr("sum_data_rate")[0]
        self.eva_sum_data_rate = self.training_env.get_attr("eve_sum_data_rate")[0]
        self.logger.record("custom/beam_power", self.beam_value)
        self.logger.record("custom/secret_rate", self.secret_rate)
        self.logger.record("custom/sum_data_rate", self.users_data_rate)
        self.logger.record("custom/eva_sum_data_rate", self.eva_sum_data_rate)
        return True

    # def _on_rollout_end(self) -> None:
    #     return super()._on_rollout_end()

    # def _on_training_end(self) -> None:

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model')
parser.add_argument('-Mu', '--num_users')
parser.add_argument('-N', '--antenna')
parser.add_argument('-Ms', '--min_sensing')
parser.add_argument('-po', '--max_txPo')
parser.add_argument('-f', "--filename")
parser.add_argument('-t', "--total_timestep")

# custom model parametes
parser.add_argument('-bs', "--beam_upscale")
parser.add_argument('-Lp', "--L_path_propagation")
parser.add_argument('-As', "--atenna_size")
args = parser.parse_args()
model_custom_parameters = f"_{args.model}"


def run_model(model_name):
    envPPO = SecCom_Env(cfg)
    env = Monitor(envPPO, info_keywords=("model_reward","user_security_rate"), filename=dir_monitor_log+model_name+f"_{int(time.time())}"+file_name+model_custom_parameters)
    log_dir = dir_tensorboard_logs + model_name + f"_{int(time.time())}" + model_custom_parameters
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Wrap with DummyVecEnv and VecNormalize
    env = DummyVecEnv([lambda: env])
    # Thêm VecNormalize để normalize observation và reward ? Not sure work
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    env.reset()
    model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=5)
    match args.model:
        case 'PPO':
            model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, clip_range=0.4,
                        learning_rate=0.0007,
                        gae_lambda=1, n_steps=1000)

        # This case had been defined as default case
        # case 'A2C':
        #     model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=5)

        case 'DDPG':
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            model = DDPG("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, action_noise=action_noise,
                         learning_starts=8500, batch_size=256)

    Total_TIMESTEPS = time_steps
    custom_tensor_callback = TensorboardCallBack()
    model.learn(total_timesteps=Total_TIMESTEPS, reset_num_timesteps=True, tb_log_name="PPO",
                callback=custom_tensor_callback)
    del model
    env.close()
    return 0


if args.antenna is not None:
    cfg.N_antennaElement = int(args.antenna)
    print("changed antenna element to: ", cfg.N_antennaElement)
    model_custom_parameters = model_custom_parameters+f"_N_atn_{cfg.N_antennaElement}"

if args.num_users is not None:
    cfg.M_users = int(args.num_users)
    print("changed number of users to: ", cfg.M_users)
    model_custom_parameters = model_custom_parameters+f"_M_usrs_{cfg.M_users}"

if args.min_sensing is not None:
    cfg.sensing_SINR_min = float(args.min_sensing)
    print("changed min sensing SINR to: ", cfg.sensing_SINR_min)
    model_custom_parameters = model_custom_parameters+f"_senSINR_min_{cfg.sensing_SINR_min}"

if args.max_txPo is not None:
    cfg.P0_basestation_power = float(args.max_txPo)
    print("changed max basestation power to: ", cfg.P0_basestation_power)
    model_custom_parameters = model_custom_parameters+f"_bs_power_{cfg.P0_basestation_power}"

if args.filename is not None:
    file_name = args.filename
print("saving directory is: ", dir_monitor_log+file_name)

if args.total_timestep is not None:
    time_steps = int(args.total_timestep)
    print("total time step running is: ", time_steps)

if args.beam_upscale is not None:
    cfg.beamforming_action_scaling = float(args.beam_upscale)
    print("Beam upscaler changes to: ", cfg.beamforming_action_scaling)
    model_custom_parameters = model_custom_parameters+f"_beam_upscaler_{cfg.beamforming_action_scaling}"

if args.L_path_propagation is not None:
    cfg.l_path_propagation = int(args.L_path_propagation)
    print("Number of Path propagation changes to: ", cfg.l_path_propagation)
    model_custom_parameters = model_custom_parameters+f"_L_path_propagation_{cfg.l_path_propagation}"

if args.atenna_size is not None:
    cfg.MA_size = int(args.atenna_size)
    print("Size of Movable antenna: ", cfg.MA_size)
    model_custom_parameters = model_custom_parameters+f"_movable_antenna_size_{cfg.MA_size}"

model_custom_parameters = model_custom_parameters+f"_timeStep_{time_steps}"
cfg.init_unstatic_value()

if args.model is None:
    print("No model name found!")
else:
    run_model(args.model)


