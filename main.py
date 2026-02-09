import os
import time
import argparse
import numpy as np
import torch
import csv

from environment import SecCom_Env
from ppo import PPOAgent, Memory
from system_configuration import Config

cfg = Config()

# set up monitor log dir
dir_logs = "logs/"

time_steps = 350000  # default time steps for all models

def flatten_obs(obs):
    # Flatten both field response vector and data rate
    field_vec = obs["user_field_response_vector"].flatten()
    data_rate = obs["user_data_rate"].flatten()
    return np.concatenate([field_vec, data_rate])

def run_manual_ppo(args, model_custom_parameters):
    # Pass configuration flags to Environment
    env = SecCom_Env(cfg, is_adversarial=args.invert_reward, fairness_mode=args.fairness)
    
    # Calculate flattened observation dimension
    obs_shape = env.observation_shape["user_field_response_vector"]
    obs_dim = (obs_shape[0] * obs_shape[1]) + env.observation_shape["user_data_rate"][0]
    action_dim = env.action_dim
    
    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)
    
    log_file_path = os.path.join(dir_logs, f"PPO_manual_{int(time.time())}{model_custom_parameters}.csv")
    csv_file = open(log_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Prepare header (Focused on maximizing sum_data_rate)
    header = ['step', 'beam_power', 'sum_data_rate', 'model_reward']
    for i in range(cfg.M_users):
        header.append(f'data_rate_user_{i}')
    csv_writer.writerow(header)
    
    agent = PPOAgent(obs_dim, action_dim, lr=0.0007)
    memory = Memory()
    
    state, info = env.reset()
    state = flatten_obs(state)
    
    total_steps = 0
    update_timestep = 1000 # Update policy every 1000 steps
    
    while total_steps < time_steps:
        action, action_logprob, value = agent.select_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
            
        next_state = flatten_obs(next_state)
        done = terminated or truncated
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        memory.values.append(value)
        
        state = next_state
        total_steps += 1
        
        # Logging
        if total_steps % 10 == 0:
            user_data_rates = info["user_data_rate"]
            sum_data_rate = user_data_rates.sum()
            
            row = [
                total_steps, 
                info["beam_power"], 
                sum_data_rate, 
                reward
            ]
            row.extend(user_data_rates.tolist())
            csv_writer.writerow(row)
            csv_file.flush()

        if done:
            state, info = env.reset()
            state = flatten_obs(state)
            
        # Update PPO agent
        if total_steps % update_timestep == 0:
            agent.update(memory)
            memory.clear()
            print(f"Step: {total_steps}/{time_steps} - Policy Updated")
            
        # Save model every 50000 steps
        if total_steps % 50000 == 0:
            model_dir = "models/"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = os.path.join(model_dir, f"PPO_{int(time.time())}_{total_steps}.pth")
            agent.save(save_path)

    csv_file.close()
    
    # Save final model
    model_dir = "models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    final_save_path = os.path.join(model_dir, f"PPO_final_{int(time.time())}.pth")
    if args.invert_reward:
        final_save_path = os.path.join(model_dir, f"PPO_EVIL_final_{int(time.time())}.pth")
        
    agent.save(final_save_path)
    print(f"Training finished. Log saved to {log_file_path}")
    print(f"Final model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='PPO')
    parser.add_argument('-inv', '--invert_reward', action='store_true', help='Invert reward to train an Evil Agent')
    parser.add_argument('-Mu', '--num_users')
    parser.add_argument('-N', '--antenna')
    parser.add_argument('-Ms', '--min_sensing')
    parser.add_argument('-po', '--max_txPo')
    parser.add_argument('-f', "--filename")
    parser.add_argument('-t', "--total_timestep")
    parser.add_argument('-fair', "--fairness", default="sum", help="Fairness mode: sum (default), min_max, jain")

    parser.add_argument('-bs', "--beam_upscale")
    parser.add_argument('-Lp', "--L_path_propagation")
    parser.add_argument('-As', "--atenna_size")
    args = parser.parse_args()

    model_custom_parameters = f"_{args.model}"

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

    if args.invert_reward:
        print("WARNING: ADVERSARIAL MODE. REWARDS INVERTED.")
        model_custom_parameters += "_INVERTED"

    cfg.init_unstatic_value()

    if args.model == "PPO":
        run_manual_ppo(args, model_custom_parameters)
    else:
        print(f"Model {args.model} not supported in manual mode yet. Only PPO is implemented.")


