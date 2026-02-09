import os
import time
import argparse
import numpy as np
import torch
import csv
import random

from environment import SecCom_Env
from ppo import PPOAgent, Memory
from system_configuration import Config
from attacks import Attacker

cfg = Config()

dir_logs = "logs/"

def flatten_obs(obs):
    field_vec = obs["user_field_response_vector"].flatten()
    data_rate = obs["user_data_rate"].flatten()
    return np.concatenate([field_vec, data_rate])

def run_adversarial_training(args, model_custom_parameters):
    env = SecCom_Env(cfg, is_adversarial=False, fairness_mode="sum")
    obs_shape = env.observation_shape["user_field_response_vector"]
    obs_dim = (obs_shape[0] * obs_shape[1]) + env.observation_shape["user_data_rate"][0]
    action_dim = env.action_dim
    
    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)
    
    log_file_path = os.path.join(dir_logs, f"PPO_ADV_TRAIN_{int(time.time())}{model_custom_parameters}.csv")
    csv_file = open(log_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    header = ['step', 'beam_power', 'sum_data_rate', 'model_reward', 'is_adv', 'epsilon']
    for i in range(cfg.M_users):
        header.append(f'data_rate_user_{i}')
    csv_writer.writerow(header)
    agent = PPOAgent(obs_dim, action_dim, lr=0.0007)
    memory = Memory()
    attack_config_template = {
        "method": "AdversarialPolicy",
        "steps": 50,
        "alpha": 0.1 
    }
    attacker = Attacker(agent, attack_config_template)
    
    EPSILON_LIST = [0.1, 0.2, 0.5, 1.0, 2.0]
    
    state, info = env.reset()
    state_flat = flatten_obs(state)
    
    total_steps = 0
    time_steps = int(args.total_timestep) if args.total_timestep else 500000
    update_timestep = 1000
    
    print(f"Starting Adversarial Training for {time_steps} steps...")
    print(f"Attack frequency: 50% | Epsilons: {EPSILON_LIST}")

    while total_steps < time_steps:
        is_adv = random.random() < 0.5
        current_epsilon = 0.0
        
        if is_adv:
    
            current_epsilon = random.choice(EPSILON_LIST)
            attacker.epsilon = current_epsilon
            attacker.alpha = current_epsilon / 5.0
        
            state_to_process = attacker.perturb(state_flat)
        else:
            state_to_process = state_flat
        action, action_logprob, value = agent.select_action(state_to_process)
        next_state, reward, terminated, truncated, info = env.step(action)
            
        next_state_flat = flatten_obs(next_state)
        done = terminated or truncated
        memory.states.append(state_to_process)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        memory.values.append(value)
        
        state_flat = next_state_flat
        total_steps += 1
        
        # Logging
        if total_steps % 10 == 0:
            user_data_rates = info["user_data_rate"]
            sum_data_rate = user_data_rates.sum()
            
            row = [
                total_steps, 
                info["beam_power"], 
                sum_data_rate, 
                reward,
                int(is_adv),
                current_epsilon
            ]
            row.extend(user_data_rates.tolist())
            csv_writer.writerow(row)
            csv_file.flush()

        if done:
            state, info = env.reset()
            state_flat = flatten_obs(state)
            
        # Update PPO agent
        if total_steps % update_timestep == 0:
            agent.update(memory)
            memory.clear()
            print(f"Step: {total_steps}/{time_steps} - Policy Updated (Robustness Training)")
            
        # Save model every 50000 steps
        if total_steps % 50000 == 0:
            model_dir = "models/"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = os.path.join(model_dir, f"PPO_ROBUST_{int(time.time())}_{total_steps}.pth")
            agent.save(save_path)

    csv_file.close()
    
    model_dir = "models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    final_save_path = os.path.join(model_dir, f"PPO_ROBUST_final_{int(time.time())}.pth")
    agent.save(final_save_path)
    print(f"Adversarial Training finished. Log saved to {log_file_path}")
    print(f"Final ROBUST model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--total_timestep", default="350000")
    parser.add_argument('-Lp', "--L_path_propagation")
    args = parser.parse_args()

    model_custom_parameters = f"_Robust_T{args.total_timestep}"
    
    if args.L_path_propagation:
        cfg.l_path_propagation = int(args.L_path_propagation)
        model_custom_parameters += f"_L{cfg.l_path_propagation}"
        
    cfg.init_unstatic_value()
    run_adversarial_training(args, model_custom_parameters)
