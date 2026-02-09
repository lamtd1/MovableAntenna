
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from environment import SecCom_Env
from ppo import PPOAgent
from system_configuration import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ATTACK_CONFIGS = [
    {"name": "Baseline (No Attack)", "method": "None", "epsilon": 0.0, "steps": 0},
    {"name": "FGSM (eps=0.5)",       "method": "FGSM", "epsilon": 0.5,  "steps": 1},
    {"name": "FGSM (eps=1.0)",       "method": "FGSM", "epsilon": 1.0,  "steps": 1},
    {"name": "PGD (eps=0.5)",        "method": "PGD", "epsilon": 0.5, "alpha": 0.1, "steps": 50},
    {"name": "PGD (eps=1.0)",        "method": "PGD", "epsilon": 1.0, "alpha": 0.2, "steps": 50},
    {"name": "AdvPolicy (eps=0.1)", "method": "AdversarialPolicy", "epsilon": 0.1, "alpha": 0.02, "steps": 50},
    {"name": "AdvPolicy (eps=0.2)", "method": "AdversarialPolicy", "epsilon": 0.2, "alpha": 0.04, "steps": 50},
    {"name": "AdvPolicy (eps=0.5)", "method": "AdversarialPolicy", "epsilon": 0.5, "alpha": 0.1,  "steps": 50},
    {"name": "AdvPolicy (eps=1.0)", "method": "AdversarialPolicy", "epsilon": 1.0, "alpha": 0.2,  "steps": 50},
    {"name": "AdvPolicy (eps=2.0)", "method": "AdversarialPolicy", "epsilon": 2.0, "alpha": 0.5,  "steps": 50},
]


# Common Settings
VICTIM_MODEL_PATH = "/Users/taduylam/Workspace/lab/MovableAntenna/models/PPO_ROBUST_final_1770467884.pth"
# VICTIM_MODEL_PATH = "/Users/taduylam/Workspace/lab/MovableAntenna/models/PPO_final_1770456612.pth"
TEST_EPISODES = 1000
SMOOTH_WINDOW = 500
MODE = 'Targeted'

def flatten_obs(obs):
    field_vec = obs["user_field_response_vector"].flatten()
    data_rate = obs["user_data_rate"].flatten()
    return np.concatenate([field_vec, data_rate])

class Attacker:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.epsilon = config.get("epsilon", 0.0)
        self.method = config.get("method", "None")
        self.steps = config.get("steps", 0)
        
        if "alpha" in config:
            self.alpha = config["alpha"]
        else:
            self.alpha = self.epsilon / 5.0 if self.method in ['PGD', 'MIM', 'AdversarialPolicy'] else 1.0 
            
        self.mode = MODE
        self.decay = 1.0 # Momentum decay factor for MIM
        
        # Load Evil Policy if needed
        self.evil_actor = None
        if self.method == "AdversarialPolicy":
            try:
                # Load PPO_EVIL model (Assume path exists)
                evil_path = '/Users/taduylam/Workspace/lab/MovableAntenna/models/PPO_ROBUST_final_1770486306.pth'
                checkpoint = torch.load(evil_path, map_location=torch.device('cpu'))
                from ppo import PPOAgent
                self.evil_input_dim = 315 
                self.evil_agent_wrapper = PPOAgent(
                    obs_dim=self.evil_input_dim, 
                    action_dim=20, 
                    lr=0.0003, gamma=0.99, K_epochs=40, eps_clip=0.2
                )
                
                if isinstance(checkpoint, dict) and 'actor.0.weight' in checkpoint:
                     self.evil_agent_wrapper.policy.load_state_dict(checkpoint, strict=False)
                elif isinstance(checkpoint, dict) and 'policy_old_state_dict' in checkpoint:
                     self.evil_agent_wrapper.policy.load_state_dict(checkpoint['policy_old_state_dict'], strict=False)
                else:
                     self.evil_agent_wrapper.policy.load_state_dict(checkpoint, strict=False)

                self.evil_actor = self.evil_agent_wrapper.policy.actor
                self.evil_actor.eval()
                print(f"Loaded Adversarial Policy from {evil_path}")
            except Exception as e:
                print(f"Failed to load Adversarial Policy: {e}")
                self.evil_actor = None
                
    def _get_evil_target(self, state_tensor):
        if self.evil_actor is None:
            return torch.zeros((1, 20)).to(device)
            
        with torch.no_grad():
            action = self.evil_actor(state_tensor)
        return action.detach()

    def perturb(self, state):
        if self.method == "None":
            return state
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).clone()
        
        # Determine Target Action
        if self.method == "AdversarialPolicy":
             target_action = self._get_evil_target(state_tensor)
        else:
            with torch.no_grad():
                dist_v, _ = self.agent.policy(state_tensor)
                best_action = dist_v.sample()
                
                if self.mode == 'MinBest':
                    target_action = best_action
                else:
                    target_action = -best_action # Targeted: Opposite of best

        # Attack Logic Dispatch
        if self.method == 'FGSM':
            return self._fgsm(state_tensor, target_action)
        elif self.method in ['PGD', 'AdversarialPolicy']:
            return self._pgd(state_tensor, target_action)
        elif self.method == 'MIM':
            return self._mim(state_tensor, target_action)
        elif self.method == 'CW':
            return self._cw(state_tensor, target_action)
            
        return state

    def _fgsm(self, state_tensor, target_action):
        state_tensor.requires_grad = True
        dist, _ = self.agent.policy(state_tensor)
        
        if self.mode == 'MinBest':
            loss = dist.log_prob(target_action).sum()
        else:
            loss = -dist.log_prob(target_action).sum()
            
        self.agent.policy.zero_grad()
        loss.backward()
        
        pert = -self.epsilon * state_tensor.grad.data.sign()
        adv_state = state_tensor + pert
        return adv_state.squeeze(0).detach().numpy()

    def _pgd(self, state_tensor, target_action):
        original_state = state_tensor.clone().detach()
        adv_state = state_tensor.clone().detach()
        
        for _ in range(self.steps):
            adv_state.requires_grad = True
            dist, _ = self.agent.policy(adv_state)
            
            if self.mode == 'MinBest':
                loss = dist.log_prob(target_action).sum()
            else:
                loss = -dist.log_prob(target_action).sum()
                
            self.agent.policy.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                pert = -self.alpha * adv_state.grad.sign()
                adv_state += pert
                
             
                diff = torch.clamp(adv_state - original_state, -self.epsilon, self.epsilon)
                adv_state = original_state + diff
                
        return adv_state.squeeze(0).detach().numpy()

    def _mim(self, state_tensor, target_action):
        original_state = state_tensor.clone().detach()
        adv_state = state_tensor.clone().detach()
        momentum = torch.zeros_like(state_tensor)
        
        for _ in range(self.steps):
            adv_state.requires_grad = True
            dist, _ = self.agent.policy(adv_state)
            
            if self.mode == 'MinBest':
                loss = dist.log_prob(target_action).sum()
            else:
                loss = -dist.log_prob(target_action).sum()
                
            self.agent.policy.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                grad = adv_state.grad
            
                grad_norm = torch.mean(torch.abs(grad))
                grad = grad / (grad_norm + 1e-10)
                
                momentum = self.decay * momentum + grad
                
                pert = -self.alpha * momentum.sign()
                adv_state += pert
                
            
                diff = torch.clamp(adv_state - original_state, -self.epsilon, self.epsilon)
                adv_state = original_state + diff
                
        return adv_state.squeeze(0).detach().numpy()

    def _cw(self, state_tensor, target_action):
        """
        Carlini-Wagner (L2) style attack adapted for continuous RL action space.
        Optimizes perturbation `w` to change the action distribution.
        Object: Minimize |Likelihood(CurrentAction)| + c * |pert|_2^2
        """
        original_state = state_tensor.clone().detach()
        
        delta = torch.zeros_like(state_tensor, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.01)
        
        kappa = self.config.get("confidence", 0) 
        c = 1.0 # Balancing constant
        
        for _ in range(self.steps):
            perturbed_state = original_state + delta
            dist, _ = self.agent.policy(perturbed_state)
            log_prob = dist.log_prob(target_action).sum()
            
            if self.mode == 'MinBest':
                loss_attack = log_prob 
            else:
                loss_attack = -log_prob
            
            loss_dist = torch.sum(delta ** 2)
            
            total_loss = loss_attack + c * loss_dist
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                delta.clamp_(-self.epsilon, self.epsilon)
                
        adv_state = original_state + delta
        return adv_state.squeeze(0).detach().numpy()


def evaluate_config(env, agent, config_dict):
    print(f"Running Config: {config_dict['name']}...")
    attacker = Attacker(agent, config_dict)
    
    rates = []
    
    start_time = time.time()
    for episode in range(TEST_EPISODES):
        state, info = env.reset()
        state_flat = flatten_obs(state)
        
        adv_state = attacker.perturb(state_flat)
        
        action, _, _ = agent.select_action(adv_state)
        
        _, _, terminated, truncated, info = env.step(action)
        
        rates.append(np.sum(info['user_data_rate']))
        
        if (episode+1) % 200 == 0:
            print(f"  Ep {episode+1}/{TEST_EPISODES}...")
            
    print(f"  Computed in {time.time() - start_time:.2f}s")
    return rates
if __name__ == "__main__":
    # Setup
    cfg = Config()
    cfg.init_unstatic_value()
    env = SecCom_Env(cfg)
    
    obs_shape = env.observation_shape["user_field_response_vector"]
    obs_dim = (obs_shape[0] * obs_shape[1]) + env.observation_shape["user_data_rate"][0]
    action_dim = env.action_dim
    
    victim_agent = PPOAgent(obs_dim, action_dim)
    if os.path.exists(VICTIM_MODEL_PATH):
        victim_agent.load(VICTIM_MODEL_PATH)
        print(f"Loaded Victim Model: {VICTIM_MODEL_PATH}")
    else:
        print("Error: Model not found.")
        exit(1)
        
    # Run All Configs
    all_results = {}
    
    # 1. Run Simulations
    for config in ATTACK_CONFIGS:
        rates = evaluate_config(env, victim_agent, config)
        all_results[config['name']] = rates
        
    # 2. Get Baseline Stats
    baseline_rates = all_results["Baseline (No Attack)"]
    baseline_mean = np.mean(baseline_rates)
    
    print("\n--- SUMMARY REPORT ---")
    print(f"{'Configuration':<30} {'Mean Rate':<10} {'Drop %':<10}")
    print("-" * 55)
    
    plot_data = [] # List of tuples (Step, Rate, Config)
    
    for config in ATTACK_CONFIGS:
        name = config['name']
        rates = all_results[name]
        mean_rate = np.mean(rates)
        
        drop_pct = 0.0
        if name != "Baseline (No Attack)":
            drop_pct = ((baseline_mean - mean_rate) / baseline_mean) * 100
            
        print(f"{name:<30} {mean_rate:<10.4f} {drop_pct:<10.2f}%")
        

        series = pd.Series(rates)
        smoothed = series.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
        
        for step, val in enumerate(smoothed):

            label_str = f"{name}"
            if name != "Baseline (No Attack)":
                label_str += f" (Drop: {drop_pct:.1f}%)"
            else:
                label_str += f" (Avg: {mean_rate:.2f})"
                
            plot_data.append({
                "Episode": step,
                "Sum Data Rate (bps/Hz)": val,
                "Configuration": label_str
            })

  
    unique_methods = []
    for c in ATTACK_CONFIGS:
        if c['method'] != "None" and c['method'] not in unique_methods:
            unique_methods.append(c['method'])

    baseline_config = next(c for c in ATTACK_CONFIGS if c['method'] == "None")
    baseline_name = baseline_config['name']
    
    for method in unique_methods:
        method_plot_data = []
    
        b_series = pd.Series(baseline_rates)
        b_smoothed = b_series.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
        for step, val in enumerate(b_smoothed):
            if step < 100: continue
            method_plot_data.append({
                "Episode": step,
                "Sum Data Rate (bps/Hz)": val,
                "Configuration": f"Baseline (Avg: {baseline_mean:.2f})"
            })
            
        method_configs = [c for c in ATTACK_CONFIGS if c['method'] == method]
        for config in method_configs:
            name = config['name']
            rates = all_results[name]
            mean_rate = np.mean(rates)
            drop_pct = ((baseline_mean - mean_rate) / baseline_mean) * 100
            
            series = pd.Series(rates)
            smoothed = series.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
            
            for step, val in enumerate(smoothed):
                if step < 100: continue
                method_plot_data.append({
                    "Episode": step,
                    "Sum Data Rate (bps/Hz)": val,
                    "Configuration": f"{name} (Drop: {drop_pct:.1f}%)"
                })

        # 3.3. Build and save the plot
        df_method = pd.DataFrame(method_plot_data)
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        sns.lineplot(
            data=df_method,
            x="Episode",
            y="Sum Data Rate (bps/Hz)",
            hue="Configuration",
            palette="tab10",
            linewidth=2.0
        )
        plt.axhline(baseline_mean, color='black', linestyle='--', alpha=0.3)
        plt.title(f"Attack Performance: {method}\nVictim: {os.path.basename(VICTIM_MODEL_PATH)}")
        plt.xlabel("Episode (Smoothed)")
        plt.ylabel("Sum Data Rate (bps/Hz)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = f"plot_{method.lower()}_{int(time.time())}.png"
        plt.savefig(save_path)
        plt.close() 
        print(f"  Generated plot for {method}: {save_path}")

    print("\nAll individual plots have been generated.")
