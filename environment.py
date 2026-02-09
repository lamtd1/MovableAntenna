from typing import Optional

import numpy as np
import gymnasium as gym

from basestation import BS


class SecCom_Env:
    reset_counter = 0
    reward = 0

    secret_rate = 0
    beam_power = 0
    sum_data_rate = 0
    eve_sum_data_rate = 0

    def __init__(self, configuration, is_adversarial=False, fairness_mode="sum"):
        self.cfg = configuration
        self.is_adversarial = is_adversarial
        self.fairness_mode = fairness_mode # 'sum', 'min_max', 'jain'
        self.antenna_array = self.cfg.default_antenna_array
        self.beamforming_matrices = self.cfg.default_beam_matrices
        self.basestation = BS(self.antenna_array, self.cfg)
        self.sum_data_rate = 0
        self.step_counter = 0
        self.observation_shape = {
            "user_field_response_vector": (self.cfg.M_users, (self.cfg.l_path_propagation * self.cfg.N_antennaElement) * 2),
            "user_data_rate": (self.cfg.M_users,),
        }
        
        self.action_dim = self.cfg.N_antennaElement + (self.cfg.Number_of_com_components * self.cfg.N_antennaElement)

        for user in self.basestation.usrList:
            user.update_attribute(self.antenna_array)

    # Observation space update
    def _get_obs(self):
        all_field_response_vector = []
        user_data_rates = []
        for i in range(self.cfg.M_users):
            all_field_response_vector.append(self.basestation.usrList[i].get_field_response_vector().reshape(
                self.cfg.l_path_propagation * self.cfg.N_antennaElement, ))
            user_data_rates.append(self.basestation.usrList[i].get_data_rate())
        
        # Convert complex vector to real-valued vector (real, imag)
        complex_vector = np.array(all_field_response_vector, dtype=np.complex128)
        all_field_response_vector = np.concatenate([np.real(complex_vector), np.imag(complex_vector)], axis=-1)
        return {
            "user_field_response_vector": all_field_response_vector,
            "user_data_rate": np.array(user_data_rates),
        }

    # Get information for monitoring
    def _get_info(self):
        user_data_rates = np.array([u.get_data_rate() for u in self.basestation.usrList])
        return {
            "antenna_array": self.antenna_array,
            "beamforming_matrices": self.beamforming_matrices,
            "user_data_rate": user_data_rates,
            "user_security_rate": self.basestation.get_secretary_rate(), # Kept for record but not used in reward
            "sensing_sirn": self.basestation.sensing_target.get_SINR(),
            "model_reward": self.reward,
            "beam_power": self.beam_power
        }

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.step_counter = 0
        self.reward = 0
        self.antenna_array = self.cfg.default_antenna_array
        self.beamforming_matrices = self.cfg.default_beam_matrices
        self.basestation = BS(self.antenna_array, self.cfg)
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Processing actions
        # Extract antenna element position
        self.antenna_array = (action[:self.cfg.N_antennaElement] + 1)*self.cfg.MA_size*self.cfg.dlambda / 2
        # Extract beamforming matrices
        self.beamforming_matrices = (action[self.cfg.N_antennaElement:] * self.cfg.beamforming_action_scaling).reshape(
            self.cfg.Number_of_com_components, self.cfg.N_antennaElement)

        self.basestation.update_system(self.antenna_array, self.beamforming_matrices)
        self.sum_data_rate = sum([u.get_data_rate() for u in self.basestation.usrList])

        # Reward calculation
        if self.is_adversarial:
            sensing_sinr = self.basestation.sensing_target.get_SINR()
            if self.validating_actions(sensing_sinr):
                self.reward = 1.0 / (self.sum_data_rate + 0.1)
            else:
                self.reward = -10.0
                
        else:
            # NORMAL MODE
            sensing_sinr = self.basestation.sensing_target.get_SINR()
            if self.validating_actions(sensing_sinr):
                if self.fairness_mode == "sum":
                    self.reward = self.sum_data_rate
                elif self.fairness_mode == "min_max":
                    # Max-Min Fairness: Reward based on minimum user rate
                    min_rate = min([u.get_data_rate() for u in self.basestation.usrList])
                    self.reward = min_rate * self.cfg.M_users # Scale to be comparable
                elif self.fairness_mode == "jain":
                    # Jain's Fairness Index * Sum Rate
                    rates = np.array([u.get_data_rate() for u in self.basestation.usrList])
                    sum_rates = rates.sum()
                    sum_sq_rates = (rates ** 2).sum()
                    if sum_sq_rates == 0:
                        jain_index = 0
                    else:
                        jain_index = (sum_rates ** 2) / (self.cfg.M_users * sum_sq_rates)
                    self.reward = jain_index * sum_rates
                else:
                    self.reward = self.sum_data_rate
            else:
                self.reward = 0
                
        reward = self.reward

        # Terminate after 1 step ~ 1 step/episode
        terminated = True
        truncated = False
        self.step_counter += 1

        # These parameters only for logging purpose
        self.beam_power = np.trace(np.dot(self.beamforming_matrices, np.conj(self.beamforming_matrices).T))
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    """
    Validating action function is:
    - For checking system model constrain
    - All print function is for live training superviser
    """
    def validating_actions(self, sensing_sinr):
        # Validate antenna element positions
        for i in range(len(self.antenna_array) - 1):
            if (self.antenna_array[i + 1] - self.antenna_array[i]) < self.cfg.D0_spacingAntenna:
                print("check position")
                return False
            
        # Validate beamforming matries
        if (np.trace(np.dot(self.beamforming_matrices,
                            np.conj(self.beamforming_matrices).T)) >= self.cfg.P0_basestation_power):
            print("check beam")
            return False
        
        # Validate sensing SINR 
        if not sensing_sinr >= self.cfg.sensing_SINR_min:
            print("check SINR")
            return False
        return True

