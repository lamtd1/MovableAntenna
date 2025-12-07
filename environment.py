from typing import Optional

import numpy as np
import gymnasium as gym

from basestation import BS


class SecCom_Env(gym.Env):
    reset_counter = 0
    reward = 0

    secret_rate = 0
    beam_power = 0
    sum_data_rate = 0
    eve_sum_data_rate = 0

    def __init__(self, configuration):
        self.cfg = configuration
        # create new BaseStation -> basestation init
        self.antenna_array = self.cfg.default_antenna_array
        self.beamforming_matrices = self.cfg.default_beam_matrices
        self.basestation = BS(self.antenna_array, self.cfg)
        self.sum_data_rate = 0

        # Observation space
        self.observation_space = gym.spaces.Dict(
            {
                "user_field_response_vector": gym.spaces.Box(-50, 50, shape=(
                self.cfg.M_users, (self.cfg.l_path_propagation * self.cfg.N_antennaElement) * 2), dtype=np.float64),
                "user_security_rate": gym.spaces.Box(0, 20, shape=(self.cfg.M_users,), dtype=np.float64),
            }
        )

        for user in self.basestation.usrList:
            user.update_attribute(self.antenna_array)

        # Action space
        """
        Action space including:
            N antenna element positions
            beamforming matrix parameters = Number of com component * Number of antenna elements
        """
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(
        (self.cfg.N_antennaElement + (self.cfg.Number_of_com_components * self.cfg.N_antennaElement)),),
                                           dtype=np.float32)

    # Observation space update
    def _get_obs(self):
        all_field_response_vector = []
        for i in range(self.cfg.M_users):
            all_field_response_vector.append(self.basestation.usrList[i].get_field_response_vector().reshape(
                self.cfg.l_path_propagation * self.cfg.N_antennaElement, ))
        
        # Convert complex vector to real-valued vector (real, imag)
        complex_vector = np.array(all_field_response_vector, dtype=np.complex128)
        all_field_response_vector = np.concatenate([np.real(complex_vector), np.imag(complex_vector)], axis=-1)
        return ({
            "user_field_response_vector": all_field_response_vector,
            "user_security_rate": self.basestation.get_secretary_rate(),
        })

    # Get information for Monitor environment wrapper
    def _get_info(self):
        return {
            "antenna_array": self.antenna_array,
            "beamforming_matrices": self.beamforming_matrices,
            "user_security_rate": self.basestation.get_secretary_rate(),
            "sensing_sirn": self.basestation.sensing_target.get_SINR(),
            "model_reward": self.reward,
            "beam_power": 1
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.step_counter = 0
        self.reward = 0
        # self.reset_counter += 1
        self.antenna_array = self.cfg.default_antenna_array
        self.beamforming_matrices = self.cfg.default_beam_matrices
        self.basestation = BS(self.antenna_array, self.cfg)
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        
        # (Gauss-Markov)
        for user in self.basestation.usrList:
            # Lưu lại state t-1
            prev_speed = user.speed
            prev_direction = user.direction

            # Nhiễu cho 3.22
            s_r = np.random.normal(0, 1)
            d_r = np.random.normal(0, 1)

            alpha = self.cfg.gm_alpha
            
            # Update Speed
            # sn,t = αn sn,t−1 + (1 − αn) ¯sn + sqrt(1 − αn^2) sr,n,t−1
            user.speed = (alpha * prev_speed) + \
                         ((1 - alpha) * self.cfg.gm_mean_speed) + \
                         (np.sqrt(1 - alpha**2) * s_r)
            
            # Update Direction
            # dn,t = αn dn,t−1 + (1 − αn) ¯dn + sqrt(1 − αn^2) dr,n,t−1
            user.direction = (alpha * prev_direction) + \
                             ((1 - alpha) * user.mean_direction) + \
                             (np.sqrt(1 - alpha**2) * d_r)
            
            new_x = user.x + prev_speed * np.cos(prev_direction)
            new_y = user.y + prev_speed * np.sin(prev_direction)

            new_x = np.clip(new_x, self.cfg.x_min, self.cfg.x_max)
            new_y = np.clip(new_y, self.cfg.y_min, self.cfg.y_max)

            user.update_location(new_x, new_y)

        # Processing actions
        # Extract antenna element position
        self.antenna_array = (action[:self.cfg.N_antennaElement] + 1)*self.cfg.MA_size*self.cfg.dlambda / 2
        # Extract beamforming matrices
        self.beamforming_matrices = (action[self.cfg.N_antennaElement:] * self.cfg.beamforming_action_scaling).reshape(
            self.cfg.Number_of_com_components, self.cfg.N_antennaElement)
        
        # Update system model state
        self.basestation.update_system(self.antenna_array, self.beamforming_matrices)

        # Extract secretary rate
        self.secret_rate = self.basestation.get_secretary_rate().sum()

        # Reward calculation
        self.reward = 0
        self.reward = self.reward + self.secret_rate
        truncated = False
        if (self.validating_actions(self.basestation.sensing_target.get_SINR())):
            print("reward ", self.reward)
            reward = self.reward
        else:
            reward = 0

        # Terminate after max_steps
        # Update logic terminate, chỉ dừng sau max_steps
        self.step_counter += 1
        if self.step_counter >= self.cfg.max_steps:
            terminated = True
        else:
            terminated = False

        # These parameters only for loggin purpose on tensorboard
        self.beam_power = np.trace(np.dot(self.beamforming_matrices, np.conj(self.beamforming_matrices).T))
        self.sum_data_rate = 0
        for i in range(self.cfg.M_users):
            self.sum_data_rate = self.sum_data_rate + self.basestation.usrList[i].get_data_rate()
        self.eve_sum_data_rate = np.array(self.basestation.eva.EACH_users_data_rate).sum()

        # Update observation and info
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

