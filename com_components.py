import numpy as np
import random
import math



class User:
    """
    User / receiver object, contain information and state including:
        - Distance from BS
        - Angle of direction of user's path propagation
        - Transmit path gain (depended on distance)
        - Field response vector
        - Channel vector
        - Data rate
        - SINR

        - Position (x, y)
        - Speed (s)
        - Direction (d)
        - Mean Direction (d_mean)

    """
    distance = 0
    SINR = 0
    data_rate = 0
    
    def __init__(self, antenna_array, configuration):
        self.cfg = configuration
        
        # (Gauss-Markov)
        self.x = random.uniform(self.cfg.x_min, self.cfg.x_max)
        self.y = random.uniform(self.cfg.y_min, self.cfg.y_max)
        # Cập nhật tính lại distance
        self.distance = np.sqrt(self.x**2 + self.y**2)

        # Speed and Direction
        self.speed = self.cfg.gm_mean_speed
        self.mean_direction = np.random.uniform(0, 2*np.pi)
        self.direction = self.mean_direction

        # Multipath AoD = LOS + small perturbation
        self.multipath_offset = np.random.normal(
            loc=0, 
            scale=self.cfg.multipath_angle_spread, 
            size=self.cfg.l_path_propagation
        )
        self._update_path_propagation()

        
        self.update_attribute(antenna_array)

    def _update_path_propagation(self):
        los_angle = np.arctan2(self.y, self.x)

        # Cập nhật lại angle_of_direction theo multipath_offset - Nhiễu này có thể gây ra vấn đề khi angle_of_direction vượt quá 2pi
        self.angle_of_direction = los_angle + self.multipath_offset
    
        # Tính lại path-gain
        safe_dist = max(self.distance, 1.0)
        # Tính toán lại path gain peak, safe_dist tránh hiện tượng đứng vuông góc dứoi BS
        self.path_gain_peak = (self.cfg.d0_path_loss_ref * ((safe_dist)**(-1*self.cfg.alpha_path_loss_exponent)) ) / self.cfg.l_path_propagation
        self.path_gain = np.random.normal(loc=self.path_gain_peak/2, scale=self.path_gain_peak, size=self.cfg.l_path_propagation)

    # Update lại vị trí -> Update lại cả distance và path gain
    def update_location(self, x, y):
        self.x = x
        self.y = y
        self.distance = np.sqrt(x**2 + y**2)
        self._update_path_propagation()

    """
    Update user's attribute as :
        field response vector
        channel vector 
    """
    def update_attribute(self, antenna_array):
        # Formular (1) for Field response vector a_m(x) shape = (L_path propagation , antenna_element) (verified) 
        self.field_response_vector = np.array(
            [np.exp(1j * (2 * np.pi / self.cfg.dlambda) * antenna_array * np.cos(angle)) for angle in self.angle_of_direction])

        self.channel_vector = np.zeros(self.cfg.N_antennaElement)
        # Formular (4) Calculating channel vector h_m = sum of path gain B_m * field response vector a_m with j from all path propagation
        # This is a vector frv /times a number gain
        # Output equal to shape (antenna_element, 1)
        for frv, gain in zip(self.field_response_vector, self.path_gain):
            self.channel_vector = self.channel_vector + (frv * gain)

    def get_field_response_vector(self):
        return self.field_response_vector

    def get_channel_vector(self):
        return self.channel_vector
    
    def set_SINR(self, sinr):
        self.SINR = sinr
        self._set_data_rate(sinr)

    def get_SINR(self):
        return self.SINR
    
    def _set_data_rate(self, SINR):
        self.data_rate = np.log2(1+SINR)

    def get_data_rate(self):
        return self.data_rate
    
    


class Target:
    """
    This is sensing target object, containing following attributes:
        - Distance from BS
        - Angle of arrival 
        - Field response vector
        - Channel vector
        - SINR
        - Data rate (for absolute no reason)
    """
    distance = 0
    angle_of_arrival = 0
    SINR = 0
    data_rate = 0

    def __init__(self, antenna_array, configuration):
        self.cfg = configuration
        if self.cfg.d_fixed_sensing_target != 0:
            self.distance = self.cfg.d_fixed_sensing_target  # For debug purpose only
        else:
            self.distance = random.uniform(self.cfg.d_min, self.cfg.d_max)
        self.angle_of_arrival = self.cfg.target_AoA
        self.doppler_freq = 2 * self.cfg.v_target_velocity * (self.cfg.fc_carrier_signal_freq/self.cfg.c_speed_of_light)  # fd ~ Doppler frequency = 2 v fc /c (11)
        self.attenuation_coefficient = np.sqrt( ((self.cfg.dlambda**2) * self.cfg.alpha_sc) / (((4*np.pi)**3) * (self.distance**4)) )  # Formular (12)
        self.update_attributes(antenna_array)

    """
    Update sensing target object attributes:
        Field response vector
        Channel vector
    """
    def update_attributes(self, antenna_array):
        # Formular (1) for field response vector of sensing target
        # !! For some reason, this pile of dog shit turn out to be shape (1,4) while it supposed tobe (4,)
        self.field_response_vector = np.array(
            [np.exp(1j * (2 * np.pi / self.cfg.dlambda) * antenna_array * np.cos(self.angle_of_arrival))]).reshape(self.cfg.N_antennaElement,)
        # Formular (11) for radar channelS. This does return shape(N antenna element, 1) as expected
        self.channel_vector = self.attenuation_coefficient * (np.exp(1j * 2 * np.pi * self.doppler_freq * self.cfg.t_time_of_target)) * self.field_response_vector

    def get_field_response_vector(self):
        return self.field_response_vector
    
    def get_channel_vector(self):
        return self.channel_vector
    
    def set_SINR(self, sinr):
        self.SINR = sinr

    def get_SINR(self):
        return self.SINR
    

class Eva:
    """
    This is Eavedropper object, containing following attributes:
        - Distance 
        - Angle of direction
        - Path gain
        - Field response vector
        - Channel vector
        - Each user SINR ~ Eavedropper on user's transmission SINR
        - Each user data rate ~ convert to data rate heard from EACH user separate
    """
    distance = 0
    EACH_users_SINR = []
    EACH_users_data_rate = []

    def __init__(self, antenna_array, configuration):

        self.cfg = configuration
        
        # Position initialization - Tính lại biến distance
        self.x = random.uniform(self.cfg.x_min, self.cfg.x_max)
        self.y = random.uniform(self.cfg.y_min, self.cfg.y_max)
        self.distance = np.sqrt(self.x**2 + self.y**2)
        
        self.angle_of_direction = np.random.uniform(low=0, high=np.pi, size=(self.cfg.l_path_propagation,))
        self.path_gain_peak = (self.cfg.d0_path_loss_ref * (
                    (self.distance) ** (-1 * self.cfg.alpha_path_loss_exponent))) / self.cfg.l_path_propagation
        self.path_gain = np.random.normal(loc=self.path_gain_peak / 2, scale=self.path_gain_peak,
                                          size=self.cfg.l_path_propagation)  # Blm ~ path gain for the Lth transmit path
        
        self.update_attribute(antenna_array)
        self.EACH_users_SINR = []
        self.EACH_users_data_rate = []

    """
    Update eavedropper object attributes:
        Field response vector
        Channel vector
    """
    def update_attribute(self, antenna_array):
        # reset sinr and data rate of each user-eva
        self.EACH_users_SINR = []
        self.EACH_users_data_rate = []

        # Formular (1) for Field response vector a_m(x) shape = (13,4)
        self.field_response_vector = np.array(
            [np.exp(1j * (2 * np.pi / self.cfg.dlambda) * antenna_array * np.cos(angle)) for angle in self.angle_of_direction])

        self.channel_vector = np.zeros(self.cfg.N_antennaElement)
        
        # Formular (4) Calculating channel vector h_m = sum of path gain B_m * field response vector a_m with j from all path propagation
        for frv, gain in zip(self.field_response_vector, self.path_gain):
            self.channel_vector = self.channel_vector + (frv * gain)

        self.channel_vector = self.channel_vector - (self.cfg.eva_channel_eva_error_percentage*np.absolute(self.channel_vector))

    def get_channel_vector(self):
        return self.channel_vector

    def append_user_eva_SINR(self, sinr):
        self.EACH_users_SINR.append(sinr)
        self._append_user_eva_data_rate(sinr)

    def _append_user_eva_data_rate(self, sinr):
        self.EACH_users_data_rate.append(np.log2(1+sinr))

    def get_eva_user_data_rate(self, i):
        return self.EACH_users_data_rate[i]

