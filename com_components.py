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
    """
    distance = 0
    SINR = 0
    data_rate = 0
    
    def __init__(self, antenna_array, configuration):
        self.cfg = configuration
        self.distance = random.uniform(self.cfg.d_min, self.cfg.d_max)
        self.angle_of_direction = np.random.uniform(low=0, high=np.pi, size=(self.cfg.l_path_propagation,))  # an array of L in (0~pi)
        # !! raising problem with low data rate here where path gain sometime overflow distribution range
        self.path_gain_peak = (self.cfg.d0_path_loss_ref * ((self.distance)**(-1*self.cfg.alpha_path_loss_exponent)) ) / self.cfg.l_path_propagation # similar to ref formular from [2]
        self.path_gain = np.random.normal(loc=self.path_gain_peak/2, scale=self.path_gain_peak, size=self.cfg.l_path_propagation)  # Blm ~ path gain for the Lth transmit path

        self.channel_vector = np.zeros(self.cfg.N_antennaElement)
        self.update_attribute(antenna_array)

    """
    Update user's attribute as :
        field response vector
        channel vector 
    """
    def update_attribute(self, antenna_array):
        # Formular (1) for Field response vector a_m(x) shape = (L_path propagation , antenna_element) (verified) 
        self.field_response_vector = np.array(
            [np.exp(1j * (2 * np.pi / self.cfg.dlambda) * antenna_array * np.cos(angle)) for angle in self.angle_of_direction])

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
        self.distance = random.uniform(self.cfg.d_min, self.cfg.d_max)
        self.angle_of_direction = np.random.uniform(low=0, high=np.pi, size=(self.cfg.l_path_propagation,))
        self.path_gain_peak = (self.cfg.d0_path_loss_ref * (
                    (self.distance) ** (-1 * self.cfg.alpha_path_loss_exponent))) / self.cfg.l_path_propagation
        self.path_gain = np.random.normal(loc=self.path_gain_peak / 2, scale=self.path_gain_peak,
                                          size=self.cfg.l_path_propagation)  # Blm ~ path gain for the Lth transmit path
        self.channel_vector = np.zeros(self.cfg.N_antennaElement)
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

