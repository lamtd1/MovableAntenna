import numpy as np


class Config:
    N_antennaElement = 4  # Number of antenna element
    M_users = 3  # Number of users

    target_AoA = 60 * (np.pi/180)  # convert target Angle of Arrival from 60 -> [radius]
    l_path_propagation = 13  # Each usr have 13 different path propagation to communicate with basestation


    d_min = 25   # comm component minimum distance from base station in [m]
    d_max = 100  # comm component maximum distance from base station in [m]
    d_fixed_sensing_target = 25  # incase 0 ~ random depend on d_min/max


    dlambda = 0.1  # wave length in [m]
    MA_size = 10
    A_antennaRange = MA_size*dlambda  # movement range of antenna elements in [m]
    D0_spacingAntenna = dlambda/2  # minimum space between 2 antenna element in [m]


    d0_path_loss_ref = 3e-2  # [w] path loss at ref distance of 1m ~ -30[dB]
    alpha_path_loss_exponent = 2.2  # path loss exponent
    sigma_Noise = 1e-11
    P0_basestation_power = 0.316227766  # [W] maximum power of base station ~ 25 [dBm]

    fc_carrier_signal_freq = 2400000000  # [Hz] sensing carrier signal frequency ~ 2.4GHz
    v_target_velocity = 20#299792458  # [m/s] target moving velocity
    c_speed_of_light = 3e8 # [m/s] Speed of light
    alpha_sc = 0.5  # alpha_s, alpha_c following Gaussian Distribution in range (0, 1)
    t_time_of_target = 1e-6

    sensing_SINR_min = 0.01  # ~ 10 [dBm]

    # Not clear defined in paper
    Number_of_target = 1
    Number_of_symbol = M_users + Number_of_target
    Number_of_com_components = Number_of_symbol

    # Eva configuration
    # eva channel error percentage ~ 30% !! this could be error in the future
    eva_channel_eva_error_percentage = 0.3

    # Custom parameters 
    beamforming_action_scaling = 0.15  # ~ 15%

    def init_unstatic_value(self):
        self.Number_of_symbol = self.M_users + self.Number_of_target
        self.Number_of_com_components = self.Number_of_symbol
        self.default_antenna_array = np.zeros(self.N_antennaElement)
        self.default_beam_matrices = np.zeros(self.Number_of_com_components* self.N_antennaElement* self.Number_of_symbol).reshape(self.Number_of_com_components, self.N_antennaElement, self.Number_of_symbol)  # this is a list of all beamforming matrix for all comm target

