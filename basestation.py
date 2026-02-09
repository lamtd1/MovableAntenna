import numpy as np
from com_components import User, Target, Eva
import math


class BS:

    def __init__(self, antenna_array, configuration):
        self.cfg = configuration
        self.secretary_rate = np.zeros(self.cfg.M_users)
        """!
        @brief [Create new System model as Base station contain number of communication targets]
            Number of communication targets includings:
                M users
                1 Sensing target
        """
        
    
        self.usrList = []
        for i in range(self.cfg.M_users):
            self.usrList.append(User(antenna_array, configuration))

    
        self.sensing_target = Target(antenna_array, configuration)
        self.secretary_rate = np.zeros(self.cfg.M_users)


    def update_system(self, antenna_array, beamforming_matrices):
        self.secretary_rate = np.zeros(self.cfg.M_users)

        # Calculate users SINR
        channel_vector_list = []  # Contain list of users channel vector h_m
        sum_norm = 0  # Sum of || h^H_m * W_m ||^2 with m from 0 ~ M

        for user in self.usrList:
            user.update_attribute(antenna_array)  # Update user's field response vector and channel vector
            channel_vector_list.append(user.get_channel_vector().conj().T)  # Extract and transpose channel vector ~ h^H_m
        self.sensing_target.update_attributes(antenna_array)  # Update target's FRV and channel vector

   
        channel_vector_list.append(self.sensing_target.get_channel_vector().conj().T)
        """
        By calculating EACH user's channel vector with all beam and saving the sum to find SINR
        """

        delta_eve_sum = 1 # short circurt paramere
        
        for i in range(self.cfg.Number_of_com_components):
            channel_beam_pair = (np.linalg.norm(channel_vector_list[i] @ beamforming_matrices[i][:]))**2
            sum_norm = 0
            for j in range(self.cfg.Number_of_com_components):
                if j == i:
                    pass
                else:
                    if i is (self.cfg.Number_of_com_components - 1):
                        pass
                    # else:
                    #    sum_eva_user += (np.linalg.norm(eva_channel_vector_transpose @ beamforming_matrices[j][:]))**2
                    sum_norm += (np.linalg.norm(channel_vector_list[i] @ beamforming_matrices[j][:]))**2
            if i is (self.cfg.Number_of_com_components - 1):
                target_SINR = channel_beam_pair / (sum_norm + self.cfg.sigma_Noise)
                # Exporting sensing SIRN
                self.sensing_target.set_SINR(target_SINR)
            else:
                usr_SINR = channel_beam_pair / (sum_norm + self.cfg.sigma_Noise)
                # Exporting user-basestation transmit SINR and datarate
                self.usrList[i].set_SINR(usr_SINR)

                # eva_user_SINR = eva_user_channel_beam_pair / (delta_eve_sum+sum_eva_user+self.cfg.sigma_Noise)
                # Exporting eavedropper SINR and data rate
                # self.eva.append_user_eva_SINR(eva_user_SINR)

                # Calculating secretary rate
                # secrate = self.usrList[i].get_data_rate() - self.eva.get_eva_user_data_rate(i)
                # if (secrate<0):
                #    self.secretary_rate[i] = 0
                # else:
                #    self.secretary_rate[i] = secrate


    def get_secretary_rate(self):
        return self.secretary_rate
        #-------------------    