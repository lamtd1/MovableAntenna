import subprocess
import argparse
import os 
import math
import sys
import platform

# Use the current python interpreter
python_dir = sys.executable

def dbm_to_watt(dbm):
    return 10 ** (dbm / 10) / 1000

def run_command(cmd_args):
    """
    Executes a command in a subprocess.
    """
    # Construct the command list
    # On Windows, 'start /wait' was used to open a new window and wait.
    # On macOS/Linux, we just run the command directly in the current terminal.
    # If we want to mimic 'start /wait', we just call it.
    
    cmd_str = f"{python_dir} main.py {cmd_args}"
    print(f"Executing: {cmd_str}")
    
    try:
        subprocess.check_call([python_dir, 'main.py'] + cmd_args.split())
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(1)

def user_number_changes():
    model = ['DDPG', 'PPO', 'A2C']
    users = [2,3,4,5]
    total_timestep = 30000
    for mod in model:
        for i in range(len(users)):
            run_command(f'-m {mod} -Mu {users[i]} -t {total_timestep}')

def antenna_element_number_changes():
    model = ['DDPG', 'PPO', 'A2C']
    antenna = [2,4,8]
    total_timestep = 300000
    for mod in model:
        for i in range(len(antenna)):
            run_command(f'-m {mod} -N {antenna[i]} -t {total_timestep}')

def SINR_limit_changes():
    # model = ['DDPG', 'PPO', 'A2C']
    model = ['PPO', 'A2C']
    SINR = [5,10,15,20]
    total_timestep = 300000
    for mod in model:
        for i in range(len(SINR)):
            run_command(f'-m {mod} -Ms {dbm_to_watt(SINR[i])} -t {total_timestep}')

def tx_power_change():
    model = ['PPO', 'A2C']
    power = [15,20,25,30]
    total_timestep = 300000
    for mod in model:
        for i in range(len(power)):
            run_command(f'-m {mod} -po {dbm_to_watt(power[i])} -t {total_timestep}')

def run_all_drl_model():
    model = ['PPO']
    total_timestep = 300_000
    for i in range(5):
        for mod in model:
            run_command(f'-m {mod} -t {total_timestep}')


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode')
args = parser.parse_args()

"""
What is going on here?
    This code only for massive automation run code. Option here we're providing 
    include:
        0. Massive run all the alternative environment changes model declared
        inside the paper

        1. Run with all user changes {2,3,4,5} with PPO  

        2. Run with changes in number of Antenna elements {2,4,8}

        3. Impact of SINR limited apply on sensing target {5,10,15,20} [dBm]

        4. Impact of maximum transmiting power P0 {15,20,25,30} [dBm]

        5. Running check on all 3 DRL model with base configuration

        6. (improvise) Custom running command

"""

match args.mode:
    case '0':
        user_number_changes()
        antenna_element_number_changes()
        SINR_limit_changes()
        tx_power_change()
        

    case '1':
        user_number_changes()

    case '2':
        antenna_element_number_changes()

    case '3':
        SINR_limit_changes()

    case '4':
        tx_power_change()

    case '5':
        run_all_drl_model()

    case '6':
        print("Under construction!")

    case None:
        print("what do you want to do? Open the source code and read!!!")


