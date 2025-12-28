import os
from glob import glob
from signal_processing_functions import fix_timestamps
import numpy as np
import pandas as pd
import pathlib

columns_wrist = ['timeStamp', 'lsm_accelX', 'lsm_accelY', 'lsm_accelZ']
columns_thigh = ['flexTimeStamp', 'flex_lsm_accelX', 'flex_lsm_accelY', 'flex_lsm_accelZ']

packet_sizes = 2


def load_data(folder):
    measurements = {}
    for id in os.listdir(folder):
        measurements[id] = {}
        for meas in os.listdir(pathlib.Path(folder, id)):
            measurements[id][meas] = {}
            for act in os.listdir(pathlib.Path(folder, id, meas)):
                measurements[id][meas][act] = {}          
                file = str(pathlib.Path(folder, id, meas, act, '*.csv'))
                data = pd.read_csv(glob((file))[0], delimiter=';', skiprows=2)
                c_data_wrist = data[columns_wrist].dropna().to_numpy()
                c_data_thigh = data[columns_thigh].dropna().to_numpy()
                c_data_wrist[:,0] = fix_timestamps(c_data_wrist[:, 0], packet_sizes)*0.001
                c_data_thigh[:,0] = fix_timestamps(c_data_thigh[:, 0], packet_sizes)*0.001
                measurements[id][meas][act]['wrist'] = c_data_wrist.astype(float) 
                measurements[id][meas][act]['thigh'] = c_data_thigh.astype(float)              

    return measurements

            

            
