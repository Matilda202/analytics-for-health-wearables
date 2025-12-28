from dataclasses import dataclass
import numpy as np

@dataclass
class MeasurementData:
    """
    Class representing data from each measurement.
    """
    name: str = None
    ts: np.ndarray = None
    acc_x_thigh: np.ndarray = None
    acc_y_thigh: np.ndarray = None
    acc_z_thigh: np.ndarray = None
    acc_x_wrist: np.ndarray = None
    acc_y_wrist: np.ndarray = None
    acc_z_wrist: np.ndarray = None
    label: int = None