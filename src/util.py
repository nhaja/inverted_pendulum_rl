from enum import Enum
import numpy as np

class StateEnum(Enum):
    theta = int(0)
    d_theta: int = 1
    x: int = 2
    d_x: int = 3


def normalize_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi
