import numpy as np


def normalize_angle(theta):
    """Normalize angle between -pi/2 and 3pi/2"""
    return ((theta + np.pi / 2) % (2 * np.pi)) - np.pi / 2
