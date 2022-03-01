import numpy as np


def normalize_angle(theta):
    """Normalize angle between -pi and pi"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi
