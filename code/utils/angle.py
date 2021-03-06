import numpy as np


def normalize_from(th, th_min):
    # return ca.fmod(th - th_min, 2 * ca.pi) + th_min
    return ((th - th_min) % (2 * np.pi)) + th_min


def zero_to_two_pi(th):
    """Normalize angle between -pi and pi"""
    return normalize_from(th, 0)


def minus_pi_to_pi(th):
    """Normalize angle between -pi and pi"""
    return normalize_from(th, -np.pi)
