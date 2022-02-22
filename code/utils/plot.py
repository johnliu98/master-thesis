import numpy as np
import matplotlib.patches as patches


def CircleSector(center, radius, theta1, theta2, resolution=50, **kwargs):
    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack(
        (radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1])
    )
    points = np.hstack([np.zeros((2, 1)), points])
    # build the polygon and add it to the axes
    poly = patches.Polygon(points.T, closed=True, **kwargs)
    return poly


def add_to_plot(plot, data):
    prev_data = plot[0].get_xydata().T
    new_data = np.zeros((prev_data.shape[0], prev_data.shape[1] + 1))
    new_data[:, :-1] = prev_data
    new_data[:, -1] = data
    plot[0].set_data(new_data)
