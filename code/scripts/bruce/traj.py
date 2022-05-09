import numpy as np
from src.trajectory import create_trajectory

ds = 2 * np.pi / 10
curvatures = 200 * [0.05]
traj = create_trajectory(curvatures, ds)

import matplotlib.pyplot as plt

plt.scatter([p.x for p in traj.points], [p.y for p in traj.points], s=1)
ax = plt.gca()
ax.set_aspect("equal")
plt.grid()
plt.show()
