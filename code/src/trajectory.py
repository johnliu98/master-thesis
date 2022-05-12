from dataclasses import dataclass
import numpy as np


@dataclass
class Pose:
    x: float
    y: float
    yaw: float

    @property
    def norm(self):
        return np.sqrt(self.x**2 + self.y**2)

    def to_array(self):
        return np.array([self.x, self.y, self.yaw])

    def __add__(self, other):
        return Pose(self.x + other.x, self.y + other.y, self.yaw + other.yaw)

    def __sub__(self, other):
        return Pose(self.x - other.x, self.y - other.y, self.yaw - other.yaw)


def distance(p1: Pose, p2: Pose) -> float:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return np.sqrt(dx * dx + dy * dy)


def dot_product(p1: Pose, p2: Pose) -> float:
    return p1.x * p2.x + p1.y * p2.y


@dataclass
class Trajectory:
    poses: list[Pose]
    width: float = 4

    def compute_errors(self, p: Pose):
        dist = np.array([distance(p, p_) for p_ in self.poses])
        i = np.argmin(dist)
        p_r = self.poses[i]

        err_y = (p.x - p_r.x) * np.sin(-p_r.yaw) + (p.y - p_r.y) * np.cos(-p_r.yaw)
        err_yaw = p.yaw - p_r.yaw

        return err_y, err_yaw

    def pose(self, s: float):
        d = 0
        for p1, p2 in zip(self.poses, self.poses[1:]):

            if d > s:
                return p1

            d += distance(p1, p2)

        return self.poses[-1]

    def plot(self, ax) -> None:
        ax.plot(
            [p.x for p in self.poses], [p.y for p in self.poses], label="ref", zorder=0
        )
        ax.plot(
            [p.x - self.width / 2 * np.sin(p.yaw) for p in self.poses],
            [p.y + self.width / 2 * np.cos(p.yaw) for p in self.poses],
            color="grey",
            label="road",
            zorder=0,
        )
        ax.plot(
            [p.x + self.width / 2 * np.sin(p.yaw) for p in self.poses],
            [p.y - self.width / 2 * np.cos(p.yaw) for p in self.poses],
            color="grey",
            zorder=0,
        )


def create_trajectory(curvatures: list[float], ds: float) -> list[Pose]:
    x, y, yaw = 0, 0, 0
    poses = [Pose(x, y, yaw)]
    for c in curvatures:
        x += ds * np.cos(yaw + ds * c)
        y += ds * np.sin(yaw + ds * c)
        yaw += ds * c
        poses.append(Pose(x, y, yaw))

    return Trajectory(poses)
