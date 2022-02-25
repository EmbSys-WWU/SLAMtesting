import numpy as np

from obstacle import ObstacleID


class Observation:
    def __init__(self, obstacle_id: ObstacleID, angle: float, distance: float):
        self.obstacle = obstacle_id
        self.angle = angle
        self.distance = distance

    def to_array(self) -> np.ndarray:
        """
        Converts the observation to a numpy array

        :return: An array of the form [distance, angle, id]
        """

        return np.array([self.distance, self.angle, self.obstacle.v])
