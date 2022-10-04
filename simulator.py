from math import hypot, atan2, pi
from random import gauss, random, uniform
from typing import List

from map import Map
from observation import Observation
from robot import Robot


class Simulator:

    def __init__(self, test_map: Map):
        self.test_map = test_map

    def simulate(self,
                 robot: Robot,
                 sensor_range: float,
                 range_bias: float,
                 range_variance: float,
                 angle_bias: float,
                 angle_variance: float,
                 outlier_probability: float,
                 rotational_error: int,
                 sensor_fov: float) -> List[Observation]:
        """
        Generates a simulated view in the map from a given position

        :param robot: Simulated robot
        :param sensor_range: Range of the robot's vision
        :param range_bias: Bias of the range sensor
        :param range_variance: Variance of the range sensor
        :param angle_bias: Bias of the angle sensor
        :param angle_variance: Variance of the angle sensor
        :param outlier_probability: Probability of an outlier (max range reading or obstructed reading)
        :param rotational_error: Multiple of 360° to be added to angular measurements and odometry
        :param sensor_fov: Field of view of the sensor
        :return: A list of observations
        """

        observations = []

        for o in self.test_map.obstacles:
            dist = hypot(o.x - robot.x, o.y - robot.y)
            angle = (atan2(o.y - robot.y, o.x - robot.x) - robot.rot + pi) % (2 * pi) - pi
            if dist < sensor_range and min(angle, 2 * pi - angle) < sensor_fov:

                # Add uncertainty to the observation
                perceived_distance_mean = dist + range_bias
                if perceived_distance_mean < 0:  # Don't allow bias to make a measurement negative
                    perceived_distance_mean = dist / 2.0
                perceived_angle_mean = angle + angle_bias
                perceived_distance = gauss(perceived_distance_mean, range_variance)
                while perceived_distance < 0:  # prevent random negative measurements
                    perceived_distance = gauss(perceived_distance_mean, range_variance)
                perceived_angle = (gauss(perceived_angle_mean, angle_variance) + pi) % (2 * pi) - pi

                # Add rotational error if desired
                perceived_angle += rotational_error * (2 * pi)

                # Randomly create outlier readings according to the beam endpoint observation model
                r = random()
                if r < outlier_probability / 2:
                    perceived_distance = sensor_range  # missed object; max range reading
                elif r < outlier_probability:
                    perceived_distance = uniform(0, dist)  # measurement obstructed; short reading

                observations.append(Observation(o.id, perceived_angle, perceived_distance))

        return observations
