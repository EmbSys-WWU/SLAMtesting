from math import hypot
from typing import List

import numpy as np

import slam.slam_correct as slam1
import slam.slam_resampling as slam2
import slam.slam_rotation as slam3
import slam.slam_sign_error as slam4

from map import Map
from motion import Odometry
from observation import Observation
from obstacle import Obstacle, ObstacleID
from slam_algorithm import SLAMAlgorithm


class CorrectSLAM(SLAMAlgorithm):
    def __init__(self, n_landmark: int):
        self.landmarks = n_landmark
        self.particles: List[slam1.Particle] = []

    def initialize(self, initial_position: np.ndarray) -> None:
        initial_position[0] /= 100.0
        initial_position[1] /= 100.0
        self.particles = [slam1.Particle(self.landmarks, initial_position) for _ in range(slam1.N_PARTICLE)]

    def step(self, motion: Odometry, observations: List[Observation]) -> None:
        """
        Updates the particle set according to the FastSLAM algorithm

        :param motion: (Velocity model) motion representation
        :param observations: List of landmark observations
        """

        u = motion.convert_to_array()[1:]       # FastSLAM algorithm only uses forward motion and second turn
        u[0] /= 100.0
        u[1] = slam1.pi_2_pi(u[1])
        z = np.array([[o.distance / 100.0 for o in observations],
                      [o.angle for o in observations],
                      [o.obstacle.v for o in observations]])
        self.particles = slam1.fast_slam1(self.particles, u.reshape(2, 1), z)

    def generate_map(self, map_size: int) -> Map:
        """
        Returns the final result of the algorithm

        :return: The map produced by the algorithm over the course of the run
        """

        # Since the algorithm doesn't provide a map generation functionality, look for the particle closest to the mean
        # estimate and with the highest weight and take its map as the algorithm's result
        mean = slam1.calc_final_state(self.particles)
        closest_particle = sorted(self.particles, key=lambda p: hypot(p.x - mean[0, 0], p.y - mean[1, 0]) * p.w)[0]

        result = Map(map_size)

        for i in range(closest_particle.lm.shape[0]):
            result.add_obstacle(Obstacle(ObstacleID(i),
                                         int(round(100 * closest_particle.lm[i, 0])),
                                         int(round(100 * closest_particle.lm[i, 1]))
                                         )
                                )

        return result


class ResamplingSLAM(SLAMAlgorithm):
    def __init__(self, n_landmark: int):
        self.landmarks = n_landmark
        self.particles: List[slam2.Particle] = []

    def initialize(self, initial_position: np.ndarray) -> None:
        initial_position[0] /= 100.0
        initial_position[1] /= 100.0
        self.particles = [slam2.Particle(self.landmarks, initial_position) for _ in range(slam2.N_PARTICLE)]

    def step(self, motion: Odometry, observations: List[Observation]) -> None:
        """
        Updates the particle set according to the FastSLAM algorithm

        :param motion: (Velocity model) motion representation
        :param observations: List of landmark observations
        """

        u = motion.convert_to_array()[1:]       # FastSLAM algorithm only uses forward motion and second turn
        u[0] /= 100.0
        u[1] = slam2.pi_2_pi(u[1])
        z = np.array([[o.distance / 100.0 for o in observations],
                      [o.angle for o in observations],
                      [o.obstacle.v for o in observations]])
        self.particles = slam2.fast_slam1(self.particles, u.reshape(2, 1), z)

    def generate_map(self, map_size: int) -> Map:
        """
        Returns the final result of the algorithm

        :return: The map produced by the algorithm over the course of the run
        """

        # Since the algorithm doesn't provide a map generation functionality, look for the particle closest to the mean
        # estimate and with the highest weight and take its map as the algorithm's result
        mean = slam2.calc_final_state(self.particles)
        closest_particle = sorted(self.particles, key=lambda p: hypot(p.x - mean[0, 0], p.y - mean[1, 0]) * p.w)[0]

        result = Map(map_size)

        for i in range(closest_particle.lm.shape[0]):
            result.add_obstacle(Obstacle(ObstacleID(i),
                                         int(round(100 * closest_particle.lm[i, 0])),
                                         int(round(100 * closest_particle.lm[i, 1]))
                                         )
                                )

        return result


class RotationSLAM(SLAMAlgorithm):
    def __init__(self, n_landmark: int):
        self.landmarks = n_landmark
        self.particles: List[slam3.Particle] = []

    def initialize(self, initial_position: np.ndarray) -> None:
        initial_position[0] /= 100.0
        initial_position[1] /= 100.0
        self.particles = [slam3.Particle(self.landmarks, initial_position) for _ in range(slam3.N_PARTICLE)]

    def step(self, motion: Odometry, observations: List[Observation]) -> None:
        """
        Updates the particle set according to the FastSLAM algorithm

        :param motion: (Velocity model) motion representation
        :param observations: List of landmark observations
        """

        u = motion.convert_to_array()[1:]       # FastSLAM algorithm only uses forward motion and second turn
        u[0] /= 100.0
        z = np.array([[o.distance / 100.0 for o in observations],
                      [o.angle for o in observations],
                      [o.obstacle.v for o in observations]])
        self.particles = slam3.fast_slam1(self.particles, u.reshape(2, 1), z)

    def generate_map(self, map_size: int) -> Map:
        """
        Returns the final result of the algorithm

        :return: The map produced by the algorithm over the course of the run
        """

        # Since the algorithm doesn't provide a map generation functionality, look for the particle closest to the mean
        # estimate and with the highest weight and take its map as the algorithm's result
        mean = slam3.calc_final_state(self.particles)
        closest_particle = sorted(self.particles, key=lambda p: hypot(p.x - mean[0, 0], p.y - mean[1, 0]) * p.w)[0]

        result = Map(map_size)

        for i in range(closest_particle.lm.shape[0]):
            result.add_obstacle(Obstacle(ObstacleID(i),
                                         int(round(100 * closest_particle.lm[i, 0])),
                                         int(round(100 * closest_particle.lm[i, 1]))
                                         )
                                )

        return result


class SignErrorSLAM(SLAMAlgorithm):
    def __init__(self, n_landmark: int):
        self.landmarks = n_landmark
        self.particles: List[slam4.Particle] = []

    def initialize(self, initial_position: np.ndarray) -> None:
        initial_position[0] /= 100.0
        initial_position[1] /= 100.0
        self.particles = [slam4.Particle(self.landmarks, initial_position) for _ in range(slam4.N_PARTICLE)]

    def step(self, motion: Odometry, observations: List[Observation]) -> None:
        """
        Updates the particle set according to the FastSLAM algorithm

        :param motion: (Velocity model) motion representation
        :param observations: List of landmark observations
        """

        u = motion.convert_to_array()[1:]       # FastSLAM algorithm only uses forward motion and second turn
        u[0] /= 100.0
        u[1] = slam4.pi_2_pi(u[1])
        z = np.array([[o.distance / 100.0 for o in observations],
                      [o.angle for o in observations],
                      [o.obstacle.v for o in observations]])
        self.particles = slam4.fast_slam1(self.particles, u.reshape(2, 1), z)

    def generate_map(self, map_size: int) -> Map:
        """
        Returns the final result of the algorithm

        :return: The map produced by the algorithm over the course of the run
        """

        # Since the algorithm doesn't provide a map generation functionality, look for the particle closest to the mean
        # estimate and with the highest weight and take its map as the algorithm's result
        mean = slam4.calc_final_state(self.particles)
        closest_particle = sorted(self.particles, key=lambda p: hypot(p.x - mean[0, 0], p.y - mean[1, 0]) * p.w)[0]

        result = Map(map_size)

        for i in range(closest_particle.lm.shape[0]):
            result.add_obstacle(Obstacle(ObstacleID(i),
                                         int(round(100 * closest_particle.lm[i, 0])),
                                         int(round(100 * closest_particle.lm[i, 1]))
                                         )
                                )

        return result


def slam_factory_correct(n_landmarks: int) -> CorrectSLAM:
    """
    Sets up a SLAM algorithm for the test

    :param n_landmarks: Number of landmarks to use the algorithm for
    :return: A FastSLAM instance
    """

    return CorrectSLAM(n_landmarks)


def slam_factory_rotation(n_landmarks: int) -> RotationSLAM:
    """
    Sets up a SLAM algorithm for the test

    :param n_landmarks: Number of landmarks to use the algorithm for
    :return: A FastSLAM instance
    """

    return RotationSLAM(n_landmarks)


def slam_factory_resampling(n_landmarks: int) -> ResamplingSLAM:
    """
    Sets up a SLAM algorithm for the test

    :param n_landmarks: Number of landmarks to use the algorithm for
    :return: A FastSLAM instance
    """

    return ResamplingSLAM(n_landmarks)


def slam_factory_sign_error(n_landmarks: int) -> SignErrorSLAM:
    """
        Sets up a SLAM algorithm for the test

        :param n_landmarks: Number of landmarks to use the algorithm for
        :return: A FastSLAM instance
        """

    return SignErrorSLAM(n_landmarks)
