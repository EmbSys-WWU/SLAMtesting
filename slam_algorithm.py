from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from observation import Observation
from motion import Velocity, Odometry
from map import Map


class SLAMAlgorithm(ABC):

    @abstractmethod
    def initialize(self, initial_position: np.ndarray) -> None:
        pass

    @abstractmethod
    def step(self, motion: Union[Velocity, Odometry], observations: List[Observation]) -> None:
        pass

    @abstractmethod
    def generate_map(self, map_size: int) -> Map:
        pass
