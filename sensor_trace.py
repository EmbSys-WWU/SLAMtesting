from typing import List, Tuple, Union

import numpy as np

from motion import Velocity, Odometry
from observation import Observation


class TimeStep:
    def __init__(self, motion: Union[Velocity, Odometry], observations: List[Observation]):
        self.motion = motion
        self.observations = observations


class SensorTrace:
    def __init__(self, starting_position: np.ndarray):
        self.starting_position = starting_position
        self.time_steps: List[TimeStep] = []
        self.coordinates: List[Tuple[int, int]] = []
        self.index = 0

    def add_step(self, step: TimeStep, x: int, y: int) -> None:
        self.time_steps.append(step)
        self.coordinates.append((x, y))

    def __len__(self) -> int:
        return len(self.time_steps) - self.index

    def get_next(self) -> TimeStep:
        result = self.time_steps[self.index]
        self.index += 1
        return result

    def has_next(self) -> bool:
        return len(self.time_steps) > self.index

    def reset(self) -> None:
        self.index = 0
