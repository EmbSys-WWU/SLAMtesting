import numpy as np


#  ----------------------------------------------------------------------------------------------  #
#  --------------------------------- ODOMETRY MOTION MODEL --------------------------------------  #
#  ----------------------------------------------------------------------------------------------  #
class Odometry:
    def __init__(self, t1: float, d: float, t2: float):
        self.turn1 = Turn(t1)
        self.drive = Drive(d)
        self.turn2 = Turn(t2)

    def convert_to_array(self) -> np.ndarray:
        return np.array([self.turn1.amount, self.drive.amount, self.turn2.amount], dtype=np.float)


class Motion:
    def __init__(self, amount: float):
        self.amount = amount


class Turn(Motion):
    def __init__(self, amount: float):
        super(Turn, self).__init__(amount)


class Drive(Motion):
    def __init__(self, amount: float):
        super(Drive, self).__init__(amount)


#  ----------------------------------------------------------------------------------------------  #
#  --------------------------------- VELOCITY MOTION MODEL --------------------------------------  #
#  ----------------------------------------------------------------------------------------------  #
class Velocity:
    def __init__(self, v: float, yaw_rate: float):
        self.v = v
        self.yaw_rate = yaw_rate

    def convert_to_array(self) -> np.ndarray:
        return np.array([self.v, self.yaw_rate], dtype=np.float)
