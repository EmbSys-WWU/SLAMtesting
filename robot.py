from math import pi, sin, cos


class Robot:
    def __init__(self, x: int, y: int, rot: float):
        self.x = x
        self.y = y
        self.rot = rot

    def move_by(self, d_x: int, d_y: int, d_rot: float) -> None:
        self.x += d_x
        self.y += d_y
        self.rot = (self.rot + d_rot + pi) % (2 * pi) - pi

    def move_by_odometry(self, d: float, rot: float) -> None:
        """
        Moves the robot a given distance forward and then turns it by a given angle

        :param d: Distance to drive forward
        :param rot: Angle to rotate around
        """

        dx = int(round(cos(self.rot) * d))
        dy = int(round(sin(self.rot) * d))
        self.move_by(dx, dy, rot)
