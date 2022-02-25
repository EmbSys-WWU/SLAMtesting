from math import hypot, atan2
from random import choice
from typing import List, Optional

from obstacle import Obstacle, ObstacleID


class Map:
    def __init__(self, size: int):
        self.size = size
        self.obstacles: List[Obstacle] = []
        self.new_obstacles: List[Obstacle] = []

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def done(self) -> None:
        """
        Tells the map that no more obstacles will be added, essential for picking obstacles in order
        """

        self.obstacles.sort(key=lambda o: atan2(o.y, o.x))

    def fits(self, obstacle: Obstacle, minimum_distance: float) -> bool:
        """
        Determines whether a given obstacle could be inserted into the map while maintaining the minimum distance
        to all other obstacles within the map

        :param obstacle: The obstacle to be examined
        :param minimum_distance: minimum distance to be maintained
        :return: True if minimum distance can be maintained, False otherwise
        """

        for o in self.obstacles:
            if hypot(o.x - obstacle.x, o.y - obstacle.y) < minimum_distance:
                return False

        return True

    def visited(self, obstacle: Optional[Obstacle]) -> bool:
        """
        Determines whether an obstacle has been visited before

        :param obstacle: The obstacle to be examined
        :return: True if it has been visited, False otherwise
        """

        if obstacle is None:
            return True         # if obstacle is None, a new one should be chosen, so mark it as visited
        else:
            return obstacle not in self.new_obstacles

    def visit(self, obstacles: List[ObstacleID]) -> None:
        """
        Marks the contained obstacles as visited

        :param obstacles: List of obstacle ids to be marked as visited
        """

        self.new_obstacles = [o for o in self.new_obstacles if o.id not in obstacles]

    def get_random_destination(self) -> Obstacle:
        """
        Selects a random new obstacle as a destination that has not been visited yet

        :return: Obstacle that has not yet been visited
        """

        # if all obstacles have been visited, reset the list of visited objects
        if not self.new_obstacles:
            self.new_obstacles = self.obstacles.copy()

        return choice(self.new_obstacles)

    def get_next_destination(self) -> Obstacle:
        """
        Selects the next obstacle in order of angular position relative to (0, 0)

        :return: Obstacle that has not yet been visited
        """

        # if all obstacles have been visited, reset the list of visited objects
        if not self.new_obstacles:
            self.new_obstacles = self.obstacles.copy()

        return self.new_obstacles[0]
