class ObstacleID:
    def __init__(self, value: int):
        self.v = value

    def __eq__(self, other) -> bool:
        return self.v == other.v


class Obstacle:
    def __init__(self, obstacle_id: ObstacleID, x: int, y: int):
        self.id = obstacle_id
        self.x = x
        self.y = y
