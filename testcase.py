from typing import Callable

from PIL import Image, ImageDraw

from map import Map
from sensor_trace import SensorTrace
from slam_algorithm import SLAMAlgorithm


OBSTACLE_THICKNESS = 1
OBSTACLE_COLOR = "#ff0000"      # red
TRAJECTORY_COLOR = "#0000ff"    # blue


class TestCase:
    def __init__(self, ground_truth: Map, sensor_trace: SensorTrace):
        self.ground_truth = ground_truth
        self.sensor_trace = sensor_trace

    def execute(self,
                slam_factory: Callable[[int], SLAMAlgorithm],
                metric: Callable[[Map, Map, str], float],
                save_file: str = 'execution_result.png') -> float:
        """
        Executes the test case with a given slam algorithm and determines whether the test was passed

        :param slam_factory: Function to set up the algorithm for the test
        :param metric: Function that determines whether a SLAM result is a pass or a fail
        :param save_file: File name to save an image of the result to
        :return: True if passed, False otherwise
        """

        it = 1

        slam_algorithm = slam_factory(len(self.ground_truth.obstacles))

        slam_algorithm.initialize(self.sensor_trace.starting_position)

        while self.sensor_trace.has_next():
            if it % 100 == 0:
                print(f"Step {it}")
            next_step = self.sensor_trace.get_next()
            slam_algorithm.step(next_step.motion, next_step.observations)
            it += 1

        self.sensor_trace.reset()
        result = slam_algorithm.generate_map(self.ground_truth.size)

        return metric(self.ground_truth, result, save_file)

    def visualize(self, file: str) -> None:
        """
        Generates a visual representation of the map and the vehicle trajectory on it (not used in execution of the
        test simulation, only for validation of the auto-generation of test cases)

        :param file: Name of file to save to (png format)
        """

        # Convert to centimeters
        map_size = self.ground_truth.size * 100

        image = Image.new(mode='RGB', size=(map_size, map_size), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Draw obstacles
        for o in self.ground_truth.obstacles:
            # Convert to centimeters (obstacles are in millimeters) and also from [-n/2, n/2] to [0, n]
            x = int(round(o.x / 10.0 + map_size / 2.0))
            y = int(round(o.y / 10.0 + map_size / 2.0))

            # Define bounding box of ellipse
            lower_x = x - OBSTACLE_THICKNESS
            lower_y = y - OBSTACLE_THICKNESS
            upper_x = x + OBSTACLE_THICKNESS
            upper_y = y + OBSTACLE_THICKNESS
            bounding_box = [(lower_x, lower_y), (upper_x, upper_y)]

            draw.ellipse(xy=bounding_box, fill=OBSTACLE_COLOR, outline=OBSTACLE_COLOR)

        # Draw trajectory
        # x, y, rot = self.sensor_trace.starting_position
        # x = int(round(x / 10.0 + map_size / 2.0))
        # y = int(round(y / 10.0 + map_size / 2.0))
        # for step in self.sensor_trace.time_steps:
        #     v = step.motion.v / 10.0
        #     yaw_rate = step.motion.yaw_rate
        #
        #     turn_radius = abs(v / yaw_rate)
        #     distance = abs(2 * turn_radius * sin(yaw_rate / 2))  # since t=1s, alpha = yaw_rate
        #     angle = rot + yaw_rate / 2
        #
        #     new_x = x + int(round(distance * cos(angle)))
        #     new_y = y + int(round(distance * sin(angle)))
        #     new_rot = rot + yaw_rate
        #
        #     draw.arc(xy=[(x, y), (new_x, new_y)], start=degrees(rot), end=degrees(new_rot), fill=TRAJECTORY_COLOR)
        #
        #     x = new_x
        #     y = new_y
        #     rot = new_rot

        coordinates = [(int(round(x / 10.0 + map_size / 2.0)), int(round(y / 10.0 + map_size / 2.0)))
                       for x, y in self.sensor_trace.coordinates]

        draw.line(xy=coordinates, fill=TRAJECTORY_COLOR, width=0)

        # Save image to disk
        image.save(file, 'PNG')
