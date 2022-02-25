from math import sqrt, atan2, hypot, sin, cos, pi
from random import uniform, gauss
from typing import Optional, Tuple

import numpy as np

from map import Map
from motion import Velocity, Odometry
from obstacle import Obstacle, ObstacleID
from robot import Robot
from sensor_trace import SensorTrace, TimeStep
from simulator import Simulator
from testcase import TestCase


DISTANCE_RATIO = 0.15   # Factor by which to decrease the threshold distance to reject obstacles too close to each other
SENSOR_RANGE = 2000     # Range of the sensor


def generate_testcase(path_length: int,
                      map_size: int,
                      map_density: float,
                      sensor_angle_variance: float,
                      sensor_angle_bias: float,
                      sensor_distance_variance: float,
                      sensor_distance_bias: float,
                      odometry_position_variance: float,
                      odometry_heading_bias: float,
                      odometry_heading_variance: float,
                      step_length: int,
                      outlier_probability: float,
                      rotational_error: int,
                      add_inactivity: int,
                      symmetry: float,
                      directional_traverse: bool) -> TestCase:
    """
    Automatically generates a test case based on the given parameters

    :param path_length: Total number of steps in the test case
    :param map_size: Total side length of the (quadratic) map
    :param map_density: Number of landmarks per unit area
    :param sensor_angle_variance: Variance in sensor angle measurements (in degrees)
    :param sensor_angle_bias: Bias in sensor angle measurements (counter-clockwise rotation; in degrees)
    :param sensor_distance_variance: Variance in sensor distance measurements (in mm)
    :param sensor_distance_bias: Bias in sensor distance measurements (in mm)
    :param odometry_position_variance: Uncertainty in both x- and y-direction after movement
                                       (assumed constant, independent and axis-aligned for simplicity)
    :param odometry_heading_bias: Bias in angular heading after movement
    :param odometry_heading_variance: Variance in angular heading after movement
    :param step_length: Length of a step in the path
    :param outlier_probability: The probability that a given measurement is an outlier
    :param rotational_error: Multiple of 360° to be added to angular measurements and odometry
    :param add_inactivity: Number of initial runs without any information, to check resistance to inactivity
    :param symmetry: Fraction of landmarks that are to be placed in a (point) symmetrical manner
    :param directional_traverse: Determines whether the robot's trajectory should be unidirectional or random
    :return: A test case following the specifications
    """

    print('Generating testcase')

    test_map = generate_map(map_size,
                            map_density,
                            symmetry)

    sensor_trace = generate_odometry_sensor_trace(test_map,
                                                  path_length,
                                                  sensor_angle_variance,
                                                  sensor_angle_bias,
                                                  sensor_distance_variance,
                                                  sensor_distance_bias,
                                                  odometry_position_variance,
                                                  odometry_heading_bias,
                                                  odometry_heading_variance,
                                                  step_length,
                                                  outlier_probability,
                                                  rotational_error,
                                                  add_inactivity,
                                                  directional_traverse)

    return TestCase(test_map, sensor_trace)


def generate_map(map_size: int,
                 map_density: float,
                 symmetry: float) -> Map:
    """
    Automatically generates a ground truth map with the given specifications

    :param map_size: Side length in meters of the (quadratic) map
    :param map_density: Number of landmarks per square meter
    :param symmetry: Fraction of landmarks that are to be placed in a (point) symmetrical fashion
    :return: A map with the given properties
    """

    # Convert to millimeters to place the obstacles
    x_lower_bound = map_size * -500
    x_upper_bound = map_size * 500
    y_lower_bound = map_size * -500
    y_upper_bound = map_size * 500

    number_of_landmarks = int(map_size * map_size * map_density)   # round down to avoid collision with minimum distance
    symmetrical_landmarks = int(round(symmetry * number_of_landmarks))

    # Average (minimum) distance between obstacles
    minimum_distance = 1000 * DISTANCE_RATIO/sqrt(map_density)

    m = Map(map_size)

    landmarks_placed = 0

    # First place the symmetrical portion of landmarks
    while landmarks_placed < symmetrical_landmarks:
        x = int(round(uniform(x_lower_bound, x_upper_bound)))
        y = int(round(uniform(y_lower_bound, y_upper_bound)))

        if 2 * hypot(x, y) >= minimum_distance:
            o1 = Obstacle(ObstacleID(landmarks_placed), x, y)
            o2 = Obstacle(ObstacleID(landmarks_placed + 1), -x, -y)     # mirrored through (0, 0)

            if m.fits(o1, minimum_distance) and m.fits(o2, minimum_distance):
                m.add_obstacle(o1)
                m.add_obstacle(o2)
                landmarks_placed += 2

    # Then fill in the rest non-symmetrically
    while landmarks_placed < number_of_landmarks:

        x = int(round(uniform(x_lower_bound, x_upper_bound)))
        y = int(round(uniform(y_lower_bound, y_upper_bound)))

        o = Obstacle(ObstacleID(landmarks_placed), x, y)

        if m.fits(o, minimum_distance):
            m.add_obstacle(o)
            landmarks_placed += 1

    m.done()

    return m


def generate_odometry_sensor_trace(test_map: Map,
                                   path_length: int,
                                   sensor_angle_variance: float,
                                   sensor_angle_bias: float,
                                   sensor_distance_variance: float,
                                   sensor_distance_bias: float,
                                   odometry_position_variance: float,
                                   odometry_heading_bias: float,
                                   odometry_heading_variance: float,
                                   step_length: int,
                                   outlier_probability: float,
                                   rotational_error: int,
                                   add_inactivity: int,
                                   directional_traverse: bool) -> SensorTrace:
    """
    Automatically generates a path through the given map and a corresponding sensor trace

    :param test_map: Ground truth map
    :param path_length: Total number of steps in the test case
    :param sensor_angle_variance: Variance in sensor angle measurements (in degrees)
    :param sensor_angle_bias: Bias in sensor angle measurements (counter-clockwise rotation; in degrees)
    :param sensor_distance_variance: Variance in sensor distance measurements (in mm)
    :param sensor_distance_bias: Bias in sensor distance measurements (in mm)
    :param odometry_position_variance: Uncertainty in both x- and y-direction after movement
                                       (assumed constant, independent and axis-aligned for simplicity)
    :param odometry_heading_bias: Bias in angular heading after movement
    :param odometry_heading_variance: Variance in angular heading after movement
    :param step_length: Length of a step in the path
    :param outlier_probability: The probability that a given measurement is an outlier
    :param rotational_error: Multiple of 360° to be added to angular measurements and odometry
    :param add_inactivity: Number of initial runs without any information, to check resistance to inactivity
    :param directional_traverse: Determines whether the robot's trajectory should be unidirectional or random
    :return: The sensor trace corresponding to a path through the given map with the given properties
    """

    sim = Simulator(test_map)

    robot = Robot(0, 0, 0)

    trace = SensorTrace(np.array([robot.x, robot.y, robot.rot]))

    # Add an observation run before movement
    observations = sim.simulate(robot,
                                SENSOR_RANGE,
                                sensor_distance_bias,
                                sensor_distance_variance,
                                sensor_angle_bias,
                                sensor_angle_variance,
                                outlier_probability,
                                rotational_error)
    test_map.visit([o.obstacle for o in observations])
    trace.add_step(TimeStep(Odometry(0, 0, 0), observations), robot.x, robot.y)

    print("Start generation")

    # Then add the real run
    current_destination: Optional[Obstacle] = None
    i = 0
    for i in range(int((path_length - 1) / 2)):
        # Find a destination to head to (an obstacle that has not yet been observed)
        if test_map.visited(current_destination):
            if directional_traverse:
                current_destination = test_map.get_next_destination()
            else:
                current_destination = test_map.get_random_destination()

        add_odometry_step(test_map,
                          current_destination,
                          robot,
                          trace,
                          sim,
                          step_length,
                          sensor_angle_variance,
                          sensor_angle_bias,
                          sensor_distance_variance,
                          sensor_distance_bias,
                          odometry_position_variance,
                          odometry_heading_bias,
                          odometry_heading_variance,
                          outlier_probability,
                          rotational_error)

    # Stop after half the run to add a period of inactivity
    index = i

    print("Add inactivity")

    # Add inactivity, if desired
    for i in range(add_inactivity):
        time_step = TimeStep(Odometry(0, 0, 0), [])
        trace.add_step(time_step, robot.x, robot.y)

    for i in range(index, path_length - 1):
        # Find a destination to head to (an obstacle that has not yet been observed)
        if test_map.visited(current_destination):
            if directional_traverse:
                current_destination = test_map.get_next_destination()
            else:
                current_destination = test_map.get_random_destination()

        add_odometry_step(test_map,
                          current_destination,
                          robot,
                          trace,
                          sim,
                          step_length,
                          sensor_angle_variance,
                          sensor_angle_bias,
                          sensor_distance_variance,
                          sensor_distance_bias,
                          odometry_position_variance,
                          odometry_heading_bias,
                          odometry_heading_variance,
                          outlier_probability,
                          rotational_error)

    print("Done")

    return trace


def add_odometry_step(test_map: Map,
                      current_destination: Obstacle,
                      robot: Robot,
                      trace: SensorTrace,
                      sim: Simulator,
                      step_length: int,
                      sensor_angle_variance: float,
                      sensor_angle_bias: float,
                      sensor_distance_variance: float,
                      sensor_distance_bias: float,
                      odometry_position_variance: float,
                      odometry_heading_bias: float,
                      odometry_heading_variance: float,
                      outlier_probability: float,
                      rotational_error: int) -> None:
    """
    Adds one time step on the given map to the given sensor trace

    :param test_map: Ground map
    :param current_destination: Destination to drive towards
    :param robot: Vehicle to move
    :param trace: Sensor trace to add to
    :param sim: Simulation object for generating observations
    :param step_length: Length of next movement
    :param sensor_angle_variance: Variance in sensor angle measurements (in degrees)
    :param sensor_angle_bias: Bias in sensor angle measurements (counter-clockwise rotation; in degrees)
    :param sensor_distance_variance: Variance in sensor distance measurements (in mm)
    :param sensor_distance_bias: Bias in sensor distance measurements (in mm)
    :param odometry_position_variance: Uncertainty in both x- and y-direction after movement
                                       (assumed constant, independent and axis-aligned for simplicity)
    :param odometry_heading_bias: Bias in angular heading after movement
    :param odometry_heading_variance: Variance in angular heading after movement
    :param outlier_probability: Probability of an outlier of some sort
    :param rotational_error: Determines how often 2*pi should be added to angles
    """

    # This is not the angle straight to the obstacle, since the robot moves forward first, but it will eventually
    # converge
    angle_to_destination = atan2(current_destination.y - robot.y, current_destination.x - robot.x) - robot.rot
    measured_forward_movement = step_length
    measured_rotational_movement = (angle_to_destination + pi) % (2 * pi) - pi

    # Add noise to (actual) robot path
    noisy_forward_movement = gauss(measured_forward_movement,
                                   odometry_position_variance)
    noisy_rotational_movement = gauss(measured_rotational_movement + odometry_heading_bias,
                                      odometry_heading_variance)

    # Create odometry information
    odometry = Odometry(0, measured_forward_movement, measured_rotational_movement + rotational_error * 2 * pi)

    # Move robot to new location
    robot.move_by_odometry(noisy_forward_movement, noisy_rotational_movement)

    # Simulate noisy measurements
    observations = sim.simulate(robot,
                                SENSOR_RANGE,
                                sensor_distance_bias,
                                sensor_distance_variance,
                                sensor_angle_bias,
                                sensor_angle_variance,
                                outlier_probability,
                                rotational_error)

    # Mark visited obstacles as visited
    test_map.visit([o.obstacle for o in observations])

    # Create sensor TimeStep
    time_step = TimeStep(odometry, observations)

    # Add to SensorTrace
    trace.add_step(time_step, robot.x, robot.y)


def generate_velocity_sensor_trace(test_map: Map,
                                   path_length: int,
                                   sensor_angle_variance: float,
                                   sensor_angle_bias: float,
                                   sensor_distance_variance: float,
                                   sensor_distance_bias: float,
                                   odometry_position_variance: float,
                                   odometry_heading_bias: float,
                                   odometry_heading_variance: float,
                                   step_length: int,
                                   outlier_probability: float,
                                   rotational_error: int,
                                   add_inactivity: int,
                                   directional_traverse: bool) -> SensorTrace:
    """
    Automatically generates a path through the given map and a corresponding sensor trace

    :param test_map: Ground truth map
    :param path_length: Total number of steps in the test case
    :param sensor_angle_variance: Variance in sensor angle measurements (in degrees)
    :param sensor_angle_bias: Bias in sensor angle measurements (counter-clockwise rotation; in degrees)
    :param sensor_distance_variance: Variance in sensor distance measurements (in mm)
    :param sensor_distance_bias: Bias in sensor distance measurements (in mm)
    :param odometry_position_variance: Uncertainty in both x- and y-direction after movement
                                       (assumed constant, independent and axis-aligned for simplicity)
    :param odometry_heading_bias: Bias in angular heading after movement
    :param odometry_heading_variance: Variance in angular heading after movement
    :param step_length: Length of a step in the path
    :param outlier_probability: The probability that a given measurement is an outlier
    :param rotational_error: Multiple of 360° to be added to angular measurements and odometry
    :param add_inactivity: Number of initial runs without any information, to check resistance to inactivity
    :param directional_traverse: Determines whether the robot's trajectory should be unidirectional or random
    :return: The sensor trace corresponding to a path through the given map with the given properties
    """

    sim = Simulator(test_map)

    robot = Robot(0, 0, 0)

    trace = SensorTrace(np.array([robot.x, robot.y, robot.rot]))

    # Add an observation run before movement
    observations = sim.simulate(robot,
                                SENSOR_RANGE,
                                sensor_distance_bias,
                                sensor_distance_variance,
                                sensor_angle_bias,
                                sensor_angle_variance,
                                outlier_probability,
                                rotational_error)
    test_map.visit([o.obstacle for o in observations])
    trace.add_step(TimeStep(Velocity(0, 0), observations), robot.x, robot.y)

    # Then add the real run
    current_destination: Optional[Obstacle] = None
    forward_to_yaw_factor = 0
    angle = 0.0
    i = 0
    for i in range(int((path_length - 1) / 2)):
        # Find a destination to head to (an obstacle that has not yet been observed)
        if test_map.visited(current_destination):
            current_destination, forward_to_yaw_factor, angle = select_new_target(test_map, robot, directional_traverse)

        # If the robot points away from the obstacle, it turns around first
        if abs(angle) > pi / 2.0:
            measured_velocity = 0.0
            measured_yaw_velocity = pi
        else:
            # Assume dt = 1, so velocity is just equivalent to the step size
            measured_velocity = step_length
            measured_yaw_velocity = measured_velocity * forward_to_yaw_factor

        velocities = Velocity(measured_velocity, measured_yaw_velocity)

        # Generate noisy (actual) data
        noisy_velocity = gauss(measured_velocity, odometry_position_variance)
        noisy_yaw_velocity = gauss(measured_yaw_velocity + odometry_heading_bias, odometry_heading_variance)

        # Move robot
        move_robot_by(robot, noisy_velocity, noisy_yaw_velocity)

        # Simulate noisy measurements
        observations = sim.simulate(robot,
                                    SENSOR_RANGE,
                                    sensor_distance_bias,
                                    sensor_distance_variance,
                                    sensor_angle_bias,
                                    sensor_angle_variance,
                                    outlier_probability,
                                    rotational_error)

        # Mark visited obstacles as visited
        test_map.visit([o.obstacle for o in observations])

        # Create sensor TimeStep
        time_step = TimeStep(velocities, observations)

        # Add to SensorTrace
        trace.add_step(time_step, robot.x, robot.y)

    index = i

    # Then add a period of inactivity, if desired
    for i in range(add_inactivity):
        time_step = TimeStep(Velocity(0, 0), [])
        trace.add_step(time_step, robot.x, robot.y)

    # Then resume the run
    for i in range(index, path_length):
        # Find a destination to head to (an obstacle that has not yet been observed)
        if test_map.visited(current_destination):
            current_destination, forward_to_yaw_factor, angle = select_new_target(test_map, robot, directional_traverse)

        # If the robot points away from the obstacle, it turns around first
        if abs(angle) > pi / 2.0:
            measured_velocity = 0.0
            measured_yaw_velocity = pi
        else:
            # Assume dt = 1, so velocity is just equivalent to the step size
            measured_velocity = step_length
            measured_yaw_velocity = measured_velocity * forward_to_yaw_factor

        velocities = Velocity(measured_velocity, measured_yaw_velocity)

        # Generate noisy (actual) data
        noisy_velocity = gauss(measured_velocity, odometry_position_variance)
        noisy_yaw_velocity = gauss(measured_yaw_velocity + odometry_heading_bias, odometry_heading_variance)

        # Move robot
        move_robot_by(robot, noisy_velocity, noisy_yaw_velocity)

        # Simulate noisy measurements
        observations = sim.simulate(robot,
                                    SENSOR_RANGE,
                                    sensor_distance_bias,
                                    sensor_distance_variance,
                                    sensor_angle_bias,
                                    sensor_angle_variance,
                                    outlier_probability,
                                    rotational_error)

        # Mark visited obstacles as visited
        test_map.visit([o.obstacle for o in observations])

        # Create sensor TimeStep
        time_step = TimeStep(velocities, observations)

        # Add to SensorTrace
        trace.add_step(time_step, robot.x, robot.y)

    return trace


def select_new_target(test_map: Map, robot: Robot, directional: bool) -> Tuple[Obstacle, float, float]:
    """
    Selects a new target from the given map and calculates the ratio of forward to angular velocity necessary to reach
    it from the given robot position

    :param test_map: Map from which to choose a target
    :param robot: Current robot position in the map
    :param directional: Determines whether the next target should be selected in order or randomly
    :return: A tuple of (1) the obstacle that is the new target, (2) the desired ratio of forward and angular
             velocity and (3) the angle to the target
    """

    if directional:
        next_obstacle = test_map.get_next_destination()
    else:
        next_obstacle = test_map.get_random_destination()

    dx = next_obstacle.x - robot.x
    dy = next_obstacle.y - robot.y
    d = hypot(dx, dy)
    alpha = atan2(dy, dx) - robot.rot

    # Velocity factor needed to traverse twice the angle (since it is an arc, the angle difference at the end will be
    # -alpha) in the time it takes the robot to traverse the distance d, based on arc length geometry
    factor = 2 * sin(alpha) / d

    return next_obstacle, factor, alpha


def move_robot_by(robot: Robot, v: float, yaw_rate: float) -> None:
    """
    Moves the given robot to the position it will be after driving for t=1s with given forward and angular velocities

    :param robot: Robot to move
    :param v: Forward velocity
    :param yaw_rate: Angular velocity
    """

    turn_radius = abs(v / yaw_rate)
    distance = abs(2 * turn_radius * sin(yaw_rate / 2))    # since t=1s, alpha = yaw_rate
    angle = robot.rot + yaw_rate / 2

    dx = int(round(distance * cos(angle)))
    dy = int(round(distance * sin(angle)))

    robot.move_by(dx, dy, yaw_rate)
