from typing import List, Callable

from random import uniform, randint, random, choice
from multiprocessing import Array, Process

from equivalence_class import EquivalenceClass
from generate_testcase import generate_testcase, SENSOR_RANGE
from map import Map
from testcase import TestCase
from slam_algorithm import SLAMAlgorithm


PATH_LENGTH = 1000
DIST_TO_ANGLE = 0.002           # Since the units for angles and distances are not equivalent, use this factor to
                                # convert between them (e.g. for errors in distances/angles)

########################################################################################################################
#                                            EQUIVALENCE CLASS DEFINITIONS                                             #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# The equivalence classes for these parameters depend on precise values and are left to the user to define.
# ----------------------------------------------------------------------------------------------------------------------

# ========== MAP SIZE ==========
MAP_SIZE_LOWER_BOUND = 2
MAP_SIZE_UPPER_BOUND = 20
MAP_SIZE_BOUNDARIES = [4, 10]

MAP_SIZE_DEFAULT = 8

# Generating equivalence classes
MAP_SIZE_EQUIVALENCE_CLASSES = []
low = MAP_SIZE_LOWER_BOUND
for b in MAP_SIZE_BOUNDARIES:
    MAP_SIZE_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, b))
    low = b
MAP_SIZE_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, MAP_SIZE_UPPER_BOUND))

# ========== LANDMARK DENSITY ==========
DENSITY_LOWER_BOUND = 1.0
DENSITY_UPPER_BOUND = 15.0
DENSITY_BOUNDARIES = [3.0, 9.0]

DENSITY_DEFAULT = 5.0

# Generating equivalence classes
DENSITY_EQUIVALENCE_CLASSES = []
low = DENSITY_LOWER_BOUND
for b in DENSITY_BOUNDARIES:
    DENSITY_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, b))
    low = b
DENSITY_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, DENSITY_UPPER_BOUND))

# ========== STEP LENGTH ==========
STEP_LENGTH_LOWER_BOUND = 250
STEP_LENGTH_UPPER_BOUND = SENSOR_RANGE      # Step length must not exceed sensor range
STEP_LENGTH_BOUNDARIES = [500, 1000]

STEP_LENGTH_DEFAULT = 800

# Generating equivalence classes
STEP_LENGTH_EQUIVALENCE_CLASSES = []
low = STEP_LENGTH_LOWER_BOUND
for b in STEP_LENGTH_BOUNDARIES:
    STEP_LENGTH_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, b))
    low = b
STEP_LENGTH_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, STEP_LENGTH_UPPER_BOUND))

# ========== SYMMETRY ==========
SYMMETRY_LOWER_BOUND = 0.0
SYMMETRY_UPPER_BOUND = 0.95                 # 100% symmetry should be excluded
SYMMETRY_BOUNDARIES = [0.1, 0.6]

SYMMETRY_DEFAULT = 0.0

# Generating equivalence classes
SYMMETRY_EQUIVALENCE_CLASSES = []
low = SYMMETRY_LOWER_BOUND
for b in SYMMETRY_BOUNDARIES:
    SYMMETRY_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, b))
    low = b
SYMMETRY_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, SYMMETRY_UPPER_BOUND))

# ----------------------------------------------------------------------------------------------------------------------
# The following factors of uncertainty should all include their own equivalence class for the case that error = 0.
# ----------------------------------------------------------------------------------------------------------------------

# ========== VARIANCE ==========
VARIANCE_LOWER_BOUND = 0.0
VARIANCE_UPPER_BOUND = 100.0
VARIANCE_BOUNDARIES = [20.0]

VARIANCE_DEFAULT = 10.0

# Generating equivalence classes
VARIANCE_EQUIVALENCE_CLASSES = [EquivalenceClass(0.0, 0.0)]
low = VARIANCE_LOWER_BOUND
for b in VARIANCE_BOUNDARIES:
    VARIANCE_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, b))
    low = b
VARIANCE_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, VARIANCE_UPPER_BOUND))

# ========== BIAS ==========
BIAS_LOWER_BOUND = -25.0
BIAS_UPPER_BOUND = 25.0
BIAS_BOUNDARIES = [-10.0, 0.0, 10.0]

BIAS_DEFAULT = 0.0

# Generating equivalence classes
BIAS_EQUIVALENCE_CLASSES = [EquivalenceClass(0.0, 0.0)]
low = BIAS_LOWER_BOUND
for b in BIAS_BOUNDARIES:
    BIAS_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, b))
    low = b
BIAS_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, BIAS_UPPER_BOUND))

# ========== OUTLIER PROBABILITY ==========
OUTLIERS_LOWER_BOUND = 0.0
OUTLIERS_UPPER_BOUND = 0.15
OUTLIERS_BOUNDARIES = [0.05]

OUTLIERS_DEFAULT = 0.02

# Generating equivalence classes
OUTLIERS_EQUIVALENCE_CLASSES = [EquivalenceClass(0.0, 0.0)]
low = OUTLIERS_LOWER_BOUND
for b in OUTLIERS_BOUNDARIES:
    OUTLIERS_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, b))
    low = b
OUTLIERS_EQUIVALENCE_CLASSES.append(EquivalenceClass(low, OUTLIERS_UPPER_BOUND))

# ----------------------------------------------------------------------------------------------------------------------
# The following parameters have obvious inherent equivalence classes that do not depend on exact values.
# ----------------------------------------------------------------------------------------------------------------------

# ========== INACTIVITY ==========
# No = 0 steps, yes = 2000 steps
INACTIVITY_DEFAULT = 0

# Generating equivalence classes
INACTIVITY_EQUIVALENCE_CLASSES = [EquivalenceClass(0, 0), EquivalenceClass(2000, 2000)]

# ========== ROTATIONAL ERROR ==========
# None = 0, underflow = -1, overflow = 1
ROTATION_DEFAULT = 0

# Generating equivalence classes
ROTATION_EQUIVALENCE_CLASSES = [EquivalenceClass(-1, -1), EquivalenceClass(0, 0), EquivalenceClass(1, 1)]

# ========== DIRECTIONALITY ==========
# random = 0 (False), directional = 1 (True)
DIRECTIONALITY_DEFAULT = 0

# Generating equivalence classes
DIRECTIONALITY_EQUIVALENCE_CLASSES = [EquivalenceClass(0, 0), EquivalenceClass(1, 1)]

########################################################################################################################
########################################################################################################################
########################################################################################################################


def parallel_execute(index: int,
                     results: Array,
                     function: Callable[[Callable[[int], SLAMAlgorithm], Callable[[Map, Map], float], str], float],
                     slam_factory: Callable[[int], SLAMAlgorithm],
                     metric: Callable[[Map, Map], float],
                     save_file_name: str) -> None:
    """
    Helper function for parallel testcase execution

    :param index: Process index
    :param results: Shared array to write results to
    :param function: Evaluation function
    :param slam_factory: <i> Parameter for evaluation function </i>
    :param metric: <i> Parameter for evaluation function </i>
    :param save_file_name: <i> Parameter for evaluation function </i>
    """

    try:
        results[index] = function(slam_factory, metric, save_file_name)
    except Exception as e:  # if a testcase fails catastrophically, count it as failure
        print(str(e))
        results[index] = -1


class TestSuite:

    def __init__(self):
        self.testcases: List[TestCase] = []
        self.type: str = ""
        self.number_of_testcases: int = max(map(len, [MAP_SIZE_EQUIVALENCE_CLASSES,
                                                      DENSITY_EQUIVALENCE_CLASSES,
                                                      STEP_LENGTH_EQUIVALENCE_CLASSES,
                                                      SYMMETRY_EQUIVALENCE_CLASSES,
                                                      VARIANCE_EQUIVALENCE_CLASSES,
                                                      BIAS_EQUIVALENCE_CLASSES,
                                                      OUTLIERS_EQUIVALENCE_CLASSES,
                                                      INACTIVITY_EQUIVALENCE_CLASSES,
                                                      ROTATION_EQUIVALENCE_CLASSES,
                                                      DIRECTIONALITY_EQUIVALENCE_CLASSES]))

    def generate_testcases(self) -> None:
        """
        Generates a systematic test suite covering every equivalence class defined above
        """

        self.type = "tested"

        map_sizes = MAP_SIZE_EQUIVALENCE_CLASSES.copy()
        densities = DENSITY_EQUIVALENCE_CLASSES.copy()
        step_lengths = STEP_LENGTH_EQUIVALENCE_CLASSES.copy()
        symmetries = SYMMETRY_EQUIVALENCE_CLASSES.copy()
        variances = VARIANCE_EQUIVALENCE_CLASSES.copy()
        biases = BIAS_EQUIVALENCE_CLASSES.copy()
        outliers = OUTLIERS_EQUIVALENCE_CLASSES.copy()
        inactivities = INACTIVITY_EQUIVALENCE_CLASSES.copy()
        rotations = ROTATION_EQUIVALENCE_CLASSES.copy()
        directions = DIRECTIONALITY_EQUIVALENCE_CLASSES.copy()

        for i in range(self.number_of_testcases):

            map_size = choice(map_sizes)
            map_sizes.remove(map_size)
            if not map_sizes:
                map_sizes = MAP_SIZE_EQUIVALENCE_CLASSES.copy()

            density = choice(densities)
            densities.remove(density)
            if not densities:
                densities = DENSITY_EQUIVALENCE_CLASSES.copy()

            step_length = choice(step_lengths)
            step_lengths.remove(step_length)
            if not step_lengths:
                step_lengths = STEP_LENGTH_EQUIVALENCE_CLASSES.copy()

            symmetry = choice(symmetries)
            symmetries.remove(symmetry)
            if not symmetries:
                symmetries = SYMMETRY_EQUIVALENCE_CLASSES.copy()

            variance = choice(variances)
            variances.remove(variance)
            if not variances:
                variances = VARIANCE_EQUIVALENCE_CLASSES.copy()

            bias = choice(biases)
            biases.remove(bias)
            if not biases:
                biases = BIAS_EQUIVALENCE_CLASSES.copy()

            outlier_probability = choice(outliers)
            outliers.remove(outlier_probability)
            if not outliers:
                outliers = OUTLIERS_EQUIVALENCE_CLASSES.copy()

            inactivity = choice(inactivities)
            inactivities.remove(inactivity)
            if not inactivities:
                inactivities = INACTIVITY_EQUIVALENCE_CLASSES.copy()

            rotation = choice(rotations)
            rotations.remove(rotation)
            if not rotations:
                rotations = ROTATION_EQUIVALENCE_CLASSES.copy()

            directionality = choice(directions)
            directions.remove(directionality)
            if not directions:
                directions = DIRECTIONALITY_EQUIVALENCE_CLASSES.copy()

            distance_variance = variance.select()
            angle_variance = distance_variance * DIST_TO_ANGLE
            distance_bias = bias.select()
            angle_bias = distance_bias * DIST_TO_ANGLE

            self.testcases.append(generate_testcase(path_length=PATH_LENGTH,
                                                    map_size=map_size.select(),
                                                    map_density=density.select(),
                                                    sensor_angle_variance=angle_variance,
                                                    sensor_angle_bias=angle_bias,
                                                    sensor_distance_variance=distance_variance,
                                                    sensor_distance_bias=distance_bias,
                                                    odometry_position_variance=distance_variance,
                                                    odometry_heading_bias=angle_bias,
                                                    odometry_heading_variance=angle_variance,
                                                    step_length=step_length.select(),
                                                    outlier_probability=outlier_probability.select(),
                                                    rotational_error=rotation.select(),
                                                    add_inactivity=inactivity.select(),
                                                    symmetry=symmetry.select(),
                                                    directional_traverse=bool(directionality.select())))

    def generate_random(self):
        """
        Generates a random test suite using randomized parameter values
        """

        self.type = "random"
        for i in range(self.number_of_testcases):
            distance_variance = uniform(VARIANCE_LOWER_BOUND, VARIANCE_UPPER_BOUND)
            angle_variance = distance_variance * DIST_TO_ANGLE
            distance_bias = uniform(BIAS_LOWER_BOUND, BIAS_UPPER_BOUND)
            angle_bias = distance_bias * DIST_TO_ANGLE
            self.testcases.append(generate_testcase(path_length=PATH_LENGTH,
                                                    map_size=randint(MAP_SIZE_LOWER_BOUND,
                                                                     MAP_SIZE_UPPER_BOUND),
                                                    map_density=uniform(DENSITY_LOWER_BOUND,
                                                                        DENSITY_UPPER_BOUND),
                                                    sensor_angle_variance=angle_variance,
                                                    sensor_angle_bias=angle_bias,
                                                    sensor_distance_variance=distance_variance,
                                                    sensor_distance_bias=distance_bias,
                                                    odometry_position_variance=distance_variance,
                                                    odometry_heading_bias=angle_bias,
                                                    odometry_heading_variance=angle_variance,
                                                    step_length=randint(STEP_LENGTH_LOWER_BOUND,
                                                                        STEP_LENGTH_UPPER_BOUND),
                                                    outlier_probability=uniform(OUTLIERS_LOWER_BOUND,
                                                                                OUTLIERS_UPPER_BOUND),
                                                    rotational_error=choice(ROTATION_EQUIVALENCE_CLASSES).select(),
                                                    add_inactivity=choice(INACTIVITY_EQUIVALENCE_CLASSES).select(),
                                                    symmetry=uniform(SYMMETRY_LOWER_BOUND,
                                                                     SYMMETRY_UPPER_BOUND),
                                                    directional_traverse=(random() < 0.5)))

    def generate_fixed(self):
        """
        Generates a random test suite using fixed default parameter values
        """

        self.type = "fixed"
        for i in range(self.number_of_testcases):
            self.testcases.append(generate_testcase(path_length=PATH_LENGTH,
                                                    map_size=MAP_SIZE_DEFAULT,
                                                    map_density=DENSITY_DEFAULT,
                                                    sensor_angle_variance=VARIANCE_DEFAULT * DIST_TO_ANGLE,
                                                    sensor_angle_bias=BIAS_DEFAULT * DIST_TO_ANGLE,
                                                    sensor_distance_variance=VARIANCE_DEFAULT,
                                                    sensor_distance_bias=BIAS_DEFAULT,
                                                    odometry_position_variance=VARIANCE_DEFAULT,
                                                    odometry_heading_bias=BIAS_DEFAULT * DIST_TO_ANGLE,
                                                    odometry_heading_variance=VARIANCE_DEFAULT * DIST_TO_ANGLE,
                                                    step_length=STEP_LENGTH_DEFAULT,
                                                    outlier_probability=OUTLIERS_DEFAULT,
                                                    rotational_error=ROTATION_DEFAULT,
                                                    add_inactivity=INACTIVITY_DEFAULT,
                                                    symmetry=SYMMETRY_DEFAULT,
                                                    directional_traverse=bool(DIRECTIONALITY_DEFAULT)))

    def execute(self, slam_factory: Callable[[int], SLAMAlgorithm], metric: Callable[[Map, Map], float]) -> List[float]:
        """
        Executes the test suite on a given algorithm

        :param slam_factory: Function that sets up the algorithm for the test
        :param metric: Evaluation metric for the result of the algorithm
        :return: Number of passed test cases
        """

        results = Array('d', self.number_of_testcases, lock=False)
        processes: List[Process] = []

        for i, case in enumerate(self.testcases):
            processes.append(Process(target=parallel_execute,
                                     args=(i,
                                           results,
                                           case.execute,
                                           slam_factory,
                                           metric,
                                           f"execution_results/testcase_{self.type}_{i}.png")))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return list(results)
