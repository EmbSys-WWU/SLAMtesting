import os
import pickle
import random
import time
from typing import List

import evaluation_metric
from slam import fastslam
from testsuite import TestSuite


FILE_NAME = 'test_suites/suites.pkl'


def generate_test_suites(f: str = None) -> List[TestSuite]:
    """
    Reads test suites from disk, if they have already been stored there. If not, generates new test suites and saves
    them to disk

    :return: A list of test suites in the format of [systematic suite, random suite, fixed suite]
    """

    if f is None:
        file_name = FILE_NAME
    else:
        file_name = f

    if os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            test_suites = pickle.load(file)

        return test_suites

    else:
        # Create test suites
        tested_suite = TestSuite()
        tested_suite.generate_testcases()

        random_suite = TestSuite()
        random_suite.generate_random()

        fixed_suite = TestSuite()
        fixed_suite.generate_fixed()

        test_suites = [tested_suite, random_suite, fixed_suite]

        # Save test suites to file
        with open(file_name, 'wb+') as file:
            pickle.dump(test_suites, file)

        return test_suites


def execute_test(tested_cases: TestSuite, random_cases: TestSuite, fixed_cases: TestSuite) -> None:
    """
    Executes the three given test suites and prints the results at the end

    :param tested_cases: Systematic test suite
    :param random_cases: Random test suite
    :param fixed_cases: Fixed test suite
    """

    start = time.perf_counter()

    tested_results_correct = tested_cases.execute(fastslam.slam_factory_correct,
                                                  evaluation_metric.evaluate_with_known_correspondences)
    average_error_tested_correct = sum(tested_results_correct) / len(tested_results_correct)

    random_results_correct = random_cases.execute(fastslam.slam_factory_correct,
                                                  evaluation_metric.evaluate_with_known_correspondences)
    average_error_random_correct = sum(random_results_correct) / len(random_results_correct)

    fixed_results_correct = fixed_cases.execute(fastslam.slam_factory_correct,
                                                evaluation_metric.evaluate_with_known_correspondences)
    average_error_fixed_correct = sum(fixed_results_correct) / len(fixed_results_correct)

    ####################################################################################################################

    tested_results_resampling = tested_cases.execute(fastslam.slam_factory_resampling,
                                                     evaluation_metric.evaluate_with_known_correspondences)
    average_error_tested_resampling = sum(tested_results_resampling) / len(tested_results_resampling)

    random_results_resampling = random_cases.execute(fastslam.slam_factory_resampling,
                                                     evaluation_metric.evaluate_with_known_correspondences)
    average_error_random_resampling = sum(random_results_resampling) / len(random_results_resampling)

    fixed_results_resampling = fixed_cases.execute(fastslam.slam_factory_resampling,
                                                   evaluation_metric.evaluate_with_known_correspondences)
    average_error_fixed_resampling = sum(fixed_results_resampling) / len(fixed_results_resampling)

    ####################################################################################################################

    tested_results_rotation = tested_cases.execute(fastslam.slam_factory_rotation,
                                                   evaluation_metric.evaluate_with_known_correspondences)
    average_error_tested_rotation = sum(tested_results_rotation) / len(tested_results_rotation)

    random_results_rotation = random_cases.execute(fastslam.slam_factory_rotation,
                                                   evaluation_metric.evaluate_with_known_correspondences)
    average_error_random_rotation = sum(random_results_rotation) / len(random_results_rotation)

    fixed_results_rotation = fixed_cases.execute(fastslam.slam_factory_rotation,
                                                 evaluation_metric.evaluate_with_known_correspondences)
    average_error_fixed_rotation = sum(fixed_results_rotation) / len(fixed_results_rotation)

    ####################################################################################################################

    tested_results_sign_error = tested_cases.execute(fastslam.slam_factory_sign_error,
                                                     evaluation_metric.evaluate_with_known_correspondences)
    average_error_tested_sign_error = sum(tested_results_sign_error) / len(tested_results_sign_error)

    random_results_sign_error = random_cases.execute(fastslam.slam_factory_sign_error,
                                                     evaluation_metric.evaluate_with_known_correspondences)
    average_error_random_sign_error = sum(random_results_sign_error) / len(random_results_sign_error)

    fixed_results_sign_error = fixed_cases.execute(fastslam.slam_factory_sign_error,
                                                   evaluation_metric.evaluate_with_known_correspondences)
    average_error_fixed_sign_error = sum(fixed_results_sign_error) / len(fixed_results_sign_error)

    end = time.perf_counter()

    print(f"""

=============================================================================================================
=============================================================================================================

Error results of the respective baseline evaluations:
Testcase: Results {tested_results_correct}, Average {average_error_tested_correct}.
Random:   Results {random_results_correct}, Average {average_error_random_correct}.
Fixed:    Results {fixed_results_correct}, Average {average_error_fixed_correct}.

Error results of the respective evaluations without resampling filter:
Testcase: Results {tested_results_resampling}, Average {average_error_tested_resampling}.
Random:   Results {random_results_resampling}, Average {average_error_random_resampling}.
Fixed:    Results {fixed_results_resampling}, Average {average_error_fixed_resampling}.

Error results of the respective evaluations without rotation filter:
Testcase: Results {tested_results_rotation}, Average {average_error_tested_rotation}.
Random:   Results {random_results_rotation}, Average {average_error_random_rotation}.
Fixed:    Results {fixed_results_rotation}, Average {average_error_fixed_rotation}.

Error result of the respective evaluations with sign error:
Testcase: Results {tested_results_sign_error}, Average {average_error_tested_sign_error}.
Random:   Results {random_results_sign_error}, Average {average_error_random_sign_error}.
Fixed:    Results {fixed_results_sign_error}, Average {average_error_fixed_sign_error}.

Total evaluation time was {end - start}s.

=============================================================================================================
=============================================================================================================

""")


def visualize_testcases(suite: TestSuite) -> None:
    """
    Visualizes all testcases from the given test suite

    :param suite: Test suite to visualize
    """

    for i, case in enumerate(suite.testcases):
        case.visualize(f"test_cases/case{i}.png")


def main() -> None:
    """
    Main program
    """

    random.seed()
    suites = generate_test_suites()
    visualize_testcases(suites[0])
    execute_test(suites[0], suites[1], suites[2])


def debug() -> None:
    """
    Auxiliary program to be used for validation of the simulation environment
    """

    random.seed()
    test_suite = TestSuite()
    print(test_suite.number_of_testcases)
    test_suite.generate_testcases()
    visualize_testcases(test_suite)


if __name__ == "__main__":
    main()
    # debug()
