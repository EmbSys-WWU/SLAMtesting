# Coverage-Driven SLAM Testing

This project contains a systematic test data generator with accompanying simulation environment for SLAM testing. It generates a partitioning of the input space based on the given equivalence class boundaries and automatically creates a test suite that covers all equivalence classes.

## Supported SLAM Algorithms and Features

This project is not specific to any SLAM algorithm. The only prerequisite for application is an adapter class that inherits from and implements the abstract methods of ``slam_algorithm.SLAMAlgorithm``. However, there are certain restrictions to what we can support.

#### Map

We support only feature-based maps. For simplicity, all maps are squares of a given side length and all features are point locations.

#### Motion Model

We support both odometry and velocity motion models. To switch from one to the other, switch around the methods for test case generation implemented in ``generate_testcase.py``. However, as the evaluation was done with the odometry motion model, it is better tested.

#### Sensors

We support only algorithms based on range-and-bearing sensor information.

#### Obstacle Correspondence

We recommend using a SLAM algorithm that preserves obstacle correspondences. An obstacle's ID is passed to the algorithm upon observation. If you want to use this project for an algorithm that does not preserve this ID, you must switch the error metric (see ``evaluation_metric.py``) in ``main.py``. Without obstacle correspondences, our implementation provides the Earth Mover's Distance (see below) as a default metric; however, this is less accurate and does not control for errors due to rotation and translation of your local coordinate system.

## Parameter Adjustment

The test generator is based on the parameter definitions laid out in `testsuite.py`. Each parameter declares three constants that are relevant for equivalence class partitioning. These are the fields `<parameter>_LOWER_BOUND`, `<parameter>_UPPER_BOUND` and the list `<parameter>_BOUNDARIES`. The domain of the parameter is characterized as the interval between the lower and upper bounds. The values from the `<parameter>_BOUNDARIES` list signal boundaries between equivalence classes. From these values, equivalence classes are automatically generated. To adjust the equivalence classes, simply edit the relevant constants.

For three parameters, namely directionality, rotation error and inactivity, the equivalence classes are fixed by the nature of the parameter. If needed, these need to be edited manually. For the parameters relating to measurement or odometry errors, namely variance, bias and outlier probability, the equivalence class {0} is automatically added to the equivalence classes and needs to be removed if not desired.

## Experimental Results and Replication

The evaluation of this test approach was conducted using PythonRobotics' FastSLAM 1 implementation (located in the `slam` package). The evaluation run is currently configured (in `main.main()`) to run three times on three different test suites. The generator tries to read each suite from disk or, if it finds none, generates a new suite and writes it to disk using Python's `pickle` library. It then executes the evaluation run using the test suites and writes the results to `execution_results/results.txt`. Currently, this file still contains the raw data from our experiments. If you redo the experiments, you may want to delete the previous results first, otherwise the new results will be appended to the end of the file.

If you wish to replicate our experiments, you can use the `.pkl` files provided in the `test_suites` folder. These contain the test suites we used for our experiments. If you leave them in the folder, the generator will recognize and use them for the evaluation. Keep in mind that there is some nondeterminism in the SLAM algorithm, which means that different runs with the same test suite will not produce the exact same results. If you wish to try another experiment, remove the `.pkl` files from their folder and run the program to generate and evaluate new test suites. Visualizations of each test suite generated according to our method will be generated in the `test_cases` folder. For us, each evaluation run took approximately 8 hours.

## Software Used

The following software from other developers has been used in this project.

##### Kabsch Metric
Kromann: https://github.com/charnley/rmsd <br>
(BSD-2 license)

##### Earth Mover's Distance
Rubner: https://robotics.stanford.edu/~rubner/emd/default.htm <br>
(Modifications by paper authors marked)

##### Sample SLAM Implementation
Sakai et al.: https://github.com/AtsushiSakai/PythonRobotics/blob/master/SLAM/FastSLAM1/fast_slam1.py <br>
(MIT license, Modifications by paper authors marked)
