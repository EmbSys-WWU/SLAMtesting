# Coverage-Driven SLAM Testing

This project contains a systematic test data generator with accompanying simulation environment for SLAM testing. It generates a partitioning of the input space based on the given equivalence class boundaries (to modify, see ``testsuite.py``) and automatically creates a test suite that covers all equivalence classes.

## Supported SLAM Algorithms and Features

This project is not specific to any SLAM algorithm. The only prerequisite for application is an adapter class that inherits from and implements the abstract methods of ``slam_algorithm.SLAMAlgorithm``. However, there are certain restrictions to what we can support.

### Map

We support only feature-based maps. For simplicity, all maps are squares of a given side length and all features are point locations.

### Motion Model

We support both odometry and velocity motion models. To switch from one to the other, switch around the methods for test case generation implemented in ``generate_testcase.py``. However, as the evaluation was done with the odometry motion model, it is better tested.

### Sensors

We support only algorithms based on range-and-bearing sensor information.

### Obstacle Correspondence

We recommend using a SLAM algorithm that preserves obstacle correspondences. An obstacle's ID is passed to the algorithm upon observation. If you want to use this project for an algorithm that does not preserve this ID, you must switch the error metric (see ``evaluation_metric.py``) in ``main.py``. Without obstacle correspondences, our implementation provides the Earth Mover's Distance (see below) as a default metric; however, this is less accurate and does not control for errors due to rotation and translation of your local coordinate system.

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
