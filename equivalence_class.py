from random import randint, uniform
from typing import Union


class EquivalenceClass:
    def __init__(self, from_value: Union[int, float], to_value: Union[int, float]):
        if type(from_value) != type(to_value):
            raise Exception("Boundaries must be of same type!")

        self.lower = from_value
        self.upper = to_value

    def select(self) -> Union[int, float]:
        """
        Randomly selects an element from the equivalence class. Its type corresponds to the type of the boundary values
        :return: A random value of the equivalence class
        """

        if isinstance(self.lower, int):
            return randint(self.lower, self.upper)
        else:
            return uniform(self.lower, self.upper)
