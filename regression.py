"""Module contains abstract class Regrission"""

from abc import ABC, abstractmethod

import numpy as np


class Regression(ABC):
    """An abstract base Regression class."""
    def __init__(self, theta: np.ndarray = None):
        self._theta = theta
        self.var_x = None
        self.var_y = None

    @property
    def theta(self):
        """Return the parameters of the model."""
        return self._theta

    @theta.setter
    def theta(self, param: np.ndarray) -> None:
        self._theta = param

    @abstractmethod
    def predict(self, var_x: np.ndarray) -> float:
        """Return the prediction, given list of feature"""

    def fit(self, var_x: np.matrix, var_y: np.ndarray,
            precision=10**-12) -> np.ndarray:
        """Train the regression model.
        where var_x     = independent variables
              var_y     = dependent variabels
              precision = precision at which to stop gradient descent
        """
        # store data
        self.var_x = var_x
        self.var_y = var_y

        theta = self.theta
        if theta is None:
            theta = np.zeros(var_x.shape[1])

        ratio = 1   # assume a large initial learning ratio
        cost = self._cost(theta)

        while cost != 0:
            grad = self._gradient(theta)
            theta_n = theta - ratio * grad
            cost_n = self._cost(theta_n)

            # reduce learning ratio if mean square error increases (divergence)
            while cost_n > cost:
                ratio /= 2
                theta_n = theta - ratio * grad
                cost_n = self._cost(theta_n)

            # stop loop if precision is achieved
            if (cost - cost_n) / cost < precision:
                break

            # prepare for the next loop
            theta = theta_n
            cost = cost_n

        # set the internal theta valua
        self.theta = theta
        return theta

    @abstractmethod
    def _cost(self, theta: np.ndarray) -> float:
        """Return the cost at location theta."""

    @abstractmethod
    def _gradient(self, theta: np.ndarray) -> np.ndarray:
        """Return the gradient at coefficient theta."""
        if self.var_x.shape[0] != self.var_y.size:
            raise IndexError(
                "Lengths of x (independent) and y (dependent) must be equal")
