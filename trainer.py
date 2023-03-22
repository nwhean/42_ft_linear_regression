"""This module implements the required functions for ft_linear_regression."""

import csv
from typing import Optional

import numpy as np

from regression import Regression


class LinearRegression(Regression):
    """Ordinary least squares Linear Regression."""
    def predict(self, var_x: np.ndarray | np.matrix,
                theta: Optional[np.ndarray] = None) -> float:
        """
        Return the predicted value, y based on linear regression model.
        Where
        y = theta0 + theta1 * var_x1 + theta2 * var_x2 + ... + thetan * var_xn
        """
        if theta is None:
            theta = self.theta

        if len(var_x.shape) == 1:
            return theta[0] + np.sum(theta[1:] * var_x)
        elif len(var_x.shape) == 2:
            return theta[0] + np.sum((var_x.T * theta[1:]).T, axis=1)
        else:
            return None

    def score(self) -> float:
        """Return the coefficient of determination of the prediction."""
        return 1 - self._residual_ss(self.theta) / residual_total(y)

    def _residual_ss(self, theta: np.ndarray) -> float:
        """Return the residual sum of squares."""
        pred = self.predict(self.var_x, theta)
        return np.sum((self.var_y - pred)**2)

    def _cost(self, theta: np.ndarray) -> float:
        """Return the mean squared error."""
        return self._residual_ss(theta) / self.var_y.size

    def _gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Return the gradient of linear regression model at the given coefficient c
        """
        super()._gradient(theta)
        count = len(self.var_y)
        pred = self.predict(self.var_x, theta)
        residual = self.var_y - pred
        grad = [np.sum(residual)]
        for j in range(self.var_x.shape[1]):
            grad.append(np.sum(residual * self.var_x[:, j]))
        grad = np.array(grad)
        return grad * -2 / count


def read_file(filename: str, x_name: str, y_name: str) -> tuple[list[float]]:
    """Read data from file."""
    var_x = []
    var_y = []
    with open(filename, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            var_x.append(float(row[x_name]))
            var_y.append(float(row[y_name]))
    return np.array(var_x), np.array(var_y)

def write_file(filename: str, theta: list[float]) -> None:
    """Write the regression parameters to a file."""
    with open(filename, 'w', newline='', encoding="utf-8") as file:
        fieldnames = ["theta" + str(i) for i in range(2)]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict(zip(fieldnames, theta)))

def normalise_array(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalise the input array to improve rate of convergence."""
    # calculate the normalisation parameters
    shift = data.min()
    scale = data.max() - shift
    return (data - shift) / scale, shift, scale

def normalise_matrix(data: np.matrix
        ) -> tuple[np.matrix, np.ndarray, np.ndarray]:
    """Normalise the input matrix to improve rate of convergence."""
    shift = data.min(axis=0)
    scale = data.max(axis=0) - shift
    data = (data.T - shift) / scale
    return data.T, shift, scale

def denormalise(theta: list[float], shift_x: np.ndarray, scale_x: np.ndarray,
                shift_y: float, scale_y: float) -> list[float]:
    """Denormalise coefficients based on given normalisation parameters."""
    retval = np.array([shift_y
              + scale_y * (theta[0] - np.sum((theta[1:] * shift_x) / scale_x))
              ])
    retval = np.append(retval, theta[1:] * scale_y / scale_x)
    return np.array(retval)

def mean(nums: np.ndarray) -> float:
    """Return the mean of a list of float."""
    return np.sum(nums) / nums.size

def residual_total(var_y: np.ndarray) -> float:
    """Return the total sum of squares."""
    y_mean = mean(var_y)
    return np.sum((var_y - y_mean)**2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    INFILE = "data.csv"
    OUTFILE = "coefficient.csv"

    print("Reading data...")
    km, price = read_file(INFILE, "km", "price")
    km = np.expand_dims(km, axis=1)
    x, x_shift, x_scale = normalise_matrix(km)
    y, y_shift, y_scale = normalise_array(price)

    print("Descending gradient...")
    model = LinearRegression()
    model.fit(x, y)
    coeff = denormalise(model.theta, x_shift, x_scale, y_shift, y_scale)

    write_file(OUTFILE, coeff)
    print(f"Coefficients written to {OUTFILE}.")

    # Calculate precision of regression
    for i, val in enumerate(coeff):
        print(f"theta{i} = {coeff[i]}")
    print(f"R_squared = {model.score()}")

    # plot graph
    plt.scatter(km, price)  # plot data as scatter plot
    model = LinearRegression(coeff)
    x0 = [min(km), max(km)]
    y0 = [model.predict(i) for i in x0]
    plt.plot(x0, y0)    # plot regression line

    # add title and labels
    plt.title("Regression Line")
    plt.xlabel("Mileage")
    plt.ylabel("Price")

    # add thousand separators to x and y axes
    current_values = plt.gca().get_xticks()
    plt.gca().xaxis.set_ticks(current_values)
    plt.gca().set_xticklabels([f'{x:,.0f}' for x in current_values])
    current_values = plt.gca().get_yticks()
    plt.gca().yaxis.set_ticks(current_values)
    plt.gca().set_yticklabels([f'{x:,.0f}' for x in current_values])

    plt.show()
