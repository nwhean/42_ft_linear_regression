"""This module implements the required functions for ft_linear_regression."""

import csv
from typing import Optional


def read_file(filename: str, x_name: str, y_name: str) -> tuple[list[float]]:
    """Read data from file."""
    var_x = []
    var_y = []
    with open(filename, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            var_x.append(float(row[x_name]))
            var_y.append(float(row[y_name]))
    return var_x, var_y

def write_file(filename: str, theta: list[float]) -> None:
    """Write the regression parameters to a file."""
    with open(filename, 'w', newline='', encoding="utf-8") as file:
        fieldnames = ["theta" + str(i) for i in range(2)]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict(zip(fieldnames, theta)))

def normalise(var_x: list[float], var_y: list[float]) -> tuple[list[float]]:
    """Normalise the input data to improve rate of convergence."""
    # calculate the normalisation parameters
    min_x = min(var_x)
    max_x = max(var_x)
    range_x = max_x - min_x

    min_y = min(var_y)
    max_y = max(var_y)
    range_y = max_y - min_y

    x_norm = [(i - min_x) / range_x for i in var_x]
    y_norm = [(i - min_y) / range_y for i in var_y]

    return x_norm, y_norm, [min_x, range_x, min_y, range_y]

def denormalise(theta: list[float], norm_param: list[float]) -> list[float]:
    """Denormalise coefficients based on given normalisation parameters."""
    min_x, range_x, min_y, range_y = norm_param
    retval = []
    retval.append(min_y + range_y * (theta[0] - theta[1] * min_x / range_x))
    retval.append(theta[1] * range_y / range_x)
    return retval

def predict(theta: list[float], var_x: float) -> float:
    """
    Return the predicted value, y based on linear regression model.
    Where
    y = theta0 + theta1 * var_x
    """
    return theta[0] + theta[1] * var_x

def gradient(theta: list[float], var_x: list[float], var_y: list[float]
        ) -> list[float]:
    """
    Return the gradient of linear regression model at the given coefficient c
    """
    if len(var_x) != len(var_y):
        raise IndexError(
            "Lengths of x (independent) and y (dependent) must be equal")
    count = len(var_y)
    grad = []
    pred = [predict(theta, i) for i, j in zip(var_x, var_y)]
    grad.append(sum(i - j for i, j in zip(var_y, pred)))
    grad.append(sum((i - j) * k for i, j, k in zip(var_y, pred, var_x)))
    return [-2.0 / count * i for i in grad]

def train(var_x: list[float], var_y: list[float],
          theta: Optional[list[float]] = None,
          precision=10**-12) -> list[float]:
    """
    Given the initial coefficients 'theta', return the final coefficient based
    on gradient descent method.
    """
    if theta is None:
        theta = [0, 0]

    ratio = 1   # assume a large initial learning ratio
    mse = mean_squared_error(theta, var_x, var_y)

    while mse != 0:
        grad = gradient(theta, var_x, var_y)
        theta_n = [i - ratio * j for i, j in zip(theta, grad)]
        mse_n = mean_squared_error(theta_n, var_x, var_y)

        # reduce learning ratio if mean square error increases (divergence)
        while mse_n > mse:
            ratio /= 2
            theta_n = [i - ratio * j for i, j in zip(theta, grad)]
            mse_n = mean_squared_error(theta_n, var_x, var_y)

        # stop loop if precision is achieved
        if (mse - mse_n) / mse < precision:
            break

        # prepare for the next loop
        theta = theta_n
        mse = mse_n

    return theta

def mean(nums: list[float]) -> float:
    """Return the mean of a list of float."""
    return sum(nums) / len(nums)

def residual_ss(theta: list[float], var_x: list[float], var_y: list[float]
        ) -> float:
    """Return the residual sum of squares."""
    pred = [predict(theta, i) for i, j in zip(var_x, var_y)]
    return sum((i - j)**2 for i, j in zip(var_y, pred))

def mean_squared_error(theta: list[float], var_x: list[float],
                       var_y: list[float]) -> float:
    """Return the mean squared error."""
    return residual_ss(theta, var_x, var_y) / len(var_y)

def residual_total(var_y: list[float]) -> float:
    """Return the total sum of squares."""
    y_mean = mean(var_y)
    return sum((i - y_mean)**2 for i in var_y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    INFILE = "data.csv"
    OUTFILE = "coefficient.csv"

    print("Reading data...")
    km, price = read_file(INFILE, "km", "price")
    x, y, param = normalise(km, price)

    print("Descending gradient...")
    coeff_norm = train(x, y)
    coeff = denormalise(coeff_norm, param)

    write_file(OUTFILE, coeff)
    print(f"Coefficients written to {OUTFILE}.")

    # Calculate precision of regression
    for i, val in enumerate(coeff):
        print(f"theta{i} = {coeff[i]}")
    r2 = 1 - residual_ss(coeff, km, price) / residual_total(price)
    print(f"R_squared = {r2}")

    # plot graph
    plt.scatter(km, price)  # plot data as scatter plot
    x0 = [min(km), max(km)]
    y0 = [predict(coeff, i) for i in x0]
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
