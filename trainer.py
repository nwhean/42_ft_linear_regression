import csv
from typing import Optional

import matplotlib.pyplot as plt


def read_file(filename: str, x_name: str, y_name: str) -> tuple[list[float]]:
    x = []
    y = []
    with open('data.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row[x_name]))
            y.append(float(row[y_name]))
    return x, y

def predict(c: list[float], x: float) -> float:
    """
    Return the predicted value, y based on linear regression model.
    Where
    y = c0 + c1 x
    """
    return c[0] + c[1] * x

def gradient(c: list[float], x: list[float], y: list[float]) -> list[float]:
    """
    Return the gradient of linear regression model at the given coefficient c
    """
    if len(x) != len(y):
        raise IndexError(
            "Lengths of x (independent) and y (dependent) must be equal")
    n = len(y)
    grad = []
    pred = [predict(c, i) for i, j in zip(x, y)]
    grad.append(sum(i - j for i, j in zip(y, pred)))
    grad.append(sum((i - j) * k for i, j, k in zip(y, pred, x)))
    return [-2.0 / n * i for i in grad]

def train(x: list[float], y: list[float], alpha: Optional[list[float]] = None,
          precision=10**-12) -> list[float]:
    """
    Given the initial coefficients 'alpha', return the final coefficient based
    on gradient descent method.
    """
    if alpha is None:
        alpha = [0, 0]
    
    ratio = 1   # assume a large initial learning ratio
    mse = mean_squared_error(alpha, x, y)

    while mse != 0:
        grad = gradient(alpha, x, y)
        alpha_n = [i - ratio * j for i, j in zip(alpha, grad)]
        mse_n = mean_squared_error(alpha_n, x, y)
        
        # reduce learning ratio if mean square error increases (divergence)
        while mse_n > mse:
            ratio /= 2
            alpha_n = [i - ratio * j for i, j in zip(alpha, grad)]
            mse_n = mean_squared_error(alpha_n, x, y)
        
        # stop loop if precision is achieved
        if (mse - mse_n) / mse < precision:
            break
        
        # prepare for the next loop
        alpha = alpha_n
        mse = mse_n
        
    return alpha

def mean(nums: list[float]) -> float:
    """Return the mean of a list of float."""
    return sum(nums) / len(nums)

def residual_ss(c: list[float], x: list[float], y: list[float]) -> float:
    """Return the residual sum of squares."""
    pred = [predict(c, i) for i, j in zip(x, y)]
    return sum((i - j)**2 for i, j in zip(y, pred))

def mean_squared_error(c: list[float], x: list[float], y: list[float]) -> float:
    """Return the mean squared error."""
    return residual_ss(c, x, y) / len(y)

def residual_total(c: list[float], x: list[float], y: list[float]) -> float:
    """Return the total sum of squares."""
    y_mean = mean(y)
    pred = [predict(c, i) for i, j in zip(x, y)]
    return sum((i - y_mean)**2 for i in y)

if __name__ == "__main__":
    print("Reading data...")
    km, price = read_file("data.csv", "km", "price")

    # calculate the normalisation parameters
    min_x = min(km)
    max_x = max(km)
    range_x = max_x - min_x

    min_y = min(price)
    max_y = max(price)
    range_y = max_y - min_y

    x = [(i - min_x) / range_x for i in km]
    y = [(i - min_y) / range_y for i in price]

    print("Descending gradient...")
    alpha = train(x, y)
    coeff = []
    coeff.append(min_y + range_y * (alpha[0] - alpha[1] * min_x / range_x))
    coeff.append(alpha[1] * range_y / range_x)

    outfile = 'coefficient.csv'
    with open(outfile, 'w', newline='') as f:
        fieldnames = ["theta" + str(i) for i in range(2)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({i: j for i, j in zip(fieldnames, coeff)})
    
    for i, val in enumerate(coeff):
        print(f"theta{i} = {coeff[i]}")
    R2 = 1 - residual_ss(coeff, km, price) / residual_total(coeff, km, price)
    print(f"R_squared = {R2}")
    print(f"Coefficients written to {outfile}.")

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
    plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_values])
    current_values = plt.gca().get_yticks()
    plt.gca().yaxis.set_ticks(current_values)
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    
    plt.show()
