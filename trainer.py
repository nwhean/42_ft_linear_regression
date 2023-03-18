import csv
from typing import Optional


def read_file(filename: str, x_name: str, y_name: str) -> tuple[list[float]]:
    """Read data from file."""
    x = []
    y = []
    with open('data.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row[x_name]))
            y.append(float(row[y_name]))
    return x, y

def write_file(filename: str, theta: list[float]) -> None:
    """Write the regression parameters to a file."""
    with open(filename, 'w', newline='') as f:
        fieldnames = ["theta" + str(i) for i in range(2)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({i: j for i, j in zip(fieldnames, theta)})

def normalise(x: list[float], y: list[float]) -> tuple[list[float]]:
    """Normalise the input data to improve rate of convergence."""
    # calculate the normalisation parameters
    min_x = min(x)
    max_x = max(x)
    range_x = max_x - min_x

    min_y = min(y)
    max_y = max(y)
    range_y = max_y - min_y

    x_norm = [(i - min_x) / range_x for i in x]
    y_norm = [(i - min_y) / range_y for i in y]
    
    return x_norm, y_norm, min_x, range_x, min_y, range_y

def denormalise(theta: list[float], min_x: float, range_x: float,
                min_y: float, range_y: float) -> list[float]:
    """Denormalise coefficients based on given normalisation parameters."""
    coeff = []
    coeff.append(min_y + range_y * (theta[0] - theta[1] * min_x / range_x))
    coeff.append(theta[1] * range_y / range_x)
    return coeff

def predict(theta: list[float], x: float) -> float:
    """
    Return the predicted value, y based on linear regression model.
    Where
    y = theta0 + theta1 x
    """
    return theta[0] + theta[1] * x

def gradient(theta: list[float], x: list[float], y: list[float]) -> list[float]:
    """
    Return the gradient of linear regression model at the given coefficient c
    """
    if len(x) != len(y):
        raise IndexError(
            "Lengths of x (independent) and y (dependent) must be equal")
    n = len(y)
    grad = []
    pred = [predict(theta, i) for i, j in zip(x, y)]
    grad.append(sum(i - j for i, j in zip(y, pred)))
    grad.append(sum((i - j) * k for i, j, k in zip(y, pred, x)))
    return [-2.0 / n * i for i in grad]

def train(x: list[float], y: list[float], theta: Optional[list[float]] = None,
          precision=10**-12) -> list[float]:
    """
    Given the initial coefficients 'theta', return the final coefficient based
    on gradient descent method.
    """
    if theta is None:
        theta = [0, 0]
    
    ratio = 1   # assume a large initial learning ratio
    mse = mean_squared_error(theta, x, y)

    while mse != 0:
        grad = gradient(theta, x, y)
        theta_n = [i - ratio * j for i, j in zip(theta, grad)]
        mse_n = mean_squared_error(theta_n, x, y)
        
        # reduce learning ratio if mean square error increases (divergence)
        while mse_n > mse:
            ratio /= 2
            theta_n = [i - ratio * j for i, j in zip(theta, grad)]
            mse_n = mean_squared_error(theta_n, x, y)
        
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

def residual_ss(theta: list[float], x: list[float], y: list[float]) -> float:
    """Return the residual sum of squares."""
    pred = [predict(theta, i) for i, j in zip(x, y)]
    return sum((i - j)**2 for i, j in zip(y, pred))

def mean_squared_error(theta: list[float], x: list[float], y: list[float]
        ) -> float:
    """Return the mean squared error."""
    return residual_ss(theta, x, y) / len(y)

def residual_total(theta: list[float], x: list[float], y: list[float]) -> float:
    """Return the total sum of squares."""
    y_mean = mean(y)
    pred = [predict(theta, i) for i, j in zip(x, y)]
    return sum((i - y_mean)**2 for i in y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Reading data...")
    km, price = read_file("data.csv", "km", "price")
    x, y, min_x, range_x, min_y, range_y = normalise(km, price)

    print("Descending gradient...")
    theta = train(x, y)
    coeff = denormalise(theta, min_x, range_x, min_y, range_y)

    outfile = 'coefficient.csv'
    write_file(outfile, coeff)
    print(f"Coefficients written to {outfile}.")
    
    # Calculate precision of regression
    for i, val in enumerate(coeff):
        print(f"theta{i} = {coeff[i]}")
    R2 = 1 - residual_ss(coeff, km, price) / residual_total(coeff, km, price)
    print(f"R_squared = {R2}")

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
