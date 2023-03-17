import csv
from typing import Optional


def predict(c: list[float], x: list[float]) -> float:
    """
    Return the predicted value, y based on linear regression model.
    Where
    y = c0 + c1 x
    """
    return c[0] + sum(i*j for i, j in zip(c[1:], x))

def gradient(c: list[float], x: list[float], y: list[float]) -> list[float]:
    """
    Return the gradient of linear regression model at the given coefficient c
    """
    if len(x) != len(y):
        raise IndexError(
            "Lengths of x (independent) and y (dependent) must be equal")
    n = len(y)
    grad = []
    pred = [predict(c, [i]) for i, j in zip(x, y)]
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
    pred = [predict(alpha, [i]) for i, j in zip(x, y)]
    mse = sum((i - j)**2 for i, j in zip(y, pred))

    while mse != 0:
        grad = gradient(alpha, x, y)
        alpha_n = [i - ratio * j for i, j in zip(alpha, grad)]
        pred = [predict(alpha_n, [i]) for i, j in zip(x, y)]
        mse_n = sum((i - j)**2 for i, j in zip(y, pred))
        
        # reduce learning ratio if mean square error increases (divergence)
        while mse_n > mse:
            ratio /= 2
            alpha_n = [i - ratio * j for i, j in zip(alpha, grad)]
            pred = [predict(alpha_n, [i]) for i, j in zip(x, y)]
            mse_n = sum((i - j)**2 for i, j in zip(y, pred))
        
        # stop loop if precision is achieved
        if (mse - mse_n) / mse < precision:
            break
        
        # prepare for the next loop
        alpha = alpha_n
        mse = mse_n
        
    return alpha


if __name__ == "__main__":
    km = []
    price = []

    with open('data.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            km.append(float(row['km']))
            price.append(float(row['price']))

    # calculate the normalisation parameters
    min_x = min(km)
    max_x = max(km)
    range_x = max_x - min_x

    min_y = min(price)
    max_y = max(price)
    range_y = max_y - min_y

    km = [(i - min_x) / range_x for i in km]
    price = [(i - min_y) / range_y for i in price]

    alpha = train(km, price)
    coeff = []
    coeff.append(min_y + range_y * (alpha[0] - alpha[1] * min_x / range_x))
    coeff.append(alpha[1] * range_y / range_x)

    print(min_x, max_x, range_x)
    print(min_y, max_y, range_y)
    print(alpha)
    print(coeff)
