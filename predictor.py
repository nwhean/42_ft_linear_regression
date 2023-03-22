"""This module asks user for input mileage and provide a price estimate."""

import csv

import numpy as np

from trainer import LinearRegression


if __name__ == "__main__":
    INFILE = "coefficient.csv"

    theta = [0.0, 0.0]
    try:
        f = open(INFILE, encoding="utf-8")
    except FileNotFoundError:
        print(f"{INFILE} not found. Coefficient defaults to 0.")
    else:
        # read only 1 row from the file
        reader = csv.DictReader(f)
        row = next(reader)
        theta = [float(row["theta" + str(i)]) for i in range(len(row))]
        f.close()

    theta = np.array(theta)
    model = LinearRegression(theta)
    # prompt user
    mileage = float(input("Please enter mileage: "))
    price = model.predict(mileage)
    print(f"The predicted price is {price:.0f}.")
