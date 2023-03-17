import csv

from trainer import predict


if __name__ == "__main__":
    alpha = [0.0, 0.0]
    
    infile = "coefficient.csv"
    try:
        f = open(infile)
    except FileNotFoundError:
        print(f"{infile} not found. Coefficient defaults to 0.")
    else:
        # read only 1 row from the file
        reader = csv.DictReader(f)
        row = next(reader)
        alpha = [float(row["theta" + str(i)]) for i in range(len(row))]
        f.close()
    
    # prompt user 
    mileage = float(input("Please enter mileage: "))
    price = predict(alpha, [mileage])
    print(f"The predicted price is {price:.0f}.")
