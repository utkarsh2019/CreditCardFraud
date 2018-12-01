'''
After generating the subsets of the data, we needed to perform
transformations on the labels of the data points.  The original
data set had values of 1 for fraudulent transactions and 0 for
regular transactions.  We transformed this into -1s for fraudulent
transactions and +1s for non-fraudulent transactions.

We also shuffled the data points so that all of the -1s weren't
consecutive and all of the 1s weren't consecutive.  This mixes
them to avoid something weird happening with our training.

TO RUN:
Alter the input and output filename arrays with the subset files.
(Line 20, 21)
'''
import csv
from random import shuffle

def run():
    input = ['../dataset/creditCardFraudSubset250.csv', '../dataset/creditCardFraudSubset500.csv', '../dataset/creditCardFraudSubset.csv']
    output = ['../dataset/250.csv', '../dataset/500.csv', '../dataset/1000.csv']

    for c in range(0,3):
        allData = []
        reader = None
        row1 = ''
        
        with open(input[c], 'rb') as f:
            reader = csv.reader(f)
            row1 = next(reader)

            for row in reader:
                allData += [row]
            
        shuffle(allData)

        for i in allData:
            if (i[30] == '0'):
                i[30] = '1' # Non-Fraudulent
            elif (i[30] == '1'):
                i[30] = '-1' # Fraudulent

        with open(output[c], "wb") as f: 
            writer = csv.writer(f)
            writer.writerow(row1)
            writer.writerows(allData)

if __name__ == "__main__":
    # run()
    print("Please don't run this unless changing data subsets")