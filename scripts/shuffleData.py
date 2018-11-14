import csv
from random import shuffle

allData = []
csv1 = '../dataset/creditCardFraudSubset.csv'

with open(csv1, 'rb') as f:
    reader = csv.reader(f)
    row1 = next(reader)

    for row in reader:
        allData += [row]
    
    shuffle(allData)

    with open('../dataset/1000.csv', "wb") as f: 
        writer = csv.writer(f)
        writer.writerow(row1)
        writer.writerows(allData)