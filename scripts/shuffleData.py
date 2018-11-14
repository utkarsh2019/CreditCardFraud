import csv
from random import shuffle


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

    with open(output[c], "wb") as f: 
        writer = csv.writer(f)
        writer.writerow(row1)
        writer.writerows(allData)