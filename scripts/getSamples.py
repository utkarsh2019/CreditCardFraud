import sys, getopt, random

numberOfDataPoints = 284807 # Total number of data points

# Thanks, TutorialsPoint https://www.tutorialspoint.com/python/python_command_line_arguments.htm
def main(argv):
    inputFileName = './../dataset/creditcard.csv'
    outputFileName = './../dataset/creditCardFraudSubset.csv'
    n = 508 # number of non-fraudulent data points
    f = 492 # number of fraudlent data points
    withHeaders = False # if the headers should be included as the first row of the file

    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'hwi:o:n:f:', [])
    except getopt.GetoptError:
        print 'getSamples.py -h'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'getSamples.py -<arg> <argument value> -<arg> <argument value> ...'
            print '- - - - - - -'
            print '-i <inputFile>\t.csv file of the CreditCardFraud data'
            print '-o <outputFile>\tfile to output subset of data'
            print '-n <number>\tnumber of non-fraudulent points to select (508 by default)'
            print '-f <number>\tnumber of fraudulent points to select (492 by default)'
            print '-w\t\twith column headers as first row'
            print '-h\t\toutput help for arguments'
            sys.exit()
        elif opt == '-i':
            inputFileName = arg
        elif opt == '-o':
            outputFileName = arg
        elif opt == '-n':
            n = int(arg)
        elif opt == '-f':
            f = int(arg)
        elif opt == '-w':
            withHeaders = True
        
    # Seed Random Number Generator
    random.seed(2050)

    # Select n random numbers in range (0, numberOfDataPoints)
    randomNumbers = dict()
    for i in range(n):
        number = random.randint(0, numberOfDataPoints)
        if number not in randomNumbers:
            randomNumbers[number] = 1
    # Get sorted list of numbers
    sortedNumbers = randomNumbers.keys()
    sortedNumbers.sort()

    # Traverse data and get data points
    fraudulentData = []
    nonFraudulentData = []
    lineCounter = -1 # the line we are working with in the data file
    current = 0 # the index of the current line number we're looking for in the data file
    headerLine = ''
    infile = open(inputFileName, 'r')
    headerLine = infile.readline()
    try:
        for line in infile:
            lineCounter = lineCounter + 1
            tokens = line.split(',')
            # Fraudulent Case
            if tokens[len(tokens)-1] == '"1"\n':
                fraudulentData += [line]
                # Handle a fraudlent case being picked as one of our random numbers by accident
                if lineCounter == sortedNumbers[current]:
                    '''print 'collision on ' + str(lineCounter) + ' and ' + str(sortedNumbers[current])'''
                    current += 1
            # Non-Fraudulent Case
            elif lineCounter == sortedNumbers[current]:
                nonFraudulentData += [line]
                current += 1
    except IndexError:
        pass    
    infile.close()

    # Scramble fraudulent data if not selecting all of them
    if f < 492:
        random.shuffle(fraudulentData)

    # Write data points to subset file
    outfile = open(outputFileName, 'w')
    if withHeaders:
        outfile.write(headerLine)
    for line in range(f):
        outfile.write(fraudulentData[line])
    for line in nonFraudulentData:
        outfile.write(line)
    outfile.flush()
    outfile.close()



if __name__ == "__main__":
   main(sys.argv[1:])