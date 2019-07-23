from IrisANN import *
"""
iris.py solves the "Gardens of Heaven". Follow the prompts for instructions!
"""

"""
generate_set searches for a text file filename in the same directory as the
program. The text file must have the format of that in the Fisher Iris database
It then returns a shuffled list of pairs of input vectors/output strings.
"""
def generate_set(filename):
    with open(filename, "r") as f:
        data = f.read().splitlines()
    newSet = [generate_entry(e) for e in data]
    r.shuffle(newSet)
    return newSet

"""
generate_entry formats a single entry string, and returns a pair of input/output
"""
def generate_entry(entry):
    attributes = [float(x) for x in entry.split(",")[:4]]
    output     = entry.split(",")[-1]
    return (attributes, output)

"""
maxima takes a set of data from the Fisher Iris database and returns a list of
the maximum values from each numerical column in the set
"""
def maxima(data):
    cols   = [[],[],[],[]]
    colMax = []
    for i in range(len(data[0][0])):
        for j in range(len(data)):
            cols[i].append(data[j][0][i])
    for col in cols:
        colMax.append(max(col))
    return colMax

"""
normalize_data takes the values in data and a vector of attribute maxima
colMaxima. It normalizes the values in data to the domain [-3,3], which is
suitable for the hyperbolic tangent activation function in the neural network.
"""
def normalize_data(data, colMaxima):
    cols = [[],[],[],[]]
    for i in range(len(data[0][0])):
        for j in range(len(data)):
            cols[i].append(data[j][0][i])
    for i in range(len(cols)):
        colMax = colMaxima[i]
        for j in range(len(cols[i])):
            cols[i][j] = (cols[i][j] / colMax - 0.5) * 6
    for i in range(len(data[0][0])):
        for j in range(len(data)):
            data[j][0][i] = cols[i][j]
    return data

"""
run_tests takes an IrisANN and runs tests inputted by the user. The user can
either input the name of a text file in the current directory or input data
entries manually. The user can press q to quit anytime.
"""
def run_tests(ann):
    q = True
    while q:
        choice = input("would you like to test a set of entries in a .txt file? (y/n), or q (quit): ")
        if choice == "y":
            filename = input("what's the name of the file?: ")
            testSet  = normalize_data(generate_set(filename), dataMaxima)
            ann.test(testSet)
        elif choice == "n":
            while q:
                userInput = input("please enter data attributes, b (back), or q (quit): ")
                if userInput == "q":
                    q = False
                if userInput == "b":
                    break
                try:
                    testEntry = normalize_data([generate_entry(userInput)], dataMaxima)
                    ann.test(testEntry)
                except ValueError:
                    continue
        elif choice == "q":
            q = False

if __name__ == "__main__":
    neuralNet  = IrisANN()
    allData    = generate_set("all_data.txt")
    dataMaxima = maxima(allData)
    training   = normalize_data(generate_set("training.txt"), dataMaxima)
    validation = normalize_data(generate_set("validation.txt"), dataMaxima)
    neuralNet.back_propogation(training, validation)
    run_tests(neuralNet)
