from Neuron import *
import random as r
"""
The IrisANN class represents an artificial neural network that solves the
"Gardens of Heaven" problem. It contains the following properties:
    inputLayer   - a list of neurons representing the input attributes
    hiddenLayer  - a list of hidden neurons in the network
    outputLayer  - a list of neurons representing the output classes
    weights      - a dictionary representing weighted edges between neurons
    learningRate - a constant that defines the rate at which the network learns
"""
class IrisANN:
    def __init__(self):
        self.inputLayer = [Neuron("slen"),Neuron("swid"),Neuron("plen"),Neuron("pwid")]
        self.hiddenLayer = [Neuron("h1"),Neuron("h2"),Neuron("h3"),Neuron("h4")]
        self.outputLayer = [Neuron("Iris-setosa"),Neuron("Iris-versicolor"),Neuron("Iris-virginica")]
        self.initialize_weights()
        self.learningRate = 0.08

    """
    initialize_weights adds weighted edges between the layers of the network.
    Each input neuron is connected to each hidden neuron, and each hidden
    neuron is connected to each output neuron. The weights are set to random
    values between -1/(number of weights) to 1/(number of weights).
    """
    def initialize_weights(self):
        self.weights = {}
        numWeights = len(self.inputLayer) + len(self.hiddenLayer) + len(self.outputLayer)
        for iNode in self.inputLayer:
            for hNode in self.hiddenLayer:
                self.weights[iNode.name + "|" + hNode.name] = r.uniform(-1/numWeights,1/numWeights)
        for hNode in self.hiddenLayer:
            for oNode in self.outputLayer:
                self.weights[hNode.name + "|" + oNode.name] = r.uniform(-1/numWeights,1/numWeights)

    """
    forward_propogation runs inputSet through the neural network, activating
    the appropriate neurons based on input/output.
    """
    def forward_propogation(self, inputSet):
        for i in range(len(self.inputLayer)):
            self.inputLayer[i].output = inputSet[i]
        for hNeuron in self.hiddenLayer:
            hNeuron.input = 0
            for iNeuron in self.inputLayer:
                hNeuron.input += self.weights[iNeuron.name + "|" + hNeuron.name] * iNeuron.output
            hNeuron.activation()
        for oNeuron in self.outputLayer:
            oNeuron.input = 0
            for hNeuron in self.hiddenLayer:
                oNeuron.input += self.weights[hNeuron.name + "|" + oNeuron.name] * hNeuron.output
            oNeuron.activation()

    """
    back_propogation runs the back propogation learning algorithm on the
    neural network. It runs training on examples until the neural network
    correctly identifies enough entries in validation. It has 5 helper
    functions that initialize the error dictionary, propogate the outer,
    hidden, and inner layers, and update the weights of the network's edges.
    """
    def back_propogation(self, examples, validation):
        def initialize_errors():
            errors = {}
            for neuron in self.inputLayer:
                errors[neuron.name] = 1
            for neuron in self.hiddenLayer:
                errors[neuron.name] = 1
            for neuron in self.outputLayer:
                errors[neuron.name] = 1
            return errors

        def propogate_outer(errors):
            for i in range(len(self.outputLayer)):
                neuron = self.outputLayer[i]
                if y == neuron.name:
                    yVal = 1
                else:
                    yVal = -1
                errors[neuron.name] = neuron.activation_prime() * (yVal - neuron.output)

        def propogate_hidden(errors):
            for hNeuron in self.hiddenLayer:
                errorSum = 0
                for oNeuron in self.outputLayer:
                    errorSum += self.weights[hNeuron.name + "|" + oNeuron.name] * errors[oNeuron.name]
                errors[hNeuron.name] = hNeuron.activation_prime() * errorSum

        def propogate_inner(errors):
            for iNeuron in self.inputLayer:
                errorSum = 0
                for hNeuron in self.hiddenLayer:
                    errorSum += self.weights[iNeuron.name + "|" + hNeuron.name] * errors[hNeuron.name]
                errors[iNeuron.name] = iNeuron.activation_prime() * errorSum

        def update_weights(errors):
            for edge in self.weights.keys():
                srcName = edge.split("|")[0]
                destName = edge.split("|")[1]
                for neuron in self.inputLayer:
                    if neuron.name == srcName:
                        src = neuron
                for neuron in self.hiddenLayer:
                    if neuron.name == srcName:
                        src = neuron
                self.weights[edge] += self.learningRate * errors[destName] * src.output

        errors = initialize_errors()
        while not self.validate(validation):
            for x, y in examples:
                self.forward_propogation(x)
                propogate_outer(errors)
                propogate_hidden(errors)
                propogate_inner(errors)
                update_weights(errors)

    """
    validate is a method that returns if the number of incorrect classifications
    the neural network makes from a dataset examples is smaller than a threshold
    """
    def validate(self, examples):
        def num_invalid():
            invalidCount = 0
            for x, y in examples:
                self.forward_propogation(x)
                for neuron in self.outputLayer:
                    if neuron.name == y:
                        if neuron.output < 0.9:
                            invalidCount += 1
                    elif neuron.output > -0.5:
                        invalidCount += 1
            return invalidCount
        return num_invalid() <= 5

    """
    test is a method that runs the neural network on a dataset tests, and
    prints the ANN's prediction as well as the results of the output neurons
    """
    def test(self, tests):
        invalidCount = 0
        for x, y in tests:
            self.forward_propogation(x)
            answer = {"Iris-setosa":0, "Iris-versicolor":0, "Iris-virginica":0}
            maxVal = max([n.output for n in self.outputLayer])
            for neuron in self.outputLayer:
                if neuron.output == maxVal:
                    print("prediction: " + neuron.name)
                answer[neuron.name] = neuron.output
            print("raw output: " + str([answer[n] for n in answer.keys()]))
            print("--------------------------------------------------------------------------------")
