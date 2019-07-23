import numpy
"""
The Neuron class represent the neurons of an artificial neural network.
It contains the following properties:
    name   - identifier for the neuron
    input  - current input value for the neuron
    output - current output of the neuron
"""
class Neuron:
    def __init__(self, name):
        self.name = name
        self.input = 0
        self.output = 0

    """
    activation sets the output of the neuron to the result of taking the
    hyperbolic tangent of the neuron's input
    """
    def activation(self):
        self.output = numpy.tanh(self.input)

    """
    activation_prime returns the result of the deriviative of the activation
    function of the neuron's input
    """
    def activation_prime(self):
        return (1 - numpy.tanh(self.input) * numpy.tanh(self.input))
