import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value)

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0,1, size=weights_shape) #creates a matrix of random floats between 0 and 1

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        stddev = np.sqrt(2.0 / (fan_in + fan_out)) #no of ip and op units
        return np.random.normal(0, stddev, size=weights_shape).astype(float) # Draws values from a normal distribution with, converts to float
        
class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, size=weights_shape)



#NOTE: The core idea is to initialize the weights so that:
# The variance of the activations is preserved across layers (forward pass), and
# The variance of gradients is preserved (backward pass)
# To achieve this, weights are sampled from a normal distribution with:
# Mean = 0
# Standard deviation based on fan_in and/or fan_out