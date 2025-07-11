import numpy as np

class TanH:
    def __init__(self):
        self.trainable = False
        self.activations = None

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        return error_tensor * (1 - self.activations ** 2)
