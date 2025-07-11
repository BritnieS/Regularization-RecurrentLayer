import numpy as np

class Dropout:
    def __init__(self, probability):
        self.probability = probability  # Probability to keep a unit
        self.trainable = False
        self.testing_phase = False
        self.mask = None

    def forward(self, input_tensor):
        if not self.testing_phase:
            # Training phase: randomly mask and scale
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability) / self.probability
            return input_tensor * self.mask
        else:
            # Testing phase: pass through (already scaled during training)
            return input_tensor

    def backward(self, error_tensor):
        # Backward pass uses the same mask from the forward pass
        return error_tensor * self.mask
