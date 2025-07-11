import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None
        self.label_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        # Add epsilon for numerical stability
        eps = 1e-12
        return -np.sum(label_tensor * np.log(input_tensor + eps))

    def backward(self, label_tensor):
        eps = 1e-12
        return -label_tensor / (self.input_tensor + eps)