class Dropout:
    def __init__(self, probability=1.0):
        self.probability = probability
        self.trainable = False
        self.testing_phase = False

    def forward(self, input_tensor):
        return input_tensor  # Pass-through for now

    def backward(self, error_tensor):
        return error_tensor  # Pass-through for now
