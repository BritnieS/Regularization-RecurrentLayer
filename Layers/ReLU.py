class ReLU:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return (input_tensor > 0) * input_tensor

    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor > 0)