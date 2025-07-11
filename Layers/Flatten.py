class Flatten:
    def __init__(self):
        self.trainable = False
        self.input_shape = None

    # Flattens the input for use in fully connected layers
    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape # Save the original shape
        batch_size = input_tensor.shape[0] # First dimension is always batch size
        return input_tensor.reshape(batch_size, -1) # Flatten remaining dimensions

    # Restores the gradient shape for earlier layers during backprop
    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)