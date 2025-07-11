import numpy as np
import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.layers = []
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.input_tensor = None
        self.label_tensor = None
        self.data_layer = None
        self.loss_layer = None

    def append_layer(self, layer):
    # If the layer is trainable, initialize its weights and assign a copy of the optimizer
        if hasattr(layer, 'trainable') and layer.trainable:
            if hasattr(layer, 'initialize'):
                layer.initialize(self.weights_initializer, self.bias_initializer)
            if hasattr(layer, 'optimizer'):
                layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def forward(self, input_tensor=None):
        if input_tensor is not None:
            tensor = input_tensor
            self.input_tensor = input_tensor
        elif self.data_layer is not None:
            tensor, self.label_tensor = self.data_layer.next()
        elif self.input_tensor is not None:
            tensor = self.input_tensor
        else:
            raise ValueError("No input data provided for forward pass.")
        for layer in self.layers:
            tensor = layer.forward(tensor)
        return tensor

    def backward(self, error_tensor):
        tensor = error_tensor
        for layer in reversed(self.layers):
            tensor = layer.backward(tensor)
        return tensor

    def train(self, iterations):
        if self.data_layer is None or self.loss_layer is None:
            raise Exception("Data layer or loss layer not set in the network.")
        for _ in range(iterations):
            input_tensor, label_tensor = self.data_layer.next()
            prediction = self.forward(input_tensor)
            loss = self.loss_layer.forward(prediction, label_tensor)

            # Add regularization loss
            reg_loss = 0
            for layer in self.layers:
                if hasattr(layer, 'optimizer') and layer.optimizer and layer.optimizer.regularizer:
                    if hasattr(layer, 'weights'):
                        reg_loss += layer.optimizer.regularizer.norm(layer.weights)
            loss += reg_loss

            error_tensor = self.loss_layer.backward(label_tensor)
            self.backward(error_tensor)

    def test(self, input_data):
        results = []
        for sample in input_data:
            prediction = self.forward(sample[None, ...])  # add batch axis
            results.append(prediction)
        return np.vstack(results)

    @property
    def phase(self):
        return "test" if self.layers[0].testing_phase else "train"

    @phase.setter
    def phase(self, value):
        is_testing = (value == "test")
        for layer in self.layers:
            layer.testing_phase = is_testing
