import numpy as np
from Layers.FullyConnected import FullyConnected


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_fc = FullyConnected(input_size + hidden_size, hidden_size)
        self.output_fc = FullyConnected(hidden_size, output_size)

        self.hidden_state = np.zeros((1, hidden_size))
        self.memorize_flag = False
        self.trainable = True
        self.testing_phase = False
        self._optimizer = None

    @property
    def memorize(self):
        return self.memorize_flag

    @memorize.setter
    def memorize(self, value):
        self.memorize_flag = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self.hidden_fc.optimizer = value
        self.output_fc.optimizer = value

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fc.initialize(weights_initializer, bias_initializer)
        self.output_fc.initialize(weights_initializer, bias_initializer)

    @property
    def weights(self):
        return self.hidden_fc.weights

    @weights.setter
    def weights(self, value):
        self.hidden_fc.weights = value

    @property
    def gradient_weights(self):
        return self.hidden_fc.gradient_weights

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.hidden_inputs = []
        self.outputs = []

        if not self.memorize_flag:
            self.hidden_state = np.zeros((1, self.hidden_size))

        for t in range(input_tensor.shape[0]):
            xt = input_tensor[t:t + 1]
            concat = np.concatenate((xt, self.hidden_state), axis=1)
            self.hidden_inputs.append(concat)

            self.hidden_state = self.hidden_fc.forward(concat)
            output = self.output_fc.forward(self.hidden_state)
            self.outputs.append(output)

        return np.vstack(self.outputs)

    def backward(self, error_tensor):
        d_hidden = np.zeros((1, self.hidden_size))
        d_input = []

        # Reset accumulators
        hidden_grad_accum = np.zeros_like(self.hidden_fc.weights)
        output_grad_accum = np.zeros_like(self.output_fc.weights)

        for t in reversed(range(error_tensor.shape[0])):
            d_out = error_tensor[t:t + 1]
            d_hidden_out = self.output_fc.backward(d_out)
            d_hidden_total = d_hidden + d_hidden_out

            output_grad_accum += self.output_fc.gradient_weights

            d_concat = self.hidden_fc.backward(d_hidden_total)
            hidden_grad_accum += self.hidden_fc.gradient_weights

            d_input_t = d_concat[:, :self.input_size]
            d_hidden = d_concat[:, self.input_size:]
            d_input.insert(0, d_input_t)

        # Set accumulated gradients
        self.hidden_fc._gradient_weights = hidden_grad_accum
        self.output_fc._gradient_weights = output_grad_accum

        return np.vstack(d_input)

    def calculate_regularization_loss(self):
        reg_loss = 0
        if self.optimizer and self.optimizer.regularizer:
            reg_loss += self.optimizer.regularizer.norm(self.hidden_fc.weights)
            reg_loss += self.optimizer.regularizer.norm(self.output_fc.weights)
        return reg_loss
