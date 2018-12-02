import numpy as np


class Evaluator(object):
    def __init__(self,height,width):
        self.loss_value = None
        self.grads_values = None
        self.f_outputs = None
        self.width = width
        self.height = height

    def set_data(self,f):
        self.loss_value = None
        self.grads_values = None
        self.f_outputs = None
        self.f_outputs = f

    def evaluate(self, x, prev, f_outputs):
        x = x.reshape((1, self.height, self.width, 3))
        prev = prev.reshape((1, self.height, self.width, 3))
        outs = f_outputs([x, prev])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    def loss(self, x, prev):
        assert self.loss_value is None
        loss_value, grad_values = self.evaluate(x, prev,  self.f_outputs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x, prev):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values