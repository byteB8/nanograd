import math
import random
from nanograd.engine import Value


class Module:
    def __init__(self):
        self.parameters = []

    def parameters(self):
        return self.parameters

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1),
                        label=f"w{i}") for i in range(nin)]
        self.b = Value(0.0, label='b')
        self.nonlin = nonlin

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh() if self.nonlin else act
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'tanh' if self.nonlin else 'linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        self.layers = []
        last_size = nin
        for nout in nouts:
            layer = Layer(last_size, nout)
            self.layers.append(layer)
            last_size = nout

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
