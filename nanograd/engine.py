import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self, a + 2 works, but 2 + a does not work (2.__add__(a))
        if not isinstance(other, Value):
            other = Value(other)
        return other + self

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)
                          ), "only supporting int/float powers for now"
        out = Value(self.data**other, _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += out.grad * other * self.data**(other-1)
        out._backward = _backward
        return out

    def __rpow__(self, other):  # other ** self, handles cases like 2 ** a
        assert isinstance(other, (int, float)
                          ), "only supporting int/float bases for now"
        out = Value(other**self.data, _children=(self,), _op=f'{other}**')

        def _backward():
            self.grad += out.grad * other**self.data * math.log(other)
        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self, a * 2 works, but 2 * a does not work (2.__mul__(a))
        return self.__mul__(other)

    def __truediv__(self, other):  # self / other
        if not isinstance(other, Value):
            other = Value(other)
        return self * other**-1

    # other / self, a / 2 works, but 2 / a does not work (2.__truediv__(a))
    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other * self**-1

    def __neg__(self):
        return Value(-self.data, _children=(self,), _op='neg')

    def __sub__(self, other):  # self - other
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data - other.data, _children=(self, other), _op='-')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * -1.0
        out._backward = _backward
        return out

    def __rsub__(self, other):  # other - self
        if not isinstance(other, Value):
            other = Value(other)
        return other + (-self)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1) / (math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.grad * math.exp(x)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
            topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):  # make sure o.grad is initialized with 1.0
            node._backward()
