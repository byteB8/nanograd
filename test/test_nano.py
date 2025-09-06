import unittest
from nanograd.engine import Value
from nanograd.nn import Neuron, Layer, MLP


class TestValue(unittest.TestCase):
    def test_add(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        self.assertEqual(c.data, 3.0)

    def test_mul(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a * b
        self.assertEqual(c.data, 2.0)

    def test_nn(self):
        model = MLP(3, [4, 4, 1])
        xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5],
              [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
        ys = [1.0, -1.0, -1.0, 1.0]
        model.train(xs, ys)
        self.assertAlmostEqual(model([Value(xi)
                               for xi in xs[0]]).data, ys[0], places=1)

    def test_relu(self):
        a = Value(1.0)
        b = a.relu()
        self.assertEqual(b.data, 1.0)
        a = Value(-1.0)
        b = a.relu()
        self.assertEqual(b.data, 0.0)

    def test_tanh(self):
        a = Value(1.0)
        b = a.tanh()
        self.assertAlmostEqual(b.data, 0.7615941559, places=2)
        a = Value(-1.0)
        b = a.tanh()
        self.assertAlmostEqual(b.data, -0.7615941559, places=2)

    def test_exp(self):
        a = Value(1.0)
        b = a.exp()
        self.assertAlmostEqual(b.data, 2.7182818284, places=2)
        a = Value(-1.0)
        b = a.exp()
        self.assertAlmostEqual(b.data, 0.3678794411, places=2)

    def test_sub(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a - b
        self.assertEqual(c.data, -1.0)

    def test_neg(self):
        a = Value(1.0)
        b = -a
        self.assertEqual(b.data, -1.0)

    def test_div(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a / b
        self.assertEqual(c.data, 0.5)

    def test_rdiv(self):
        a = Value(1.0)
        b = 2.0
        c = b / a
        self.assertEqual(c.data, 2.0)

    def test_rsub(self):
        a = Value(1.0)
        b = 2.0
        c = b - a
        self.assertEqual(c.data, 1.0)

    def test_radd(self):
        a = Value(1.0)
        b = 2.0
        c = b + a
        self.assertEqual(c.data, 3.0)

    def test_rmul(self):
        a = Value(1.0)
        b = 2.0
        c = b * a
        self.assertEqual(c.data, 2.0)

    def test_rpow(self):
        a = Value(1.0)
        b = 2.0
        c = b ** a
        self.assertEqual(c.data, 2.0)

    def test_rtruediv(self):
        a = Value(1.0)
        b = 2.0
        c = b / a
        self.assertEqual(c.data, 2.0)

    def test_backward(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a * b
        c.backward()
        self.assertEqual(a.grad, 2.0)
        self.assertEqual(b.grad, 1.0)

    def test_relu_backward(self):
        a = Value(1.0)
        b = a.relu()
        b.backward()
        self.assertEqual(a.grad, 1.0)

    def test_tanh_backward(self):
        a = Value(1.0)
        b = a.tanh()
        b.backward()
        self.assertAlmostEqual(a.grad, 0.4199743416, places=2)

    def test_exp_backward(self):
        a = Value(1.0)
        b = a.exp()
        b.backward()
        self.assertAlmostEqual(a.grad, 2.7182818284, places=2)

    def test_sub_backward(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a - b
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, -1.0)


if __name__ == '__main__':
    unittest.main()
