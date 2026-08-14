import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class Value:

    def __init__(self, data, _children=(), _op='', label=''): # self is the variable, data is the number. (a,2). Value.__init__(a,2)
        self.data = data #turns the input into the data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self): #will print it out nicely in a way we can interpret.
        return f"Value(data={self.data})"

    def __mul__(self, other):  
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)) # must be int or float
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other #checks if you can do the multiplicatio the other way

    def __add__(self, other):   #the __""__ means its an opperator. self is a other is b.
        other = other if isinstance(other, Value) else Value(other) #asks if other is a value object
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
        # a+b = a.__add__(b)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        out = self + (-other)
      
        return out

    def __radd__(self, other): # other + self
        return self + other
        
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/ (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
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
        for node in reversed(topo):
            node._backward()
        
