import random
from value import Value
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # puts sef.w with x per input. [(Value(data=-0.29440897345546624), 2.0), (Value(data=0.718934288002443), 3.0)]
        out = act.tanh()
        
        return out
    def parameters(self):
      return self.w + [self.b]

class Layer:
  
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
      
class MLP:
  
  def __init__(self, nin, nouts):
    sz = [nin] + nouts #combines them into one list
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] #creates layes with nin and nout based on nuymbers in the list sz
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
