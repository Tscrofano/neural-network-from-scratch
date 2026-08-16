from value import Value
from neural_network import Neuron, Layer, MLP

"""
#basic copmputations using Value
"""

a = Value(2.0)
b = Value(3.0)

c = a * b

print(c)

"""
#Back propogation
"""

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a * b
e.label = 'e'

d = e + c
d.label = 'd'

f = Value(-2.0, label='f')

L = d * f
L.label = 'L'

L.backward() #will back proporgate through every step

#Gives the gradiants for each variable used in to get L.
print(a.grad) 
print(b.grad)
print(c.grad)
print(f.grad)

"""
#Use of a single neuron
"""

#will assign a random weight to each intiger in between -1 to 1. 
#will also assign a random basis to add after multiplying eash weight and x value together. 
x = [2.0, 3.0]

n = Neuron(2)

output = n(x)

print(output)

"""
#Layers
"""
#Tells how many neurons to make and also gives them the numbers to use in each neuron created. 

layer = Layer(2, 3) #build the Layer. Layer.__init__(2,3)

x = [2.0, 3.0]

output = layer(x) #Runs the call function. layer.__call__(x)

print(output)

"""
#MLP
"""

n = MLP(3, [4, 4, 1])

x = [2.0, 3.0, -1.0]

prediction = model(x)

print(prediction)

"""
Training
"""

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

ypred = [model(x) for x in xs]

for k in range(20):
    
    #forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

    #UPDATE 
    for p in n.parameters(): #n.parameters() contains all of the weights and biases
        p.grad = 0.0
    loss.backward() #creates new p.grads based on the new p.datas

    #Goes against gradient to reduce loss function
    for p in n.parameters():
        p.data += -0.05 * p.grad #pushes p.data to our ys / ygt

    print(k, loss.data)



































