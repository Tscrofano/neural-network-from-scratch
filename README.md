# Neural Network From Scratch

A small neural-network framework implemented from scratch in Python to
understand automatic differentiation, computational graphs, and
backpropagation. Made with the help of Andrej Karpathy's "
The spelled-out intro to neural networks and backpropagation: building micrograd" informative YouTube video.

## Overview

This project implements a basic neural network without using
PyTorch, TensorFlow, or other machine-learning frameworks.

The project builds a neural network from the ground up:

Value → Neuron → Layer → MLP

## Features

- Scalar automatic differentiation
- Computational graph construction
- Reverse-mode backpropagation
- Custom mathematical operators
- Tanh activation function
- Neuron and layer abstractions
- Multilayer perceptron
- Squared-error loss
- Gradient-based optimization
- Computational graph visualization

## Project Structure

```text
value.py
    Automatic differentiation engine

neural_network.py
    Neuron, Layer, and MLP classes

neural_network_from_scratch.ipynb
    Demonstrations and experiments

requirements.txt
    Python dependencies
