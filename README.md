<div align="center">
  <h1>Neural Network in Racket</h1>
    <i>Gradient descent optimization in a purely-functional teaching language </i><br><br>
</div>

This is a demo of a toy neural network, specifically an MLP with `tanh` activation. Performing backprop in Racket turned out to be a lot more difficult than I expected. I hope the resulting implementation is correct, and feel free to open an issue if there are any mistakes. 

The training script was built entirely in the purely-functional "Intermediate Student with <tt>lambda</tt>" teaching language of Racket. I wrote it in preparation for my CS 135 final at UWaterloo. Consequently, it's constrained strictly to the use of "allowed constructs" from CS 135. <sup>You happy now, Prof. Roh?</sup>



## Try It Out
This script can be run with 4 easy steps: 

1. Clone this repo or simply download the Racket file [`rktorch.rkt`](https://github.com/AndyyyYuuu/rktorch/blob/main/rktorch.rkt).
2. Install Racket from [the website](https://racket-lang.org/download/), which will come with the [DrRacket](https://docs.racket-lang.org/drracket/) application.
3. Open `rktorch.rkt` with DrRacket.
4. Hit `Run ▶`.

The program will output, among other things, the train and test loss lists. Keep in mind that since I use `cons` to push to the beginning of my lists, the losses are appended in reverse order. If you see two lists of increasing numbers, it actually means that the loss is decreasing and that the optimization is working. 

## Overview
I will provide a short overview of how this implementation works. 

An `NdArr` object is a `list` of `list`s. These are used to represent the values and gradients of the nodes, and there are a few utility functions for working with them. 

The algorithm is centered around using `Tensor` objects to build a computation tree and then traversing the tree to perform forward and backward passes. 
A `Tensor` object is a `list` with 6 elements: 
1. The **value** of the Tensor, an `NdArr`
2. The **gradient** of the Tensor with respect to the loss, another `NdArr`
3. A `Sym` that tags the Tensor for ease of tracing
    - `'matmul`, `'tanh`, `'add`, `'mse`: the tensor is not a leaf node and was produced through the operation (matrix multiplication, hyperbolic tangent, elementwise addition, mean-squared error respectively)
    - `'param`: a leaf node parameter in the neural network that can be optimized by `optim/step`
    - `'X`: an input Tensor from the dataset which is replaced before performing each forward pass
    - `'Y`: a ground-truth output substituted from the dataset for each loss computation
    - `'Y-pred`: the output of the model
4. A list of all tensors that were involved in the previous operation to compute this tensor. This may include one (for tanh), two (for addition), or no (for leaf nodes) tensors. 
    - Recursively, every tensor includes all earlier tensors in its computation graph.
    - In that sense, the root node is sufficient representation of the model as a whole.
5. A **forward pass function** takes the tensor itself as input and outputs an updated version of itself. It first triggers forward pass functions in child nodes, then performs its operation on its own child nodes to obtain the updated value. 
6. A **backward pass function** that takes the tensor itself as input and updates the gradients of its own child nodes. Also triggers backward pass functions of the child nodes so that the entire graph can be updated recursively. 

The rest of the functionality mostly builds upon the use of this framework. 

