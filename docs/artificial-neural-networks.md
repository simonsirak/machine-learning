# Artificial Neural Networks (ANN)
This lecture will cover ANNs. They essentially estimate some underlying function based on data collected for that function.
This is done by drawing inspiration from neurons, which collect input through dendrites and trigger a spike signal through its 
axon if the inputs are strong enough (this axon is then connected to a dendrite of another neuron). 

## The Artificial Neuron
An artificial neuron has D inputs (corresponding to dendrite connections) with weights associated with each input, and an activation 
function (which helps say if the inputs are strong enough), with a single output (usually the value from the activation function). The 
activation function usually has a corresponding threshold parameter, but this is usually encoded as another weight in the neuron, as 
an added fake input.

## Perceptrons
The first abstraction of a neuron is the single-layer perceptron. A perceptron can be seen as both an algorithm for learning a binary 
classification problem (through linear separation), and as a single-layer neural network with a heaviside step function as the activation
function. The perceptron learns a function that can be interpreted as the decision boundary in linear separation methods. The perceptron 
algorithm only tweaks the weights if the perceptron missclassified an input (as seen in the update step). 
The updates are done according to
<img src="https://render.githubusercontent.com/render/math?math=w_i(t+1)=w_i(t)+r\times (d_j-y_j(t)) \times x_{j,i}">
where j denotes the j:th datapoint and i denotes the component of the input vector. We do this for all datapoints. Then 
we repeat this update step until we see only small improvements, to avoid overfitting (which is still inevitable).

The flaw with perceptrons is that there is no measure of the "best" separation of the data, and it only terminates on linearly separable
data. Multiple separations may be best. The solution to this could be to either create a more complex ANN that no longer can be interpreted
as linear separation (so it may work on data that is not linearly separable), or we could use margin maximization techniques (e.g Support 
Vector Machines, SVMs, as we see in another summary) to define the one "best" linear separation.

> Note: An alternative to perceptron learning is the Delta Rule, which acts on continuous (or in the simple case, 
linear) outputs. The update rule is then instead 
><img src="https://render.githubusercontent.com/render/math?math=w_i(t+1)=w_i(t)+r\times (d_j-y_j(t)) \times x_{j,i}">
>i.e exactly the same, but the actual output y for a given datapoint is now the weighted sum of the input rather than a 
binary classification. The Delta Rule is thus a tool to estimate the underlying function by a linear function, i.e a 
form of linear regression (although it is common to make a classification based on this output anyway). 
>The more general form of the Delta Rule is Backpropagation.

## Neural Networks
When multiple artificial neurons are used together, we get what is called a neural network (single neuron is a special 
case of a single-layer neural network, which can have multiple neurons in the layer). 

The common structures are single layer, multi layer or recurrent (i.e neural network with cycles).

We will focus on feed-forward neural networks. In these, the structure may be multi-layered, but there are no cycles.
For this, we can use *backpropagation* (assuming we use a continuous activation function). The most common activation
function is the sigmoid function, which makes backpropagation mathematically convenient. 

### Error Backpropagation
The idea behind backpropagation is to minimize the training error between our ANN estimate and the true function. 
We then change the weights by taking the gradient (with respect to all the weights) of the error function at the given datapoint. 
The trick with error backpropagation is that, through clever use of the chain rule, we can write the error in a local manner;
i.e layer i only has to consider a local generalized error that is present in layer i+1. This makes it possible to 
"propagate" the error from layer to layer (this is really just a visual explanation, we are actually calculating 
a complex chain rule evaluation by "propagating" local errors backwards).

Error Backpropagation has made training a lot more feasible. However, it is still really slow since it requires MANY 
(usually > 1000) iterations to converge, and depending on the learning rate/initial conditions, it is possible to 
get stuck in local minima. Furthermore, there are many parameters assumed by the individual (step size/learning rate,
number of layers, input/output representation, initialization, number of hidden units per layer and overall structure).

### Deep Neural Networks
With these, backpropagation suffers from a vanishingly small gradient (error gradients become smaller from layer to layer),
so it will work poorly for a large number of layers.

### Convolutional Networks
TODO

