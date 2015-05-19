# MatDNN

#Quickstart

**Setup an optimizer and his hyper-parameters**

Here we set a fixed number of 10 epochs, a minibatch size of 100 observations and use Standard Gradient Descent as our optimization approach. We also supply a constant learning rate hyper-parameter.

```matlab
opt = optimNN(    'Iter', 10, ... 
                  'MiniBatchSize', 100, ...
                  'Method','SGD', ...
                  'LearningRate', 0.1 ...
             )
```

**Setup your NN configuration**

A simple NN with 2 hidden layers of 800 units each for a 10 class classification problem. Here 784 stands for the number of features (columns) in the dataset.
```matlab
nn = setupNN(    'Topology', ...
                 [784, 800, 800, 10] ...
            )
```

**Train**

For training we supply the previous configurations and a training set with the desired classifications.
```matlab
trained_nn = train_ffnn(opt, nn, Xtrain, ctrain);
```

**Predict**

For predicting new observations we just supply the trained classifier from the previous step and the dataset to classify.
```matlab
pred_test =  predict_ffnn(trained_nn, Xtest);
```

* Optionally check the test set accuracy:
```
fprintf('\nTest Set Accuracy: %f %%\n',mean(double(pred_test == ctest)) * 100);
```

The rest of this document details the advanced usage of the library.
 
# Setting up the neural network

**Cost Function**

MatDNN let's you choose between optimizing the *negative log likelihood* cost function or *cross entropy*. By default negLogLikelihood is set.

```
Property: 'CostFunction'
Options : (['negLogLikelihood'] | 'crossentropy')
```

To define any NN param just set the desired property.

E.g.

```matlab
nn = setupNN(    'Topology', ...
                 [784, 800, 800, 10], ...
                 'CostFunction', ...
                 'crossentropy' ...
            )
```

**Activation Functions**

MatDNN also lets you define the used activation function between layers.

```
Property: 'Activation'
Options : (['relu'] | 'tanh' | 'invlog')
```

**Output Function**

```
Property: 'ActivationOutput'
Options : (['softmax'] | 'invLog')
```

**Dropout**

You might set the NN to be learned using dropout as a complexity regulation step, for that just supply the Dropout param and corresponding dropout probabilities vector.

E.g.

```matlab
nn = setupNN(    'Topology', ...
                 [784, 800, 800, 10], ...
                 'Dropout', ...
                 [0.2,0.5,0.5,0.5,0.5] ...
            )
```

**Cost Sensitive Learning**

There are situations where we might benefit from cost sensitive learning, when for example we have highly unbalanced class's. For those cases we can supply a proportions vector and try to compensate the minority class's.

E.g.
```matlab
Property: 'CostSensitive'
Options : PROPORTIONSVECTOR
```

# Optimization


**Method**

The optimizer deals with the tactic used to optimize the specified objective function. Besides standard gradient descent other optimization methods are available:

```
Property: 'Method'
Options : ('Momentum' | 'SGD' | 'NAG' | 'ADADelta')
```
E.g.

```matlab
opt = optimNN(    'Iter', 10, ... 
                  'MiniBatchSize', 100, ...
                  'Method','SGD', ...
                  'LearningRate', 0.1 ...
              )
```

Each of this methods requires his own set of parameters:

* Standard Gradient Descent ('SGD')
```  
Requires: MiniBatchSize, Learning Rate
```
* Standard Momentum ('Momentum')
```
Requires: MiniBatchSize, Learning Rate, Momentum
```
* Nesterov Accelerated Momentum ('NAG')
```
Requires: MiniBatchSize, Learning Rate, Momentum
```
* ADA Delta ('ADADelta)'
```
Parameters: MiniBatchSize, Decay Rate, e
```

**Parameters Annealing**

Both momentum and learning rate might be annealed. For that instead of specifying a constant learning rate or momentum just pass it a vector.

E.g.

A typical use case:
Let's train a classifier for 50 epochs with a linear increasing momentum from 0.5 to 0.99

```matlab
annealedMomentum = linspace(0.5,0.99,50);

opt = optimNN(    'Iter', 50, ... 
                  'MiniBatchSize', 100, ...
                  'Method','SGD', ...
                  'LearningRate', 0.1, ...
                  'Momentum', annealedMomentum ...
             )
```

This library could use a little more care on the unit tests department, feel free to test and contribute to this project. It's unlikely that I will be improving the framework at all. I have since migrated to Python and Scala for development + experiences.

