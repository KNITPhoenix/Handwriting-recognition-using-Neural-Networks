# Handwriting-recognition-using-Neural-Networks
We will implement forward propagation algorithm first to get the output and activation of each unit.
Then we will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition.
We will implement the backpropagation algorithm to learn the parameters for the neural network.
Given a training example (x (t) , y (t) ), we will first run a “forward pass” to compute all the activations throughout the network, including the output value of the hypothesis h Θ (x). Then, for each node j in layer l, we would like to compute (l) an “error term” δ j that measures how much that node was “responsible”
for any errors in our output.
For an output node, we can directly measure the difference between the (3) network’s activation and the true target value, and use that to define δ j (since layer 3 is the output layer). For the hidden units, you will compute (l) δ j based on a weighted average of the error terms of the nodes in layer (l + 1). 

Neural Network architecture looks like below:

![a](https://user-images.githubusercontent.com/33430392/125423143-16cafcc2-1d07-4a9d-a2bc-62cc51b67f4c.jpg)

After we have successfully implemented the neural network cost function and gradient computation,we use fmincg to learn a good set parameters.
