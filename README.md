# Deep Neural Networks - Fruit recognition

## Dataset
[http://www.mimuw.edu.pl/~cygan/assignment1.tgz](http://www.mimuw.edu.pl/~cygan/assignment1.tgz)

## Part 1
You are supposed to implement a convolutional neural network for a given dataset of fruits. Your
solution should contain the following features:

* it should contain both convolutional layers and fully connected layers,
* it should contain batch normalization for convolution, which you implement by basic operations
 (using only simple mathematical operations like +,-, ·, / and matrix multiplication) - it is fine
 if you use the train mode even for the test check,
* implementation should allow painless changes in the architecture, like adding new layers or
 changing number of filters by updating numbers in one place in your code,
* you need to obtain at least 97% test accuracy in front of a person checking your solution,
* you are allowed to load a saved model, however in that case you need to provide a log from the
 training that contains model performance after subsequent epochs together with timestamps.

## Part 2
You should implement two methods of visualizing what in a given image contributes the most to the
outcome of a your trained model (in this task assume the model is fixed).

In your solution provide sample visualisations, but also have a tool ready to compute the results
for images chosen during the lab session inspection.

### Occlusion
For each position in the input image consider a square of size 10 centered in that pixel (you can
change the size if you want). Fill that square with pixels with constant value (it is your task to
propose that value and justify your choice) and compute the loss function for that image. Create a
heatmap our of those values (it is expected that this method takes some time to compute the results
for a given image).

### Gradients
Create heatmaps from pixelwise gradients of the loss function.

## Additional solution features (nonobligatory)
If you want, you can additionally implement:

* dropout - check how it works with and without batch normalization,
* data augmentation,
* try different learning algorithms and different learning rates.

## Deadline
You should submit your solution by email by 23:59 on 23.04.2019 (Tuesday) to your lab teacher with
email title ”Assignment 1 - Deep neural networks”. Your code will be inspected during the lab
session following the deadline. Note that even if you are one minute late after the deadline, your
solution will not be inspected. We have no mercy whatsoever so you better not count on that.