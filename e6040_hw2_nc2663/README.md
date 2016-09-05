# Assignment 2
--------------

In this assignment we experimented with Multilayer Perceptrons configurations and empirically studied various relationships among number of layers and number of parameters.
We used [Deep Learning Tutorials Project](https://github.com/lisa-lab/DeepLearningTutorials). 

In this assignment, we used Street View House Number([SVHN](http://ufldl.stanford.edu/housenumbers/)) dataset. The dataset is similar in flavor to MNIST, but contains substantially more labeled data, and comes from a significantly harder, real world problem (recognizing digits and numbers in natural scene images). We used the Format 2 of the SVHN dataset. Each sample in this format is a MNIST-like 32-by-32 RGB image centered around a single character. Many of the images do contain some distractors on the sides, which of course makes the problem interesting.

The task was to implement an MLP to classify the images of the SVHN dataset. The input to the MLP is a color image, and the output is a digit between 0 and 9. A python routine called load data is provided to you for downloading and preprocessing the dataset.
