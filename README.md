# BNN-Project
My Deep Learning project consists in a summary presentation of Bayesian Neural Networks based on articles I have read.
It also contains a fiew implementations using Python.

# Bayesian Neural Network (BNN) for MNIST

This project implements a Bayesian Neural Network designed for digit classification on the MNIST dataset. The codebase is inspired by and builds upon the tutorial:

    Jospin, Laurent Valentin, et al. "Hands-On Bayesian Neural Networks—A Tutorial for Deep Learning Users." IEEE Computational Intelligence Magazine (2022).

It was also enhanced with the assistance of AI based coding tools.

# Key Enhancements & Innovations

While based on the tutorial's foundation, this repository introduces several significant modifications and diagnostic tools to explore Bayesian uncertainty and model architecture:

Laplace Prior Option: In addition to the standard Gaussian prior, I added the ability to use a Laplace Prior. This encourages weight sparsity, which can be visualized through the weight distribution plots.

Integrated MC-Dropout: I implemented and tested Monte-Carlo (MC) Dropout as an additional source of stochasticity, allowing for more robust uncertainty estimation during inference.

Ensemble Predictions: The script supports training multiple networks. I added visualizations to plot the mean predicted probabilities for each individual network in the ensemble, showing how different models "disagree" on difficult samples.

Advanced Uncertainty Metrics: I integrated Entropy calculation to quantify the "doubt" of the model, particularly when facing unseen classes or pure noise.

Calibration Analysis: I added Calibration Plots (Reliability Diagrams) for each test case to evaluate if the model's confidence scores actually match its real-world accuracy.

Weight Sparsity Visualization: A new feature that plots the histogram of weight distributions (log-scale) to observe the effects of the chosen prior (Gaussian vs. Laplace).

Adversarial Robustness (FGSM): Added an implementation of the Fast Gradient Sign Method (FGSM) to test how Bayesian uncertainty reacts to adversarial attacks.

# Project Structure

    dataset.py: Handles MNIST loading and filtering (allows training on a subset of classes to create "unseen" data scenarios).

    viModel.py: Defines the Bayesian layers (MeanFieldGaussianFeedForward, MeanFieldGaussian2DConvolution) and the BayesianMnistNet architecture.

    viExperiment.py: The main execution script for training, evaluating, and generating all diagnostic plots.

# Experimental Diagnostics

The code generates several plots to provide a "look under the hood" of the BNN:

Calibration Plot: Compares predicted confidence vs. observed accuracy.

Uncertainty Decomposition: Bar charts separating Aleatoric uncertainty (within-sample variance) and Epistemic uncertainty (across-sample/ensemble variance).

Entropy Analysis: Printed values showing the average information entropy for seen classes, unseen classes, and white noise.

Weight Distribution: A histogram showing how the weights are centered and spread, specifically demonstrating the "peaky" nature of the Laplace prior.

# How to Run
Installation

Ensure you have torch, torchvision, matplotlib, and numpy installed.
Training with a Laplace Prior

To train a model using the Laplace prior and reserving class '5' as the unseen class:
Bash

    python viExperiment.py --prior laplace --filteredclass 5 --nepochs 10 --numnetworks 5

Loading and Testing

To skip training and run evaluations on a previously saved model:
Bash

    python viExperiment.py --notrain --savedir ./models --nruntests 100
