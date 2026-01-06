# WIDS-2025-26

This repository contains my work for the WiDS project. The repository is organized into weekly explorations, each focusing on different techniques and gradually building up from classical machine learning to modern deep learning methods.

## Repository Structure

### Week 1: K-Nearest Neighbors

The first week focuses on K-Nearest Neighbors (KNN), a simple yet effective algorithm that classifies images by finding similar examples in the training data. This folder contains two implementations that show different approaches to the same problem.

### Week 2: Deep Neural Networks

Building on the foundation from week one, the second week dives into neural networks and deep learning. While KNN achieved around 97% accuracy by comparing images, neural networks learn to recognize the underlying patterns in handwritten digits, which allows them to generalize much better.

This folder contains multiple implementations that progressively increase in sophistication. I started by building a complete neural network from scratch using only NumPy, implementing forward propagation, backpropagation, and gradient descent myself. This gave me a deep understanding of how neural networks actually learn.

From there, I moved to using Keras and built several different architectures. I explored dense neural networks, then moved to convolutional neural networks that better understand spatial relationships in images. I experimented with techniques like batch normalization, dropout, data augmentation, and learning rate scheduling. The final implementation uses ensemble methods, training multiple models and averaging their predictions to achieve even higher accuracy.

I also implemented a similar network using PyTorch to understand the differences between frameworks and get comfortable with PyTorch's more explicit, low-level approach to building neural networks.
