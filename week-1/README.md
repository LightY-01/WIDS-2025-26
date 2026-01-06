# K-Nearest Neighbors for MNIST Digit Classification

This repository contains two different implementations of the K-Nearest Neighbors algorithm for classifying handwritten digits from the MNIST dataset. The first implementation builds the algorithm from scratch using NumPy, while the second uses scikit-learn's built-in KNN classifier.

## About the Dataset

The MNIST dataset contains images of handwritten digits from 0 to 9. Each image is 28x28 pixels, which means each sample has 784 features (one for each pixel). The training set has 42,000 samples, and we use these to predict labels for the test set.

## Implementation 1: One Nearest Neighbor (NumPy from Scratch)

The first implementation builds a 1-NN classifier completely from scratch using only NumPy. This means we wrote the entire algorithm ourselves without using any machine learning libraries.

### How it Works

The core idea behind KNN is pretty straightforward. When we want to classify a new digit, we find the most similar digit in our training data and use its label. For 1-NN specifically, we just look at the single closest training example.

To measure similarity between images, we use Euclidean distance, which is basically the straight-line distance between two points in high-dimensional space. The challenge is calculating distances between thousands of images efficiently.

### The Mathematical Trick

Instead of computing distances using loops (which would be incredibly slow), we use a clever mathematical formula. The squared Euclidean distance between two vectors can be rewritten as:

**||a - b||² = ||a||² - 2(a·b) + ||b||²**

This formula lets us use matrix multiplication, which NumPy handles extremely fast. We compute all the distances between test samples and training samples in batches rather than one at a time.

### Why Batching?

The code processes test samples in batches of 128. This is important because calculating distances for all test samples at once would require too much memory. By processing 128 samples at a time, we balance speed and memory usage.

For each batch, the code calculates the distance to every training sample, finds which training sample is closest, and assigns that training sample's label as the prediction.

## Implementation 2: K-Nearest Neighbors (Scikit-Learn)

The second implementation uses scikit-learn's KNeighborsClassifier, which is a well-tested and optimized version of the KNN algorithm. This lets us experiment with different values of k without rewriting the algorithm.

### Configuration Choices

The model is set up with several specific parameters. We use k=4, which means we look at the four nearest neighbors instead of just one. The algorithm is set to 'brute', meaning it checks every single training sample to find the nearest neighbors rather than using approximation methods.

We also use distance-weighted voting, which is an important detail. This means closer neighbors have more influence on the final prediction than farther ones. This weighting scheme helped improve accuracy compared to simple majority voting.

### Why k=4?

Through offline evaluation on a validation set, different values of k were tested. The results showed that k=4 gave the best accuracy at around 97.07%. Interestingly, using k=1 gave about 96.64% accuracy, and other values like k=3 and k=5 fell somewhere in between.

### The Distance Weighting Insight

There's an interesting pattern in the results. When k is greater than 1, accuracy can actually decrease if we use simple majority voting. This happens because the nearest neighbor is usually the correct label, but when we include more neighbors, some of them might have different labels. Averaging these together can reduce accuracy.

However, using distance as a weight solves this problem. Closer neighbors get more say in the final decision, so even if we include k=4 neighbors, the closest one (which is usually correct) has the strongest influence. This is why distance weighting gave better results than k=1 for some values of k.

## Running the Code

Both implementations follow the same basic structure. They load the training and test data, process it to separate features from labels, train the model (or in the NumPy case, just store the training data), make predictions on the test set, and save the results to a submission file.

The scikit-learn version runs with parallel processing enabled (n_jobs=-1) to make use of all available CPU cores, which speeds up the computation significantly.

## Key Takeaway

These two implementations show different approaches to the same problem. The NumPy version gives you complete control and understanding of what's happening under the hood, while the scikit-learn version provides convenience and optimization. Both achieve similar results, demonstrating that the fundamental algorithm is what matters most, not necessarily how you implement it.
