# K-Nearest Neighbors for MNIST Digit Classification

This repository contains two different implementations of the K-Nearest Neighbors algorithm for classifying handwritten digits from the MNIST dataset. The first implementation builds the algorithm from scratch using NumPy, while the second uses scikit-learn's built-in KNN classifier.

## About the Dataset

The MNIST dataset contains images of handwritten digits from 0 to 9. Each image is 28x28 pixels, which means each sample has 784 features (one for each pixel). The training set has 42,000 samples, and we use these to predict labels for the test set.

## Implementation 1: One Nearest Neighbor (NumPy from Scratch)

The first implementation builds a 1-NN classifier completely from scratch using only NumPy. This means we wrote the entire algorithm ourselves without using any machine learning libraries.

### Loading and Preparing the Data

We start by loading the data and separating features from labels:

```python
x_tr = train.iloc[:,1:].values
x_te = test.values
y_te = train.iloc[:,0]
```

The features are all the pixel values (columns 1 onwards), while the labels are in the first column. Notice we store these as NumPy arrays using `.values`, which makes the mathematical operations much faster.

### How it Works

The core idea behind KNN is pretty straightforward. When we want to classify a new digit, we find the most similar digit in our training data and use its label. For 1-NN specifically, we just look at the single closest training example.

To measure similarity between images, we use Euclidean distance, which is basically the straight-line distance between two points in high-dimensional space. The challenge is calculating distances between thousands of images efficiently.

### The Mathematical Trick

Instead of computing distances using loops (which would be incredibly slow), we use a clever mathematical formula. The squared Euclidean distance between two vectors can be rewritten as **||a - b||² = ||a||² - 2(a·b) + ||b||²**. This formula lets us use matrix multiplication, which NumPy handles extremely fast. Here's how it looks in the code:

```python
te_sq = np.sum(x_te_i**2, axis=1).reshape(1, -1)
cross = np.matmul(x_tr, x_te_i.T)
dist = np.sqrt(tr_sq - 2*cross + te_sq)
```

In this snippet, `te_sq` computes the squared norms of test samples, `cross` is the dot product between training and test samples (the 2(a·b) term), and `tr_sq` (computed once outside the loop) contains the squared norms of training samples. We then calculate the full distance using the formula and take the square root.

### Why Batching?

The code processes test samples in batches of 128 instead of all at once:

```python
l = x_te.shape[0]
batch_size = 128

preds = np.empty(l, dtype=y_te.dtype)
tr_sq = np.sum(x_tr**2, axis=1).reshape(-1, 1)

for i in range(0, l, batch_size):
    x_te_i = x_te[i:i+batch_size]
    # ... distance calculations ...
    nn_idx = dist.argmin(axis=0)
    preds[i:i+batch_size] = y_te[nn_idx]
```

This batching approach is important because calculating distances for all test samples at once would require too much memory and it takes more time waiting in memory queue. By processing 128 samples at a time, we balance speed and memory usage. For each batch, we find the index of the minimum distance using `argmin`, which tells us which training sample is closest. We then use that index to look up the corresponding label and store it in our predictions array.

## Implementation 2: K-Nearest Neighbors (Scikit-Learn)

The second implementation uses scikit-learn's KNeighborsClassifier, which is a well-tested and optimized version of the KNN algorithm. This lets us experiment with different values of k without rewriting the algorithm.

### Setting Up the Model

First, we prepare our data in the same way:

```python
x_tr = train.iloc[:,1:]
x_te = test.values
y_tr = train.iloc[:,0]
```

Then we create and configure the KNN classifier:

```python
knn = KNeighborsClassifier(n_neighbors=4, algorithm='brute', metric='euclidean', 
                           weights='distance', n_jobs=-1)
```

Let me break down what each parameter means. The `n_neighbors=4` tells the algorithm to look at the four closest training samples when making a prediction. The `algorithm='brute'` means we check every single training sample to find the nearest neighbors rather than using approximation methods, which ensures we always find the true nearest neighbors. We use `metric='euclidean'` for the same distance measure as our NumPy implementation, and `weights='distance'` is crucial—it means closer neighbors have more influence on the final prediction than farther ones. Finally, `n_jobs=-1` enables parallel processing across all available CPU cores.

### Training and Making Predictions

Training and prediction happen in just a few lines:

```python
t0 = time.time()
knn.fit(x_tr, y_tr)
y_pred = knn.predict(x_te)
print("Time elapsed:", time.time()-t0, "s")
```

The `fit` method stores the training data, and `predict` finds the k nearest neighbors for each test sample and returns the weighted vote of their labels.

### Offline Evaluation Results

Through testing on a validation set (the last 12,000 samples), different values of k were evaluated:

```python
KNN with distance as weight
0.9664166666666667 k=1 k=2
0.969 k=3 k=6
0.9685833333333334 k=5
0.9706666666666667 k=4
0.9674166666666667 k=7
```

The results show that k=4 gave the best accuracy at around 97.07%. Interestingly, using k=1 gave about 96.64% accuracy, and other values like k=3 and k=5 fell somewhere in between.

### The Distance Weighting Insight

There's an interesting pattern in these results. When k is greater than 1, accuracy can actually decrease if we use simple majority voting. This happens because the nearest neighbor is usually the correct label, but when we include more neighbors, some of them might have different labels. Averaging these together can reduce accuracy.

However, using `weights='distance'` reduces that problem by giving more influence to closer neighbours. If the true nearest neighbour is much closer than the others, distance weighting tends to behave like 1-NN; if several neighbours are at similar distances, it produces a smoother decision. This is why distance weighting gave better results than k=1 for k=4.


Both implementations follow the same basic structure and end with creating a submission file:

```python
submission['Label'] = y_pred
submission.to_csv('submission.csv', index=False)
```

This takes our predictions and saves them in the format required for submission.

## Conclusion

These two implementations show different approaches to the same problem. The NumPy version gives you complete control and understanding of what's happening under the hood—you can see exactly how distances are computed and how the nearest neighbor is found. The scikit-learn version provides convenience and optimization, letting you experiment with different parameters easily. Both achieve similar results, demonstrating that the fundamental algorithm is what matters most, not necessarily how you implement it.
