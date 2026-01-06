# Deep Learning Approaches for MNIST Digit Classification

After exploring K-Nearest Neighbors in week 1, this week we dive into neural networks and deep learning. While KNN achieved around 97% accuracy by finding similar images, neural networks learn to recognize patterns in the data itself. This fundamental difference allows them to generalize better and achieve higher accuracy on the MNIST digit classification task.

## Why Neural Networks?

KNN has a limitation—it needs to store all training data and compare each test image against thousands of training images. Neural networks, on the other hand, learn compact representations of the patterns in handwritten digits during training. Once trained, they can classify new images very quickly using just these learned patterns encoded in their weights.

## Implementation 1: Building a Neural Network From Scratch

Before using any deep learning libraries, we built a complete neural network using only NumPy. This helps us understand exactly what's happening under the hood when we train a neural network.

### Data Preprocessing

The first step is normalizing our data, which helps the network learn more effectively:

```python
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-10

X_norm = (X - mean)/std
X_test_norm = (X_test - mean)/std
```

We subtract the mean and divide by the standard deviation for each pixel. The small value `1e-10` prevents division by zero. This normalization ensures that all our features are on a similar scale, which makes gradient descent work much better.

### Network Architecture

Our network has three layers with specific dimensions:

```python
input_dim = 784    # 28x28 pixels flattened
hidden1 = 128      # first hidden layer
hidden2 = 64       # second hidden layer
num_classes = 10   # digits 0-9
```

The architecture starts with 784 input nodes (one for each pixel), narrows down to 128 nodes in the first hidden layer, then to 64 in the second, and finally outputs 10 probabilities (one for each digit). This funnel shape helps the network learn increasingly abstract representations of the input.

### Weight Initialization

We use something called He initialization, which is particularly good for networks with ReLU activation:

```python
rng = np.random.RandomState(42)
w1 = rng.randn(input_dim, hidden1) * np.sqrt(2/input_dim)
b1 = np.zeros((1, hidden1))
```

The weights are initialized with random values scaled by `sqrt(2/input_dim)`. This scaling is important because it helps prevent the gradients from becoming too large or too small as they flow through the network. The biases start at zero, which is a common practice.

### Activation Functions

We use two types of activation functions. ReLU (Rectified Linear Unit) is used in the hidden layers:

```python
def relu(x):
    if np.isnan(x).any():
        raise ValueError("Input contains NaN values")
    return np.maximum(0, x)
```

ReLU simply keeps positive values and zeros out negative ones. It's computationally efficient and helps the network learn non-linear patterns. The final layer uses softmax, which converts the network's raw outputs into probabilities that sum to one:

```python
def softmax(x):
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / np.sum(ex, axis=1, keepdims=True)
```

Notice we subtract the maximum value before taking the exponential. This is a numerical stability trick that prevents overflow when dealing with large numbers.

### The Training Loop

Training happens over multiple epochs, where each epoch processes the entire training set once:

```python
for epoch in range(epochs):
    # Forward Pass
    Z1 = np.dot(train_X_np, w1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, w2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, w3) + b3
    A3 = softmax(Z3)
```

The forward pass computes the network's predictions. We multiply the input by weights, add biases, and apply activations at each layer. This process transforms our pixel values through multiple representations until we get class probabilities.

After the forward pass, we calculate the loss using categorical cross-entropy:

```python
eps = 1e-12
l = -np.sum(y_onehot * np.log(A3 + eps)) / m
```

The small epsilon value prevents taking the log of zero. This loss measures how different our predictions are from the true labels, with lower values indicating better performance.

### Backpropagation

The backward pass computes gradients, which tell us how to adjust the weights to reduce the loss:

```python
dz3 = (A3 - train_y_np) / m
dw3 = np.dot(A2.T, dz3)
db3 = np.sum(dz3, axis=0, keepdims=True)

dz2 = np.dot(dz3, w3.T) * (A2 > 0)
dw2 = np.dot(A1.T, dz2)
db2 = np.sum(dz2, axis=0, keepdims=True)
```

The gradients flow backward through the network. For ReLU layers, we multiply by `(A > 0)`, which zeroes out gradients where the activation was negative during the forward pass. This is the derivative of ReLU.

We also clip the gradients to prevent them from becoming too large:

```python
np.clip(dw1, -1.0, 1.0, out=dw1)
np.clip(db1, -1.0, 1.0, out=db1)
```

When I ran this training cell some of the activation values exploded due to large values of gradients, exponential function in sofmax made them even big and pushed them as inf and this created nan values in Z3.
Gradient clipping is a safety measure that prevents training instability when gradients occasionally spike to very large values.

Finally, we update the weights using these gradients:

```python
w1 = w1 - alpha * dw1
b1 = b1 - alpha * db1
```

The learning rate `alpha = 0.0001` controls how big our steps are. A small learning rate means we move slowly but carefully toward better weights.

## Implementation 2: Dense Neural Network with Keras

Now that we understand how neural networks work internally, let's see how Keras makes this much simpler. We can build a similar network in just a few lines:

```python
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[784]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])
```

This network is deeper than our from-scratch version, with a first hidden layer of 512 nodes. But notice we've added two new techniques: Dropout and Batch Normalization.

Dropout randomly turns off some neurons during training:

```python
layers.Dropout(0.3)
```

This forces the network to learn robust features that don't depend on any single neuron. It's a form of regularization that helps prevent overfitting. The value 0.3 means 30% of neurons are randomly dropped during each training step.

Batch Normalization normalizes the activations between layers, similar to how we normalized the input data. This helps the network train faster and more stably.

### Training Configuration

We compile the model with specific settings:

```python
model.compile(
    optimizer = 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

Adam is an optimizer that adapts the learning rate for each weight individually, which usually works better than the simple gradient descent we used in our from-scratch implementation. The loss function `sparse_categorical_crossentropy` is the same cross-entropy we used before, but it works directly with integer labels instead of one-hot encoded labels.

We also use early stopping to prevent overfitting:

```python
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)
```

This monitors the validation loss during training. If it doesn't improve by at least 0.001 for 5 consecutive epochs, training stops and the best weights are restored. This way, we don't have to guess the perfect number of epochs—the network stops training when it's not improving anymore.

The simple dense network achieved around 97.6% accuracy on the test set, which is already better than KNN.

## Implementation 3: Convolutional Neural Network

While dense networks treat each pixel independently, convolutional networks understand the spatial relationships between pixels. This is crucial for image recognition because nearby pixels are often related.

### Basic CNN Architecture

```python
model = keras.Sequential([
    layers.Input(shape=[784]),
    layers.Reshape((28,28,1)),
    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

First, we reshape our flat 784-dimensional input back into a 28x28 image with one channel (grayscale). Then we apply convolutional layers that learn to detect local patterns like edges and curves.

The `Conv2D(32,3)` creates 32 different filters, each 3x3 pixels. These filters slide across the image, looking for specific patterns. Early layers might detect simple edges, while deeper layers detect more complex shapes like loops and curves—features that are characteristic of handwritten digits.

MaxPooling reduces the spatial dimensions by taking the maximum value in each 2x2 region. This makes the network more robust to small shifts in the digit's position and reduces the number of parameters.

After the convolutional layers extract features, we flatten the result and pass it through dense layers for the final classification. This basic CNN achieved around 98.8% accuracy, a significant improvement over the dense network.

## Implementation 4: Improved CNN with Data Augmentation

To push accuracy even higher, we need to make several architectural improvements and use data augmentation.

### Better Architecture Design

Instead of using activation directly in the Conv2D layer, we separate them with batch normalization:

```python
layers.Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer='he_normal'),
layers.BatchNormalization(),
layers.Activation('relu'),
```

This pattern (convolution, normalize, activate) is more effective than the simple approach. We set `use_bias=False` because batch normalization includes its own bias term, so having both would be redundant. The He initialization (`he_normal`) is specifically designed for ReLU activations.

We also use more filters progressively (32, 64, 128), allowing the network to learn increasingly complex features. Using `padding='same'` keeps the spatial dimensions the same after convolution, which gives us more control over when dimensions reduce.

### Data Augmentation

Real-world handwritten digits have natural variations—they might be slightly rotated or shifted. We can make our network more robust by artificially creating these variations during training:

```python
layers.Rescaling(1./255),
layers.RandomRotation(0.06),
layers.RandomTranslation(0.06, 0.06),
```

The random rotation (about 3-4 degrees) and translation (6% of image size) create slightly different versions of each training image. This effectively gives us much more training data without collecting more samples. The network learns to recognize digits regardless of these small variations.

### Global Average Pooling

Instead of flattening all the features, we use global average pooling:

```python
layers.GlobalAveragePooling2D(),
```

This takes the average of each feature map, dramatically reducing the number of parameters while maintaining good performance. Fewer parameters means less chance of overfitting.

### Learning Rate Schedule

We add a callback that reduces the learning rate when training plateaus:

```python
keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6)
```

If validation loss doesn't improve for 3 epochs, the learning rate is cut in half. This allows the network to make finer adjustments as it gets closer to optimal weights. Starting with a higher learning rate lets us explore the weight space quickly, then reducing it helps us converge precisely.

This improved architecture achieved around 99.35% accuracy, getting closer to human-level performance on MNIST.

## Implementation 5: Ensemble of CNNs

A powerful technique to squeeze out even more accuracy is ensembling—training multiple models and averaging their predictions.

### Why Ensembling Works

Each model, even with the same architecture, learns slightly different patterns because of random weight initialization and the stochastic nature of training. By averaging predictions from multiple models, we get a more robust classifier that makes fewer mistakes:

```python
n_folds = 3
preds_list = []

for i in range(n_folds):
    seed = base + i
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model = cnn_model()
    # ... train model ...
    probs = model.predict(X_test, verbose=0)
    preds_list.append(probs)

avg_probs = np.mean(np.stack(preds_list, axis=0), axis=0)
final_preds = np.argmax(avg_probs, axis=1)
```

We train three separate models with different random seeds. Each model outputs probability distributions over the 10 digit classes. We average these probability distributions and then take the most likely class. This ensemble approach achieved 99.47% accuracy with just three models, showing that diversity in predictions helps reduce errors.

The architecture for the ensemble is even deeper than before, with multiple convolutional layers per block and additional depth through strided convolutions. More augmentation (including zoom) and dropout help each model learn different aspects of the data.

## Implementation 6: Using PyTorch

Finally, we implemented a similar network using PyTorch, which gives us more low-level control over the training process.

### Defining the Model

In PyTorch, we define models as classes:

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)
```

This structure is similar to Keras's Sequential model, but we explicitly define the forward pass. PyTorch gives us more flexibility in how we structure our code and compute gradients.

### Training Loop

Unlike Keras where training happens with `model.fit()`, in PyTorch we write the training loop explicitly:

```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
```

This gives us complete control. We zero the gradients, do a forward pass, compute the loss, backpropagate, and update weights. The explicit nature of PyTorch makes it easier to debug and customize training, though it requires more code than Keras.

We also explicitly move data to GPU with `.to(device)` if available, which speeds up training significantly. The PyTorch implementation achieved 97.75% accuracy, showing that the framework choice matters less than the architecture and training process.

## Key Takeaways

We've progressively built more sophisticated models, going from a simple neural network implemented from scratch to ensemble CNNs achieving near-perfect accuracy. Each step introduced new concepts—from basic backpropagation to convolutional layers, batch normalization, data augmentation, and ensembling. The journey from 97% (dense network) to 99.47% (ensemble) required increasingly advanced techniques, but the fundamental principles remained the same: learn patterns from data through gradient descent.

Understanding these implementations from the ground up—starting with our NumPy implementation—helps demystify deep learning and shows that modern frameworks like Keras and PyTorch are ultimately doing the same mathematical operations we coded ourselves, just much more efficiently and with many optimizations.
