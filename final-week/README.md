# Final Week: Custom Convolutional Neural Network

This folder contains the culmination of my learning journey through the WiDS project. After exploring K-Nearest Neighbors and various neural network architectures, I implemented a custom Convolutional Neural Network (CNN) from fundamental building blocks to deeply understand how CNNs actually work under the hood.

## Overview

The main file `digit-classifier-pytorch-cnn.ipynb` implements a CNN for handwritten digit classification that achieves **99.225% test accuracy** on the MNIST dataset. What makes this implementation special is that I built the core convolutional and pooling layers from scratch using PyTorch's `unfold` operation, rather than simply using the pre-built `nn.Conv2d` and `nn.MaxPool2d` layers.

## Why Custom Layers?

While PyTorch provides high-level convolutional layers, implementing them myself helped me understand exactly what happens during convolution. Instead of treating convolution as a black box, I learned how it transforms input images into feature maps through sliding window operations and matrix multiplications.

## Architecture Design

The network follows a classic CNN pattern with repeated convolutional blocks that progressively increase the number of feature channels while reducing spatial dimensions. Here's the overall structure:

```
Input (1 x 28 x 28)
    ↓
[Conv 3x3 (32) → BatchNorm → ReLU → Conv 3x3 (32) → BatchNorm → ReLU → MaxPool]
    ↓
[Conv 3x3 (64) → BatchNorm → ReLU → Conv 3x3 (64) → BatchNorm → ReLU → MaxPool]
    ↓
[Conv 3x3 (128) → BatchNorm → ReLU → MaxPool → AdaptiveAvgPool]
    ↓
[Flatten → Linear (128) → BatchNorm → ReLU → Dropout → Linear (10)]
```

This architecture embodies several important design principles. First, it uses paired convolutional layers before each pooling operation, which allows the network to learn richer features at each spatial resolution. Second, the number of channels doubles after each pooling layer, compensating for the reduced spatial dimensions by increasing the depth of representation. Third, it incorporates modern techniques like batch normalization and dropout for stable training and better generalization.

## Custom Layer Implementation

### Custom Convolution Layer

The heart of this implementation is the custom `MyConv2d` layer. Rather than using PyTorch's built-in convolution, I implemented it using the `unfold` operation, which extracts sliding window patches from the input:

```python
class MyConv2d(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        kh, kw = self.kernel_size
        
        # Extract sliding window patches from the input
        # This transforms each spatial location into a column
        patches = F.unfold(x, kernel_size=(kh, kw), 
                          padding=self.padding, 
                          stride=self.stride)
        # patches shape: (B, C*kh*kw, L) where L is number of positions
        
        # Reshape weights to matrix form
        w = self.weight.view(self.out_channels, -1)  # (out_channels, C*kh*kw)
        
        # Convolution becomes matrix multiplication
        out = torch.matmul(w.unsqueeze(0).expand(B, -1, -1), patches)
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        
        # Calculate output dimensions and reshape
        H_out = (H + 2*self.padding - kh) // self.stride + 1
        W_out = (W + 2*self.padding - kw) // self.stride + 1
        
        return out.view(B, self.out_channels, H_out, W_out)
```

The `unfold` operation is the key innovation here. It takes an image and extracts all the patches that a convolutional kernel would slide over, arranging them as columns in a matrix. This transforms the convolution operation into a simple matrix multiplication, making it both conceptually clearer and computationally efficient.

### Custom Max Pooling Layer

Similarly, I implemented max pooling from scratch using the same unfold approach:

```python
class MyMaxPool2d(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        kh, kw = self.kernel_size
        
        # Extract patches for each pooling window
        patches = F.unfold(x, kernel_size=(kh, kw), 
                          padding=self.padding, 
                          stride=self.stride)
        
        # Reshape to separate channels and spatial positions within each patch
        patches = patches.view(B, C, kh*kw, -1)
        
        # Take maximum across the spatial dimension of each patch
        pooled, _ = patches.max(dim=2)  # (B, C, L)
        
        # Calculate output dimensions and reshape
        H_out = (H + 2*self.padding - kh) // self.stride + 1
        W_out = (W + 2*self.padding - kw) // self.stride + 1
        
        return pooled.view(B, C, H_out, W_out)
```

Max pooling downsamples the feature maps by taking the maximum value within each pooling window. This provides two important benefits: it makes the features somewhat invariant to small translations in the input, and it reduces the computational cost of subsequent layers by shrinking the spatial dimensions.

## Training Strategy

The training process incorporates several techniques that I learned throughout this project to achieve stable, high-performance results.

### Data Preparation

The data preparation is straightforward but important. I reshape the flattened pixel vectors into proper 28x28 images and normalize pixel values to the range zero to one:

```python
# Reshape to image format: (N, 1, 28, 28)
X = X.values.astype(np.float32) / 255.0

# Split into 90% training, 10% validation
full_ds = ImgDataset(X, y)
val_size = int(0.1 * len(full_ds))
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))
```

Normalizing the pixel values is crucial because it helps gradient descent converge more quickly. When all input features are on a similar scale, the optimization landscape becomes smoother and easier to navigate.

### Optimizer and Learning Rate

I used the Adam optimizer with an initial learning rate of 0.001 and a learning rate scheduler that reduces the rate when validation loss plateaus:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3, min_lr=1e-6
)
```

The ReduceLROnPlateau scheduler automatically detects when learning has stalled and reduces the learning rate by half. This allows the network to make large updates early in training when far from the optimum, then fine-tune with smaller updates as it approaches convergence. The weight decay parameter adds L2 regularization to prevent overfitting.

### Training Loop with Validation

The training loop tracks both training and validation performance, saving the best model based on validation loss:

```python
best_val = float('inf')
best_state = None

for epoch in range(epochs):
    # Training phase
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            val_loss += criterion(logits, yb).item() * xb.size(0)
    
    # Save best model
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

Gradient clipping prevents exploding gradients by capping the norm of gradients at 5.0. This is particularly important in the early stages of training when gradients can occasionally become very large. The model saving strategy ensures we use the version that performed best on unseen validation data, not just the final epoch.

## Regularization Techniques

### Batch Normalization

Every convolutional and fully connected layer is followed by batch normalization:

```python
MyConv2d(32, 64, 3, padding=1, bias=False),
nn.BatchNorm2d(64),
nn.ReLU(inplace=True),
```

Batch normalization normalizes the inputs to each layer, which stabilizes training by reducing internal covariate shift. This allows us to use higher learning rates and makes the network less sensitive to weight initialization. Note that I disabled bias in the convolutional layers because batch normalization has its own bias parameter, making the convolutional bias redundant.

### Dropout

The classifier includes dropout with probability 0.3:

```python
nn.Flatten(),
nn.Linear(128, 128, bias=False),
nn.BatchNorm1d(128),
nn.ReLU(inplace=True),
nn.Dropout(0.3),
nn.Linear(128, 10)
```

During training, dropout randomly sets 30% of the activations to zero, forcing the network to learn redundant representations. This prevents the network from relying too heavily on any single neuron and improves generalization to new examples.

## Key Hyperparameters

The final hyperparameters were chosen through experimentation:

- **Batch size**: 128 (balances GPU memory usage and gradient stability)
- **Epochs**: 12 (sufficient for convergence with early stopping)
- **Learning rate**: 0.001 (standard for Adam optimizer)
- **Weight decay**: 0.0001 (mild L2 regularization)
- **Dropout rate**: 0.3 (moderate regularization)
- **Gradient clipping**: 5.0 (prevents gradient explosion)

## Results

The model achieves excellent performance on the MNIST test set:

- **Training accuracy**: ~99.9%
- **Validation accuracy**: ~99.1%
- **Test accuracy**: **99.225%**
