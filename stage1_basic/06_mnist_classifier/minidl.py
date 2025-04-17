"""
mini 神经网络框架
1. Tensor 包装numpy数组，记录前向值和反向传播的梯度
2. Layer 每种层都继承基类，包含forward / backward
3. Model 顺序组合层，负责前向传播和反向传播
4. Loss 自定义损失函数
5. Optimizer 例如SGD，用于更新参数

"""
import io
from sklearn.model_selection import train_test_split
import urllib.request
try:
    import cupy as np
    print("Using cupy")
except ImportError:
    import numpy as np
    print("Using numpy")

class Tensor:
    def __init__(self, data, name=None):
        self.data = data
        self.grad = np.zeros_like(data)
        self.creator = None
        self.name = name
    
    def zero_grad(self):
        self.grad.fill(0)

class Layer:
    def __init__(self):
        self.params = []
        self.grads = []
    
    def __call__(self, *inputs):
        return self.forward(*inputs)
    
    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        for layer in self.layers:
            for p in getattr(layer, 'params', []):
                yield p

class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.W = Tensor(np.random.randn(in_features, out_features) * 0.01, name='W')
        self.b = Tensor(np.zeros((1, out_features)), name='b')
        self.params = [self.W, self.b]
        self.grads = [self.W.grad, self.b.grad]

    def forward(self, x):
        self.input = x
        return x @ self.W.data + self.b.data
    
    def backward(self, grad):
        self.W.grad += self.input.T @ grad
        self.b.grad += np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.data.T

class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask
    
class SoftmaxCrossEntropyLoss:
    def forward(self, logits, targets):
        self.targets = targets
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)

        return -np.mean(np.log(self.probs[np.arange(len(self.probs)), targets]))
    
    def backward(self):
        m = self.targets.shape[0]
        grad = self.probs.copy()
        grad[np.arange(m), self.targets] -= 1
        return grad / m
    
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
            param.zero_grad()

# 卷积网络
class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size,tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        kh, kw = self.kernel_size
        self.W = Tensor(np.random.randn(out_channels, in_channels, kh, kw) * 0.01, name='W')
        self.b = Tensor(np.zeros((1, out_channels, )), name='b')
        
        self.params = [self.W, self.b]
        self.grads = [self.W.grad, self.b.grad]

    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape
        F, _, kh,kw = self.W.data.shape
        pad_h, pad_w = self.padding, self.padding
        out_h = (H + 2 * pad_h - kh) // self.stride + 1
        out_w = (W + 2 * pad_w - kw) // self.stride + 1
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        out = np.zeros((N, F, out_h, out_w))

        for n in range(N):
            for f in range(F):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = x_pad[n, :, h_start:h_start+kh, w_start:w_start+kw]
                        out[n, f, i, j] = np.sum(region * self.W.data[f]) + self.b.data[0, f]
        return out
    
    def backward(self, grad):
        x = self.input
        N, C, H, W = x.shape
        F, _, kh, kw = self.W.data.shape

        pad_h, pad_w = self.padding, self.padding
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        dx_pad = np.zeros_like(x_pad, dtype=np.float32)

        self.W.grad.fill(0)
        self.b.grad.fill(0)

        for n in range(N):
            for f in range(F):
                for i in range(H):
                    for j in range(W):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = x_pad[n, :, h_start:h_start+kh, w_start:w_start+kw]
                        self.W.grad[f] += grad[n, f, i, j] * region
                        self.b.grad[0, f] += grad[n, f, i, j]
                        dx_pad[n, :, h_start:h_start+kh, w_start:w_start+kw] +=  grad[n, f, i, j]*self.W.data[f]
        if self.padding > 0:
            return dx_pad[:, :, pad_h:-pad_h, pad_w:-pad_w]
        else:
            return dx_pad

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size,tuple) else (kernel_size, kernel_size)
        kh, kw = self.kernel_size
        self.stride = stride or kh

    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        out_h = (H - kh) // self.stride + 1
        out_w = (W - kw) // self.stride + 1

        out = np.zeros((N, C, out_h, out_w))
        self.max_indices = {}

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        region = x[n, c, h_start:h_start+kh, w_start:w_start+kw]
                        out[n, c, i, j] = np.max(region)
                        self.max_indices[(n, c, i, j)] = (h_start, w_start,*np.unravel_index(np.argmax(region), region.shape))

        return out
    
    def backward(self, grad):
        N, C, H, W = self.input.shape
        kh, kw = self.kernel_size
        dx = np.zeros_like(self.input)

        for key, (h_start, w_start, h_idx, w_idx) in self.max_indices.items():
            n, c, i, j = key
            dx[n, c, h_start+h_idx, w_start+w_idx] += grad[n, c, i, j]
        return dx
    
class Flatten(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(self.input_shape[0], -1)
    
    def backward(self, grad):
        return grad.reshape(self.input_shape)

class DataLoader:
    def __init__(self, x, y, batch_size, shuffle=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(x)

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            self.indices = np.random.permutation(self.num_samples)
        else:
            self.indices = np.arange(self.num_samples)
        return self
    
    def __next__(self):
        if self.idx >= self.num_samples:
            raise StopIteration
        
        batch_indices = self.indices[self.idx:self.idx+self.batch_size]
        x_batch = self.x[batch_indices]
        y_batch = self.y[batch_indices]
        self.idx += self.batch_size
        return x_batch, y_batch

if __name__ == '__main__':
    # 创建数据
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    with urllib.request.urlopen(url) as response:
        npz_data = response.read()

    with np.load(io.BytesIO(npz_data)) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']
    print(train_examples[:5], train_labels[:5],test_examples[:5], test_labels[:5])
    # 数据预处理
    train_examples = train_examples.reshape(-1, 1, 28, 28)

    # 数据batch
    train_examples, train_labels = train_examples[:1000], train_labels[:1000]
    train_examples, valid_examples, train_labels, valid_labels = train_test_split(train_examples, train_labels, test_size=0.1, random_state=42)
    
    

    # 创建模型
    model = Sequential([
        Conv2D(1,8,kernel_size=3,stride=1,padding=1),
        ReLU(),
        MaxPool2D(2),
        Conv2D(8,16,kernel_size=3,stride=1,padding=1),
        ReLU(),
        MaxPool2D(2),
        Flatten(),
        Linear(16*7*7,10)])
    
    # 定义损失函数和优化器
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001)

    # 训练模型
    from tqdm import tqdm
    for epoch in range(300):
        for x_batch, y_batch in tqdm(DataLoader(train_examples, train_labels, batch_size=8, shuffle=True)):
            with np.cuda.Device(0):
                x_batch = np.asarray(x_batch)
                y_batch = np.asarray(y_batch)
                logits = model.forward(x_batch)
                loss = loss_fn.forward(logits, y_batch)
                grad = loss_fn.backward()
                model.backward(grad)
                optimizer.step()
                if epoch % 10 == 0:
                    pred = np.argmax(model.forward(valid_examples), axis=1)
                    acc = np.mean(pred == valid_labels)
                    print(f"Epoch {epoch}: Loss={loss:.4f}, Test Accuracy={acc:.4f}")

        