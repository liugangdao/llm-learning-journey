"""
mini 神经网络框架
1. Tensor 包装numpy数组，记录前向值和反向传播的梯度
2. Layer 每种层都继承基类，包含forward / backward
3. Model 顺序组合层，负责前向传播和反向传播
4. Loss 自定义损失函数
5. Optimizer 例如SGD，用于更新参数

"""
from sklearn.model_selection import train_test_split
try:
    import cupy as np
except ImportError:
    import numpy as np

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


if __name__=="__main__":
    # 创建数据
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    X = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
    y = np.genfromtxt(url, delimiter=',', dtype='int', usecols=[4], converters={4: lambda x: 0 if x == b'Iris-setosa' else 1 if x == b'Iris-versicolor' else 2})
    print(X[:4,])
    print(y[:4])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # 创建模型
    model = Sequential([
        Linear(4, 64),
        ReLU(),
        Linear(64,3)
        ])
    
    # 创建损失函数
    criterion = SoftmaxCrossEntropyLoss()
    # 创建优化器
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(300):
        # 前向传播
        logits = model.forward(X_train)
        loss = criterion.forward(logits, y_train)
        # 反向传播
        grad = criterion.backward()
        model.backward(grad)
        # 更新参数
        optimizer.step()
        # 打印损失
        if epoch % 10 == 0:
            pred = np.argmax(model.forward(X_test), axis=1)
            acc = np.mean(pred == y_test)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Test Accuracy={acc:.4f}")

