## Generalized linear models with TensorFlow

Since Scikit-learn is missing some features when it comes to GLMs, I am hacking together a customizable GLM framework based on TensorFlow.

The goal is to have a tested and flexible framework that supports all general applications such as

- link function,
- loss function,
- regularization,
- choice of optimization routine,
- input/output type,
- convergence monitoring.

### Example: simple linear regression

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
#toy model
x = np.random.uniform(0,10,100)
w = 10
b = -5
noise = np.random.normal(scale=15,size=100)
y = w * x +b + noise
```

```python
from GLM import model 

inreg = model.GLMBase(x[:,np.newaxis],y[:,np.newaxis],
                activation=None,
                loss=tf.losses.mean_squared_error,
                optim= tf.train.AdamOptimizer(learning_rate=.1))

linreg.fit(n_steps=1000, minibatches=False)
x_test = np.random.uniform(0, 10, size=100)
predicted = linreg.predict(x_test)

fig = plt.figure()
plt.scatter(x_test ,predicted, s=10)
plt.scatter(x,y, alpha=.7, c="red", s=10)
```

![](https://raw.githubusercontent.com/mmontana/TensorFlowGLM/master/img/linear_regression.png)

### Example: regularized softmax regression

