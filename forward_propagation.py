# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# tensor flow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# create the dataset
#https://cs231n.github.io/neural-networks-case-study/
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# visualization
X, y = spiral_data(100,2)

class_1 = np.where(y == 0)
class_2 = np.where(y == 1)
# class_3 = np.where(y == 2)

plt.scatter(X[class_1, 0], X[class_1, 1], c='r', marker='o', s=40, )
plt.scatter(X[class_2, 0], X[class_2, 1], c='b', marker='x', s=50, )
# plt.scatter(X[class_3, 0], X[class_3, 1], c='g', marker='^', s=60, )

# two test examples represented in green stars
plt.scatter(-0.125, 0, marker='*', c = 'g', s = 100)
plt.scatter(0.125, 0, marker='*', c = 'g', s = 100)
plt.show()

print(X.shape)
print(y.shape)

"""## Tensorflow"""

tf.random.set_seed(1234)

model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(4, activation='relu', name='layer_1'),
        Dense(1, activation='sigmoid', name='layer_2')
    ]
)

model.summary()

W1, b1 = model.get_layer("layer_1").get_weights()
W2, b2 = model.get_layer("layer_2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    X,y,
    epochs=10,
)

W1, b1 = model.get_layer("layer_1").get_weights()
W2, b2 = model.get_layer("layer_2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

X_test = np.array([
    [-0.125, 0],  # positive example
    [0.125,0]])   # negative example

predictions = model.predict(X_test)
print("predictions made by Tensorflow = \n", predictions)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

"""## Numpy"""

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

g = sigmoid

def my_dense(a_in, W, b, g):
  """
    Computes dense layer
    Args:
      a_in (ndarray (m, n)) : Data, m examples by n features
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units [n rows and j columns]
      b    (ndarray (j, )) : bias vector, j units
    Returns
      a_out (ndarray (j,))  : j units|
    """

  """
  units = W.shape[1]
  a_out = np.zeros(units)
  for j in range(units):
    w = W[:,j]
    z = np.dot(w, a_in) + b[j]
    a_out[j] = g(z)
  """

  z = np.dot(a_in, W) + b
  a_out = g(z)
  return(a_out)

def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1, ReLU)
    a2 = my_dense(a1, W2, b2, sigmoid)
    return(a2)

def my_predict(X, W1, b1, W2, b2):
    """
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    """

    p = my_sequential(X, W1, b1, W2, b2)
    return(p)

# compare the results with TensorFlow
X_test = np.array([[-0.125, 0],  # positive example
                   [0.125,0]])   # negative example

predictions = my_predict(X_test, W1, b1, W2, b2)
print("predictions made by Numpy = \n", predictions)

