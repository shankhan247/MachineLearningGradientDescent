from your_code import GradientDescent, load_data
from your_code import ZeroOneLoss
import matplotlib.pyplot as plt
import numpy as np
print('Starting example experiment')

train_features, test_features, train_targets, test_targets = \
    load_data('synthetic')
N,d = train_features.shape
bias = np.ones((N,1))
features = np.append(train_features,bias,1)
print(N)
w = np.ones((d))
bias = np.array([0.5,-0.5,-1.5,-2.5,-3.5,-4.5,-5.5])
loss = []
print(w.shape)
print(features.shape)
for b in bias:
    zeroloss = ZeroOneLoss()
    losses = zeroloss.forward(features,np.append(w,b),train_targets)
    loss.append(losses)
loss2 = []
new_features = features[0:4]
new_targets = train_targets[0:4]
for b in bias:
    losses = zeroloss.forward(new_features,np.append(w,b),new_targets)
    loss2.append(losses)

plt.plot(bias,loss, label="all points")
plt.plot(bias,loss2,label="4 points")
plt.xlabel("Bias")
plt.ylabel("loss")
plt.legend(loc = "best")
plt.show()    

