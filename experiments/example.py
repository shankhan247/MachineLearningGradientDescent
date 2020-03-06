from your_code import GradientDescent, load_data
import matplotlib.pyplot as plt
import numpy as np
print('Starting example experiment')

train_features, test_features, train_targets, test_targets = \
    load_data('mnist-binary')
print(test_features)
print(test_targets)
loss_init = 0
count = 0
converged = 0
losses = []
accuracies = []
while count <= 1000 and converged == 0:
    learner = GradientDescent(loss='hinge', learning_rate=0.0001)
    learner.fit(train_features, train_targets, batch_size=1, max_iter=1000)
    loss_new = learner.loss_epoch
    losses.append(loss_new)
    loss_diff = loss_new - loss_init
    loss_abs = np.absolute(loss_diff)
    if loss_abs < 0.001:
        converged = 1
    accuracy = learner.accuracy_epoch
    accuracies.append(accuracy)
    count = count + 1
    loss_init = loss_new
    print(loss_abs)

iterations = np.arange(1,count+1)
plt.plot(iterations,accuracies,label="accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend(loc = 'best')
plt.show()
"""
N = len(losses)
iterations = np.arange(1,N+1)
plt.plot(iterations,losses,label="loss")
#plt.plot(iterations,accuracies,label="accuracy")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(loc = 'best')
plt.show()
"""
print('Finished example experiment')
