from your_code import GradientDescent, load_data
import matplotlib.pyplot as plt
import numpy as np
print('Starting example experiment')

train_features, test_features, train_targets, test_targets = \
    load_data('mnist-binary')
lamda = np.array([0.001,0.01,0.1,1,10,100])
count = 0
counts = []
for l in lamda:
    learner = GradientDescent(loss='squared', regularization="l1", learning_rate=0.00001, reg_param=l)
    learner.fit(train_features, train_targets, max_iter=2000)
    for c in learner.model:
        if c != 0:
            count = count + 1
    counts.append(count)
    count = 0
    print(l)

count = 0
counts2 = []
for l in lamda:
    learner = GradientDescent(loss='squared', regularization="l2", learning_rate=0.00001, reg_param=l)
    learner.fit(train_features, train_targets, max_iter=2000)
    for c in learner.model:
        if c != 0:
            count = count + 1
    counts2.append(count)
    count = 0
    print(l)

plt.plot(lamda,counts,label = "l1")
plt.plot(lamda, counts2, label = "l2")
plt.xlabel("lamda")
plt.ylabel("Non-zero counts")
plt.legend(loc = 'best')
plt.show()