# Chris Johnson 
# CS 3120 Sec 1
# Hw ! Gradient Descent


import numpy as np 
import matplotlib.pyplot as plt 

# data samples
x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
y_data = 2 * x_data + 50 + np.random.random()

# plot landscape
bias = np.arange(0, 100, 1) 
weight = np.arange(-5, 5, 0.1) 
loss_val = np.zeros((len(bias), len(weight)))

for i in range(len(bias)):
    for j in range(len(weight)):
        b = bias[i]
        w = weight[j]
        loss_val[j][i] = 0        
        for n in range(len(x_data)):
            loss_val[j][i] = loss_val[j][i] + (w*x_data[n]+b - y_data[n])**2 # this is the loss 
        loss_val[j][i] = loss_val[j][i]/len(x_data)

# build linnear regression model
initial_bias = 0
initial_weight = 0 

learning_rate = 0.0001
iteration = 20000

bias_history = [initial_bias]
weight_history = [initial_weight]

# model by gradient descent
for i in range(iteration):
    bias_gradient = 0.0
    weight_gradient = 0.0
    
    for n in range(len(x_data)):
        bias_gradient = bias_gradient + (initial_bias + initial_weight * x_data[n] - y_data[n]) * 1.0
        weight_gradient = weight_gradient + (initial_bias + initial_weight * x_data[n] - y_data[n]) * x_data[n]
    
    # learn rate
    initial_bias = initial_bias - bias_gradient * learning_rate
    initial_weight = initial_weight - weight_gradient * learning_rate
    
    # store
    bias_history.append(initial_bias)
    weight_history.append(initial_weight)

 
# convergence point
print(bias_history[-1])
print(weight_history[-1])

# plot
plt.plot(bias_history, weight_history, 'o-', ms = 3, lw = 1.5, color = 'black')
plt.xlim(0, 100)
plt.ylim(-5, 5)
plt.contour(bias, weight, loss_val, 50, alpha = 0.5, cmap = plt.get_cmap('jet'))
plt.show()


