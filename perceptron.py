# prediction
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i+1]*row[i]
    return 1 if activation >= 0 else 0

# stochastic gradient descent
#def stoch_grad_desc(t_dataset, l_rate, n_epochs):
    #weights = [0 for i in range(len(train[0]))]
    #for ep in range(n_epochs):


