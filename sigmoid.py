def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache
    