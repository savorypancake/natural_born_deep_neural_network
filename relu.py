def relu(Z):
    A = max(0.0, Z)
    cache = Z
    return A, cache