def cost_gradient(AL, Y):
    """
    Compute the cost gradient.

    Arguments:
    AL -- probability vector corresponding to your label predictions (i.e. Y hat), shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Output:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)

    """
    
    dZ = AL - Y

    return dZ