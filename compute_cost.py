def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions (i.e. Y hat), shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = -1/m * np.sum((Y * np.log(AL)) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)  # make sure shape is what is expected (e.g. [[666]] -> 666).

    return cost