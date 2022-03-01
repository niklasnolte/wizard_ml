def argmax(vals, mask):
    argmax = -1
    for i in range(len(vals)):
        if mask[i]:
            if argmax == -1 or vals[i] > vals[argmax]:
                argmax = i
    assert argmax != -1
    return argmax
