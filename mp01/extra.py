import numpy as np

def estimate_geometric(PX):
    '''
    @param:
    PX (numpy array of length cX): PX[x] = P(X=x), the observed probability mass function

    @return:
    p (scalar): the parameter of a matching geometric random variable
    PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a
      geometric random variable such that E[Y]=E[X].
    '''
    # raise RuntimeError("You need to write this")

    lengthcX = np.shape(PX)[0]

    # calculate E[X]
    meanofX = 0
    for i in range(lengthcX):
        meanofX += i*PX[i]
    
    # calculate p
    p = 1/(1 + meanofX)

    PY = []
    # find and populate geometric distribution
    for j in range(lengthcX):
        PY.append(p*np.power(1-p, j))

    # turn PY into numpy array
    PY = np.array(PY)

    return p, PY
