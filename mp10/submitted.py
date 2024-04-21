'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    # raise RuntimeError("You need to write this part!")

    # initialize transition matrix 
    P = np.zeros((model.M, model.N, 4, model.M, model.N))

    ''' Things to Watch For:
    - terminal states (r, c) has P[r, c, :, :, :] = 0
    - if move is out of bounds or hits wall, remain in current cell with that probability of the move 
    '''

    # loop through each (r, c)
    for r in range (model.M):
        for c in range (model.N):
            # loop through each action (0 (left), 1 (up), 2 (right), 3 (down))
            for action in range (4):
                # only make updates for non-terminal states
                if model.TS[r, c] == False:
                    if action == 0:
                        # intended move left model.D(r, c, 0); check if move is out of bounds or is a wall
                        if c - 1 < 0 or model.W[r, c - 1] == True:
                            P[r, c, action, r, c] += model.D[r, c, 0]
                        else:
                            P[r, c, action, r, c - 1] = model.D[r, c, 0]

                        # unintended move down model.D(r, c, 1); check if move is out of bounds or is a wall
                        if r + 1 >= model.M or model.W[r + 1, c] == True:
                            P[r, c, action, r, c] += model.D[r, c, 1]
                        else:
                            P[r, c, action, r + 1, c] = model.D[r, c, 1]

                        # unintended move up model.D(r, c, 2); check if move is out of bounds or is a wall
                        if r - 1 < 0 or model.W[r - 1, c] == True:
                            P[r, c, action, r, c] += model.D[r, c, 2]
                        else:
                            P[r, c, action, r - 1, c] = model.D[r, c, 2]
                    
                    elif action == 1:
                        # intended move up model.D(r, c, 0); check if move is out of bounds or is a wall
                        if r - 1 < 0 or model.W[r - 1, c] == True:
                            P[r, c, action, r, c] += model.D[r, c, 0]
                        else:
                            P[r, c, action, r - 1, c] = model.D[r, c, 0]
                        
                        #unintended move left model.D(r, c, 1); check if move is out of bounds or is a wall
                        if c - 1 < 0 or model.W[r, c - 1] == True:
                            P[r, c, action, r, c] += model.D[r, c, 1]
                        else:
                            P[r, c, action, r, c - 1] = model.D[r, c, 1]
                        
                        # unintended move right model.D(r, c, 2); check if move is out of bounds or is a wall
                        if c + 1 >= model.N or model.W[r, c + 1] == True:
                            P[r, c, action, r, c] += model.D[r, c, 2]
                        else:
                            P[r, c, action, r, c + 1] = model.D[r, c, 2]

                    elif action == 2:
                        # intended move right model.D(r, c, 0); check if move is out of bounds or is a wall
                        if c + 1 >= model.N or model.W[r, c + 1] == True:
                            P[r, c, action, r, c] += model.D[r, c, 0]
                        else:
                            P[r, c, action, r, c + 1] = model.D[r, c, 0]
                        
                        #unintended move up model.D(r, c, 1); check if move is out of bounds or is a wall
                        if r - 1 < 0 or model.W[r - 1, c] == True:
                            P[r, c, action, r, c] += model.D[r, c, 1]
                        else:
                            P[r, c, action, r - 1, c] = model.D[r, c, 1]
                        
                        # unintended move down model.D(r, c, 2); check if move is out of bounds or is a wall
                        if r + 1 >= model.M or model.W[r + 1, c] == True:
                            P[r, c, action, r, c] += model.D[r, c, 2]
                        else:
                            P[r, c, action, r + 1, c] = model.D[r, c, 2]

                    elif action == 3:
                        # intended move down model.D(r, c, 0); check if move is out of bounds or is a wall
                        if r + 1 >= model.M or model.W[r + 1, c] == True:
                            P[r, c, action, r, c] += model.D[r, c, 0]
                        else:
                            P[r, c, action, r + 1, c] = model.D[r, c, 0]
                        
                        # unintended move right model.D(r, c, 1); check if move is out of bounds or is a wall
                        if c + 1 >= model.N or model.W[r, c + 1] == True:
                            P[r, c, action, r, c] += model.D[r, c, 1]
                        else:
                            P[r, c, action, r, c + 1] = model.D[r, c, 1]

                        # unintended move left model.D(r, c, 2); check if move is out of bounds or is a wall
                        if c - 1 < 0 or model.W[r, c - 1] == True:
                            P[r, c, action, r, c] += model.D[r, c, 2]
                        else:
                            P[r, c, action, r, c - 1] = model.D[r, c, 2]
                        
    return P

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    # raise RuntimeError("You need to write this part!")

    # initialize updated utility array
    U_next = np.zeros((model.M, model.N))

    # loop through each (r, c) cell
    for r in range (model.M):
        for c in range (model.N):
            # loop through all 4 possible actions from current (r, c) state
            for action in range (4):
                if action == 0:
                    # including possibility of staying in current cell
                    action0Sum = P[r, c, action, r, c]*U_current[r, c]
                    # intended move left 
                    if c - 1 >= 0 and model.W[r, c - 1] != True:
                        action0Sum += P[r, c, action, r, c - 1]*U_current[r, c - 1]
                    # unintended move down 
                    if r + 1 < model.M and model.W[r + 1, c] != True:
                        action0Sum += P[r, c, action, r + 1, c]*U_current[r + 1, c]
                    # unintended move up 
                    if r - 1 >= 0 and model.W[r - 1, c] != True:
                        action0Sum += P[r, c, action, r - 1, c]*U_current[r - 1, c]
                    
                    newMaxSum = action0Sum

                elif action == 1:
                    # including possibility of staying in current cell
                    action1Sum = P[r, c, action, r, c]*U_current[r, c]
                    # intended move up 
                    if r - 1 >= 0 and model.W[r - 1, c] != True:
                        action1Sum += P[r, c, action, r - 1, c]*U_current[r - 1, c]
                    #unintended move left 
                    if c - 1 >= 0 and model.W[r, c - 1] != True:
                        action1Sum += P[r, c, action, r, c - 1]*U_current[r, c - 1]
                    # unintended move right 
                    if c + 1 < model.N and model.W[r, c + 1] != True:
                        action1Sum += P[r, c, action, r, c + 1]*U_current[r, c + 1]

                    if action1Sum > newMaxSum:
                        newMaxSum = action1Sum

                elif action == 2:
                    # including possibility of staying in current cell
                    action2Sum = P[r, c, action, r, c]*U_current[r, c]
                    # intended move right
                    if c + 1 < model.N and model.W[r, c + 1] != True:
                        action2Sum += P[r, c, action, r, c + 1]*U_current[r, c + 1]
                    #unintended move up 
                    if r - 1 >= 0 and model.W[r - 1, c] != True:
                        action2Sum += P[r, c, action, r - 1, c]*U_current[r - 1, c]
                    # unintended move down 
                    if r + 1 < model.M and model.W[r + 1, c] != True:
                        action2Sum += P[r, c, action, r + 1, c]*U_current[r + 1, c]

                    if action2Sum > newMaxSum:
                        newMaxSum = action2Sum

                elif action == 3:
                    # including possibility of staying in current cell
                    action3Sum = P[r, c, action, r, c]*U_current[r, c]
                    # intended move down
                    if r + 1 < model.M and model.W[r + 1, c] != True:
                        action3Sum += P[r, c, action, r + 1, c]*U_current[r + 1, c]
                    # unintended move right 
                    if c + 1 < model.N and model.W[r, c + 1] != True:
                        action3Sum += P[r, c, action, r, c + 1]*U_current[r, c + 1]
                    #unintended move left 
                    if c - 1 >= 0 and model.W[r, c - 1] != True:
                        action3Sum += P[r, c, action, r, c - 1]*U_current[r, c - 1]

                    if action3Sum > newMaxSum:
                        newMaxSum = action3Sum

            # calculate updated utility
            U_next[r, c] = model.R[r, c] + model.gamma*newMaxSum
    
    return U_next

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    # raise RuntimeError("You need to write this part!")

    # find the transition matrix 
    P = compute_transition(model)

    # initialize utility to be 0
    U = np.zeros((model.M, model.N))

    # continue to iterate value until convergence, or for 100 iterations, whichever comes first 
    for iteration in range (100):
        U_new = compute_utility(model, U, P)
        
        # check for convergence
        convergenceFlag = True
        for r in range (model.M):
            if convergenceFlag == False:
                break
            for c in range (model.N):
                if abs(U_new[r, c] - U[r, c]) >= epsilon:
                    convergenceFlag = False
                    break

        if convergenceFlag == True:
            break
        else:
            # create a copy to avoid simply assigning reference
            U = U_new.copy()
    
    return U

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    # raise RuntimeError("You need to write this part!")

    # initialize utility to be 0 
    U = np.zeros((model.M, model.N))
    U_next = np.zeros((model.M, model.N))

    # loop until convergence or 10000 iterations, whichever comes first
    for iteration in range (10000):
        for r in range (model.M):
            for c in range (model.N):
                U_next[r, c] = model.R[r, c]
                # loop through all possible next states
                for r_next in range (model.M):
                    for c_next in range (model.N):
                        U_next[r, c] += model.gamma * model.FP[r, c, r_next, c_next] * U[r_next, c_next]

        
        # check for convergence
        convergenceFlag = True
        for r in range (model.M):
            if convergenceFlag == False:
                break
            for c in range (model.N):
                if abs(U_next[r, c] - U[r, c]) >= epsilon:
                    convergenceFlag = False
                    break

        if convergenceFlag == True:
            break
        else:
            # create a copy to avoid simply assigning reference
            U = U_next.copy()

    return U

                
