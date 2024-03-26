import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(board, side, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(board, side, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (moveList, moveTree, value)
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (int or float): value of the board after making the chosen move
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(board, side, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return ([ move ], { encode(*move): {} }, value)
    else:
        return ([], {}, evaluate(board))

'''
Note
- move[0] = from, move[1] = to
'''
    
###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(board, side, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (moveList, moveTree, value)
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (float): value of the final board in the minimax-optimal move sequence
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    # raise NotImplementedError("you need to write this!")

    '''
    Possible API Calls/Useful Definitions
    - side == False is player 0 should play next
    - move == [fro, to, promote], fro == [from_x, from_y], to == [to_x, to_y], promote = None or "q" (promote to queen)
    - piece == [x, y, type]
    - chess.lib.heuristics.evaluate(board) returns hueristic value of the board for the white player (Max)
    - encode([from_x, from_y], [to_x, to_y], promote)
    - chess.lib.convertMoves, generates starting board
    - submitted.generateMoves(board, side, flags), generates all moves legal on the current board
    - chess.lib.makeMove(side, board, [from_x, from_y], [to_x, to_y], flags, promote), returns newside, newboard, newflags

    Other Notes
    - player 0 white (goes first), player 1 black 
    - side == False if player 0 plays next; side == True if player 1 plays next
    '''

    # base case, when depth is 0, evaluate the current board state, (no more moves to take, so return empty moveList and moveTree)
    if depth == 0:
        return [], {}, chess.lib.heuristics.evaluate(board)
    
    # initialize empty movelist list and empty moveTree dictionary
    moveList = []
    moveTree = {}

    # Max/White player next; choose a path through this tree that maximizes the heuristic value of the final board
    if side == False:
        # initialize maxValue to negative infinity
        maxValue = float('-inf')
        # loop through all possible moves from current board condition
        for maxMove in generateMoves(board, side, flags):
            # make the move and recursively call minimax to explore all possible future moves/states
            newMaxSide, newMaxBoard, newMaxFlags = chess.lib.makeMove(side, board, maxMove[0], maxMove[1], flags, maxMove[2])
            rMaxList, rMaxTree, rMaxValue = minimax(newMaxBoard, newMaxSide, newMaxFlags, depth-1)
            # update moveTree with subtree returned from minimax call; key will be the move that generated the subtree
            moveTree[encode(maxMove[0], maxMove[1], maxMove[2])] = rMaxTree
            # update maxValue and the optimal path if rMaxValue from recursive call greater than current maxValue
            if rMaxValue > maxValue:
                maxValue = rMaxValue
                moveList = [maxMove] + rMaxList
        return moveList, moveTree, maxValue
    # Min/Black player next; choose a path through this tree that minimizes the heuristic value of the final board
    elif side == True:
        # initialize minValue to positive infinity
        minValue = float('inf')
        # loop through all possible moves from current board condition
        for minMove in generateMoves(board, side, flags):
            # make the move and recursively call minimax to explore all possible future moves/states
            newMinSide, newMinBoard, newMinFlags = chess.lib.makeMove(side, board, minMove[0], minMove[1], flags, minMove[2])
            rMinList, rMinTree, rMinValue = minimax(newMinBoard, newMinSide, newMinFlags, depth-1)
            # update moveTree with subtree returned from minimax call; key will be the move that generated the subtree
            moveTree[encode(minMove[0], minMove[1], minMove[2])] = rMinTree
            # update minValue and the optimal path if rMinValue from recursive call is less than current minValue
            if rMinValue < minValue:
                minValue = rMinValue
                moveList = [minMove] + rMinList
        return moveList, moveTree, minValue
  
def alphabeta(board, side, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (moveList, moveTree, value)
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (float): value of the final board in the minimax-optimal move sequence
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    # raise NotImplementedError("you need to write this!")

    '''
    Notes
    - min node can update beta; max node can update alpha
    - if beta ever falls below or equals alpha (if beta <= alpha), prune any remaining children and return
    '''
  
    # base case, when depth is 0, evaluate the current board state, (no more moves to take, so return empty moveList and moveTree)
    if depth == 0:
        return [], {}, chess.lib.heuristics.evaluate(board)
    
    # initialize empty movelist list and empty moveTree dictionary
    moveList = []
    moveTree = {}

    # Max/White player next; choose a path through this tree that maximizes the heuristic value of the final board
    if side == False:
        # initialize maxValue to negative infinity
        maxValue = float('-inf')
        # loop through all possible moves from current board condition
        for maxMove in generateMoves(board, side, flags):
            # make the move and recursively call alphabeta to explore all possible future moves/states
            newMaxSide, newMaxBoard, newMaxFlags = chess.lib.makeMove(side, board, maxMove[0], maxMove[1], flags, maxMove[2])
            rMaxList, rMaxTree, rMaxValue = alphabeta(newMaxBoard, newMaxSide, newMaxFlags, depth-1, alpha, beta)
            # update moveTree with subtree returned from alphabeta call; key will be the move that generated the subtree
            moveTree[encode(maxMove[0], maxMove[1], maxMove[2])] = rMaxTree
            # update maxValue and the optimal path if rMaxValue from recursive call greater than current maxValue
            if rMaxValue > maxValue:
                maxValue = rMaxValue
                moveList = [maxMove] + rMaxList

            # max node, so we need to update alpha based on the subtree return value 
            alpha = max(alpha, rMaxValue)
            # if beta ever falls below or equals alpha, then break and do not evaluate any more children/other possible moves
            if beta <= alpha:
                break
        return moveList, moveTree, maxValue
    # Min/Black player next; choose a path through this tree that minimizes the heuristic value of the final board
    elif side == True:
        # initialize minValue to positive infinity
        minValue = float('inf')
        # loop through all possible moves from current board condition
        for minMove in generateMoves(board, side, flags):
            # make the move and recursively call alphabeta to explore all possible future moves/states
            newMinSide, newMinBoard, newMinFlags = chess.lib.makeMove(side, board, minMove[0], minMove[1], flags, minMove[2])
            rMinList, rMinTree, rMinValue = alphabeta(newMinBoard, newMinSide, newMinFlags, depth-1, alpha, beta)
            # update moveTree with subtree returned from alphabeta call; key will be the move that generated the subtree
            moveTree[encode(minMove[0], minMove[1], minMove[2])] = rMinTree
            # update minValue and the optimal path if rMinValue from recursive call is less than current minValue
            if rMinValue < minValue:
                minValue = rMinValue
                moveList = [minMove] + rMinList
  
            # min node, so we need to update beta based on the subtree return value
            beta = min(beta, rMinValue)
            # if beta ever falls below or equals alpha, then break and do not evaluate any more children/other possible moves
            if beta <= alpha:
                break
        return moveList, moveTree, minValue
    
# FROM PAST SEMESTERS
# def stochastic(board, side, flags, depth, breadth, chooser):
#     '''
#     Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
#     Return: (moveList, moveTree, value)
#       moveLists (list): any sequence of moves, of length depth, starting with the best move
#       moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
#       value (float): average board value of the paths for the best-scoring move
#     Input:
#       board (2-tuple of lists): current board layout, used by generateMoves and makeMove
#       side (boolean): True if player1 (Min) plays next, otherwise False
#       flags (list of flags): list of flags, used by generateMoves and makeMove
#       depth (int >=0): depth of the search (number of moves)
#       breadth: number of different paths 
#       chooser: a function similar to random.choice, but during autograding, might not be random.
#     '''
#     raise NotImplementedError("you need to write this!")
