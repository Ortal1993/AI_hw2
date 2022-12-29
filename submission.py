import random

import numpy as np

import Gobblet_Gobblers_Env as gge

not_on_board = np.array([-1, -1])


# agent_id is which player I am, 0 - for the first player , 1 - if second player
def dumb_heuristic1(state, agent_id):
    is_final = gge.is_final_state(state)
    # this means it is not a final state
    if is_final is None:
        return 0
    # this means it's a tie
    if is_final == 0:
        return -1
    # now convert to our numbers the win
    winner = int(is_final) - 1
    # now winner is 0 if first player won and 1 if second player won
    # and remember that agent_id is 0 if we are first player  and 1 if we are second player won
    if winner == agent_id:
        # if we won
        return 1
    else:
        # if other player won
        return -1


# checks if a pawn is under another pawn
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False


# count the numbers of pawns that i have that aren't hidden
# player1_pawns = "B1": (not_on_board, "B")
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    #print("dumb")
    if agent_id == 0:
        for key, value in state.player1_pawns.items():#key = pawn, value = (not_on_board, "B")
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                #print("here1")
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                #print("here2")
                sum_pawns += 1

    #print(sum_pawns)
    return sum_pawns
    
#state = the next board if the action that called "smart_heuristic" would be applied
def smart_heuristic(state, agent_id):
    matrixOfCurrPlayer = stateToMatrix(state, agent_id)
    matrixOfOpponent = stateToMatrix(state, (agent_id + 1) % 2)
    
    valueOfHeuristic = 0
    if(definitelyWin(matrixOfCurrPlayer)):
        valueOfHeuristic += 50
    
    #optionsToLoose = numOfOptionsToWin(matrixOfOpponent, matrixOfCurrPlayer)
    valueOfHeuristic += blockWithBestPawn(matrixOfOpponent, matrixOfCurrPlayer, state, agent_id)
    
    numOfMovesCurr = numOfMoves(matrixOfCurrPlayer)
    #printMatrix(matrixOfCurrPlayer)
    numOfMovesOpponent = numOfMoves(matrixOfOpponent)
    #printMatrix(matrixOfOpponent)
    #print("numOfMovesCurr: " + str(numOfMovesCurr))
    #print("numOfMovesOpponent: " + str(numOfMovesOpponent))
    if((numOfMovesCurr == 1 and numOfMovesOpponent == 0) or (numOfMovesOpponent == 1 and numOfMovesCurr == 1)):
        valueOfHeuristic += firstMove(matrixOfCurrPlayer, agent_id) 
                
    valueOfHeuristic += numOfOptionsToWin(matrixOfCurrPlayer, matrixOfOpponent, agent_id, state)
    valueOfHeuristic -= numOfOptionsToWin(matrixOfOpponent, matrixOfCurrPlayer, agent_id, state)
            
    #valueOfHeuristic += dumb_heuristic2(state, agent_id)
    
    #print(valueOfHeuristic)
    return valueOfHeuristic


"""
Win: If the player has two in a row, they can place a third to get three in a row. - DONE
Block: If the opponent has two in a row, the player must play the third themselves to block the opponent. - DONE
Fork: Cause a scenario where the player has two ways to win (two non-blocked lines of 2).
Blocking an opponent's fork: If there is only one possible fork for the opponent, the player should block it. Otherwise, the player should block 
        all forks in any way that simultaneously allows them to make two in a row. Otherwise, the player should make a two in a row to force the 
        opponent into defending, as long as it does not result in them producing a fork. For example, if "X" has two opposite corners and "O" has 
        the center, "O" must not play a corner move to win. (Playing a corner move in this scenario produces a fork for "X" to win.)
Center: A player marks the center. (If it is the first move of the game, playing a corner move gives the second player more opportunities to make a 
        mistake and may therefore be the better choice; however, it makes no difference between perfect players.)
Opposite corner: If the opponent is in the corner, the player plays the opposite corner.
Empty corner: The player plays in a corner square.
Empty side: The player plays in a middle square on any of the four sides.

numOfExposedGobblinsOnBoard = dumb_heuristic2 - DONE
"""
#for example: player1_pawns = "B1": (not_on_board, "B")
#if they ar hidden?
def stateToMatrix(curr_state, agent_id):
    matrix = np.full((3, 3), " ")
    if(agent_id == 0):
        #print("player 0 mine")
        for pawn_key in curr_state.player1_pawns.keys():
            #print(curr_state.player1_pawns[pawn_key])
            curr_location = gge.find_curr_location(curr_state, pawn_key, 0)
            if(curr_location[0] == -1):
                continue
            if is_hidden(curr_state, agent_id, pawn_key):
                continue
            matrix[curr_location[0]][curr_location[1]] = pawn_key
    
    if(agent_id == 1):
        #print("player 1 mine")
        for pawn_key in curr_state.player2_pawns.keys():
            #print(curr_state.player2_pawns[pawn_key])
            curr_location = gge.find_curr_location(curr_state, pawn_key, 1)
            if(curr_location[0] == -1):
                continue
            if is_hidden(curr_state, agent_id, pawn_key):
                continue
            matrix[curr_location[0]][curr_location[1]] = pawn_key
            
    return matrix

#the num of exposed gobblins on board
def numOfMoves(matrix):
    numOfMoves = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if matrix[i][j] != " ":
                numOfMoves += 1

    return numOfMoves

#counts the number of options to win. if there is more than one option to win, it is a better state than if there is only one option to win.
def createMatrixOfWins(matrix, opponentMatrix, agent_id, state):
    matrixOfWins = np.full((3, 3), 0)
    
    maxPawn = getMaximumPawn(agent_id, state)
    
    for i in range(3):
        if (matrix[i][0] != " " and matrix[i][1] != " " and matrix[i][2] == " " and (gge.size_cmp(maxPawn, opponentMatrix[i][2]) or opponentMatrix[i][2] == " ")):
                matrixOfWins[i][2] += 1
        if (matrix[i][0] != " " and matrix[i][2] != " " and matrix[i][1] == " " and (gge.size_cmp(maxPawn, opponentMatrix[i][1]) or opponentMatrix[i][1] == " ")):
                matrixOfWins[i][1] += 1
        if (matrix[i][1] != " " and matrix[i][2] != " " and matrix[i][0] == " " and (gge.size_cmp(maxPawn, opponentMatrix[i][0]) or opponentMatrix[i][0] == " ")):
                matrixOfWins[i][0] += 1
            
    # check columns
    for j in range(3):
        if (matrix[0][j] != " " and matrix[1][j] != " " and matrix[2][j] == " " and (gge.size_cmp(maxPawn, opponentMatrix[2][j]) or opponentMatrix[2][j] == " ")):
                matrixOfWins[2][j] += 1
        if (matrix[0][j] != " " and matrix[2][j] != " " and matrix[1][j] == " " and (gge.size_cmp(maxPawn, opponentMatrix[1][j]) or opponentMatrix[1][j] == " ")):
                matrixOfWins[1][j] += 1
        if (matrix[1][j] != " " and matrix[2][j] != " " and matrix[0][j] == " " and (gge.size_cmp(maxPawn, opponentMatrix[0][j]) or opponentMatrix[0][j] == " ")):
                matrixOfWins[0][j] += 1

    # check obliques
    if (matrix[0][0] != " " and matrix[1][1] != " " and matrix[2][2] == " " and (gge.size_cmp(maxPawn, opponentMatrix[2][2]) or opponentMatrix[2][2] == " ")):
            matrixOfWins[2][2] += 1
    if (matrix[0][0] != " " and matrix[2][2] != " " and matrix[1][1] == " " and (gge.size_cmp(maxPawn, opponentMatrix[1][1]) or opponentMatrix[1][1] == " ")):
            matrixOfWins[1][1] += 1
    if (matrix[1][1] != " " and matrix[2][2] != " " and matrix[0][0] == " " and (gge.size_cmp(maxPawn, opponentMatrix[0][0]) or opponentMatrix[0][0] == " ")):
            matrixOfWins[0][0] += 1
            
    if (matrix[0][2] != " " and matrix[1][1] != " " and matrix[2][0] == " " and (gge.size_cmp(maxPawn, opponentMatrix[2][0]) or opponentMatrix[2][0] == " ")):
            matrixOfWins[2][0] += 1
    if (matrix[1][1] != " " and matrix[2][0] != " " and matrix[0][2] == " " and (gge.size_cmp(maxPawn, opponentMatrix[0][2]) or opponentMatrix[0][2] == " ")):
            matrixOfWins[0][2] += 1
    if (matrix[0][2] != " " and matrix[2][0] != " " and matrix[1][1] == " " and (gge.size_cmp(maxPawn, opponentMatrix[1][1]) or opponentMatrix[1][1] == " ")):
            matrixOfWins[1][1] += 1
    
    return matrixOfWins

def numOfOptionsToWin(matrixOfCurrPlayer, matrixOfOpponent, agent_id, state):
    matrixOfWins = createMatrixOfWins(matrixOfCurrPlayer, matrixOfOpponent, agent_id, state)
    sum = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if (matrixOfWins[i][j] > 1):
                sum *= 2
            else:
                sum += matrixOfWins[i][j]
    return sum

def printMatrix(matrix):
    print("     *******     ")
    for i in range(0, 3):
        print(" | ", matrix[i, 0], " | ", matrix[i, 1], " | ", matrix[i, 2], " | ")
    print("     *******     ")


def firstMove(matrix, agent_id):
    #print("first corner move")
    #printMatrix(matrix)
    num = 0
    if agent_id == 0:
        if matrix[0][0] == "S" or matrix[0][2] == "S" or matrix[2][2] == "S" or matrix[2][0] == "S":
            num = 5
        if matrix[0][0] == "M" or matrix[0][2] == "M" or matrix[2][2] == "M" or matrix[2][0] == "M":
            num = 3
        if matrix[0][0] == "B" or matrix[0][2] == "B" or matrix[2][2] == "B" or matrix[2][0] == "B":
            num = 1
        #print("firstMove. agent_1. num is: {}", num)
    if agent_id == 1:
        if matrix[1][1] == "S":
            num = 5
        if matrix[1][1] == "M":
            num = 3
        if matrix[1][1] == "B":
            num = 1
        #print("firstMove. agent_2. num is: {}", num)
    return num

#gives priority for a state in which we are definatlly going to win
def definitelyWin(matrix):
    #printMatrix(matrix)
    for i in range(3):
        if ((matrix[i][0] != " " and matrix[i][1] != " " and matrix[i][2] != " ")
            or (matrix[i][0] != " " and matrix[i][2] != " " and matrix[i][1] != " ")
            or (matrix[i][1] != " " and matrix[i][2] != " " and matrix[i][0] != " ")):
                return True
            
    # check columns
    for j in range(3):
        if ((matrix[0][j] != " " and matrix[1][j] != " " and matrix[2][j] != " ")
            or (matrix[0][j] != " " and matrix[2][j] != " " and matrix[1][j] != " ")
            or (matrix[1][j] != " " and matrix[2][j] != " " and matrix[0][j] != " ")):
                return True

    # check main diagonal
    if ((matrix[0][0] != " " and matrix[1][1] != " " and matrix[2][2] != " ")
        or (matrix[0][0] != " " and matrix[2][2] != " " and matrix[1][1] != " ")
        or (matrix[1][1] != " " and matrix[2][2] != " " and matrix[0][0] != " ")):
            return True
    
    # check secondary diagonal        
    if ((matrix[0][2] != " " and matrix[1][1] != " " and matrix[2][0] != " ")
        or (matrix[1][1] != " " and matrix[2][0] != " " and matrix[0][2] != " ")
        or (matrix[0][2] != " " and matrix[2][0] != " " and matrix[1][1] != " ")):
            return True
    
    #no win    
    return False

def createMatrixOfBlocks(matrix, matrixOfCurrerntPlayer, agent_id, state):
    matrixOfBlocks = np.full((3, 3), " ")
    
    maxPawn = getMaximumPawn(agent_id, state)
        
    for i in range(3):
        if (matrix[i][0] != " " and matrix[i][1] != " " and matrix[i][2] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[i][2], maxPawn) != -1 or matrixOfCurrerntPlayer[i][2] == " ")):
                matrixOfBlocks[i][2] = "X"
        if (matrix[i][0] != " " and matrix[i][2] != " " and matrix[i][1] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[i][1], maxPawn) != -1 or matrixOfCurrerntPlayer[i][1] == " ")):
                matrixOfBlocks[i][1] = "X"
        if (matrix[i][1] != " " and matrix[i][2] != " " and matrix[i][0] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[i][0], maxPawn) != -1 or matrixOfCurrerntPlayer[i][0] == " ")):
                matrixOfBlocks[i][0] = "X"
            
    # check columns
    for j in range(3):
        if (matrix[0][j] != " " and matrix[1][j] != " " and matrix[2][j] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[2][j], maxPawn) != -1 or matrixOfCurrerntPlayer[2][j] == " ")):
                matrixOfBlocks[2][j] = "X"
        if (matrix[0][j] != " " and matrix[2][j] != " " and matrix[1][j] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[1][j], maxPawn) != -1 or matrixOfCurrerntPlayer[1][j] == " ")):
                matrixOfBlocks[1][j] = "X"
        if (matrix[1][j] != " " and matrix[2][j] != " " and matrix[0][j] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[0][j], maxPawn) != -1 or matrixOfCurrerntPlayer[0][j] == " ")):
                matrixOfBlocks[0][j] = "X"

    # check obliques
    if (matrix[0][0] != " " and matrix[1][1] != " " and matrix[2][2] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[2][2], maxPawn) != -1 or matrixOfCurrerntPlayer[2][2] == " ")):
            matrixOfBlocks[2][2] = "X"
    if (matrix[0][0] != " " and matrix[2][2] != " " and matrix[1][1] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[1][1], maxPawn) != -1 or matrixOfCurrerntPlayer[1][1] == " ")):
            matrixOfBlocks[1][1] = "X"
    if (matrix[1][1] != " " and matrix[2][2] != " " and matrix[0][0] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[0][0], maxPawn) != -1 or matrixOfCurrerntPlayer[0][0] == " ")):
            matrixOfBlocks[0][0] = "X"
            
    if (matrix[0][2] != " " and matrix[1][1] != " " and matrix[2][0] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[2][0], maxPawn) != -1 or matrixOfCurrerntPlayer[2][0] == " ")):
            matrixOfBlocks[2][0] = "X"
    if (matrix[1][1] != " " and matrix[2][0] != " " and matrix[0][2] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[0][2], maxPawn) != -1 or matrixOfCurrerntPlayer[0][2] == " ")):
            matrixOfBlocks[0][2] = "X"
    if (matrix[0][2] != " " and matrix[2][0] != " " and matrix[1][1] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[1][1], maxPawn) != -1 or matrixOfCurrerntPlayer[1][1] == " ")):
            matrixOfBlocks[1][1] = "X"
    
    return matrixOfBlocks
       
#put the best pawn from the available one (not on board and not hidden) in consider of what the opponent has
#deals only with one loose option
def blockWithBestPawn(matrixOfOpponent, matrixOfCurrPlayer, state, curr_agent_id):
    matrixOfBlocks = createMatrixOfBlocks(matrixOfOpponent, matrixOfCurrPlayer, curr_agent_id, state)
    
    maxPawn = "S"
    blocks = 0
    for i in range(3):
        for j in range(3):
            if(matrixOfBlocks[i][j] == "X" and matrixOfCurrPlayer[i][j] != " "):
                #print(matrixOfBlocks[i][j])
                #print(matrixOfCurrPlayer[i][j])
                blocks += 1
                #print("\nmatrices")
                #printMatrix(matrixOfBlocks)
                #printMatrix(matrixOfCurrPlayer)
                if(gge.size_cmp(matrixOfCurrPlayer[i][j], maxPawn)):
                    maxPawn = matrixOfCurrPlayer[i][j]
    
    if blocks == 0:
        return 0   
        
    maxOpponentPawn = getMaximumPawn(curr_agent_id, state)
    #print("min: {}".format(maxPawn))
    #print("max: {}".format(maxOpponentPawn))
        
    if maxPawn == maxOpponentPawn:
        return 20
    if maxPawn == "S":
        return 1
    if maxPawn == "B":
        return 30
    if maxOpponentPawn == "S":
        return 5
    else:
        return -1

def getMaximumPawn(agent_id, state):
    pawns = {"B": 0, "M": 0, "S": 0}
    
    if(agent_id == 1):
        for key, value in state.player1_pawns.items():#key = pawn, value = (not_on_board, "B")
            if np.array_equal(value[0], not_on_board):
                pawns[value[1]] += 1
    if(agent_id == 2):
        for key, value in state.player2_pawns.items():
            if np.array_equal(value[0], not_on_board):
                pawns[value[1]] += 1   

    maxPawn = max(pawns, key=pawns.get)
    return maxPawn

# IMPLEMENTED FOR YOU - NO NEED TO CHANGE
def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


# agent_id is which agent you are - first player or second player
def random_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    rnd = random.randint(0, neighbor_list.__len__() - 1)
    return neighbor_list[rnd][0]


# TODO - instead of action to return check how to raise not_implemented
#neighbor_list = all possible states that are possible from this state
#neighbor = (action = (pawn, i), next_state)
def greedy(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = dumb_heuristic2(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


# TODO - add your code here
#curr_state = the current board
#neighbor_list = all possible states that are possible from this state
#neighbor = (action = (pawn, location(), next_state)
def greedy_improved(curr_state, agent_id, time_limit):
    #printMatrix(stateToMatrix(curr_state, agent_id))
    #printMatrix(stateToMatrix(curr_state, (agent_id + 1) % 2))
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        #print("pawn: {}, location: {}, heuristic: {}, next location:{}".format(neighbor[0][0], neighbor[0][1], curr_heuristic, gge.find_curr_location(neighbor[1], neighbor[0][0], agent_id)))
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    
    if max_neighbor != None:
        return max_neighbor[0]
    return max_neighbor


def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    
    raise NotImplementedError()


def alpha_beta(curr_state, agent_id, time_limit):
    
    raise NotImplementedError()


def expectimax(curr_state, agent_id, time_limit):
    
    raise NotImplementedError()

# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    
    raise NotImplementedError()
