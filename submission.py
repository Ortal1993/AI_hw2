import math
import random

import numpy as np

import Gobblet_Gobblers_Env as gge

import time

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
    if((numOfMovesCurr == 1 and numOfMovesOpponent == 0) or (numOfMovesOpponent == 1 and numOfMovesCurr == 1)):
        valueOfHeuristic += firstMove(matrixOfCurrPlayer, agent_id) 
                
    valueOfHeuristic += numOfOptionsToWin(matrixOfCurrPlayer, matrixOfOpponent, agent_id, state)
    valueOfHeuristic -= numOfOptionsToWin(matrixOfOpponent, matrixOfCurrPlayer, agent_id, state)
            
    valueOfHeuristic += dumb_heuristic2(state, agent_id)
    
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

#only visible pawns on board
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
        if ((matrix[i][0] != " ") and
            (matrix[i][1] != " ") and
            (matrix[i][2] == " ") and
            (gge.size_cmp(matrixOfCurrerntPlayer[i][2], maxPawn) != -1 or matrixOfCurrerntPlayer[i][2] == " ")):
                matrixOfBlocks[i][2] = "X"
                if gge.size_cmp(maxPawn, matrix[i][0]) == 1:
                    matrixOfBlocks[i][0] = "X"
                if gge.size_cmp(maxPawn, matrix[i][1]) == 1:
                    matrixOfBlocks[i][1] = "X"
        if (matrix[i][0] != " " and 
            matrix[i][2] != " " and 
            matrix[i][1] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[i][1], maxPawn) != -1 or matrixOfCurrerntPlayer[i][1] == " ")):
                matrixOfBlocks[i][1] = "X"
                if gge.size_cmp(maxPawn, matrix[i][0]) == 1:
                    matrixOfBlocks[i][0] = "X"
                if gge.size_cmp(maxPawn, matrix[i][2]) == 1:
                    matrixOfBlocks[i][2] = "X"
        if (matrix[i][1] != " " and 
            matrix[i][2] != " " and 
            matrix[i][0] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[i][0], maxPawn) != -1 or matrixOfCurrerntPlayer[i][0] == " ")):
                matrixOfBlocks[i][0] = "X"
                if gge.size_cmp(maxPawn, matrix[i][1]) == 1:
                    matrixOfBlocks[i][1] = "X"
                if gge.size_cmp(maxPawn, matrix[i][2]) == 1:
                     matrixOfBlocks[i][2] = "X"
            
    # check columns
    for j in range(3):
        if (matrix[0][j] != " " and 
            matrix[1][j] != " " and 
            matrix[2][j] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[2][j], maxPawn) != -1 or matrixOfCurrerntPlayer[2][j] == " ")):
                matrixOfBlocks[2][j] = "X"
                if gge.size_cmp(maxPawn, matrix[0][j]) == 1:
                    matrixOfBlocks[0][j] = "X"
                if gge.size_cmp(maxPawn, matrix[1][j]) == 1:
                    matrixOfBlocks[1][j] = "X"
        if (matrix[0][j] != " " and 
            matrix[2][j] != " " and 
            matrix[1][j] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[1][j], maxPawn) != -1 or matrixOfCurrerntPlayer[1][j] == " ")):
                matrixOfBlocks[1][j] = "X"
                if gge.size_cmp(maxPawn, matrix[0][j]) == 1:
                    matrixOfBlocks[0][j] = "X"
                if gge.size_cmp(maxPawn, matrix[2][j]) == 1:
                    matrixOfBlocks[2][j] = "X"
        if (matrix[1][j] != " " and 
            matrix[2][j] != " " and 
            matrix[0][j] == " " and 
            (gge.size_cmp(matrixOfCurrerntPlayer[0][j], maxPawn) != -1 or matrixOfCurrerntPlayer[0][j] == " ")):
                matrixOfBlocks[0][j] = "X"
                if gge.size_cmp(maxPawn, matrix[1][j]) == 1:
                    matrixOfBlocks[1][j] = "X"
                if gge.size_cmp(maxPawn, matrix[2][j]) == 1:
                    matrixOfBlocks[2][j] = "X"

    # check obliques
    if (matrix[0][0] != " " and 
        matrix[1][1] != " " and 
        matrix[2][2] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[2][2], maxPawn) != -1 or matrixOfCurrerntPlayer[2][2] == " ")):
            matrixOfBlocks[2][2] = "X"
            if gge.size_cmp(maxPawn, matrix[0][0]) == 1:
                    matrixOfBlocks[0][0] = "X"
            if gge.size_cmp(maxPawn, matrix[1][1]) == 1:
                matrixOfBlocks[1][1] = "X"
    if (matrix[0][0] != " " and 
        matrix[2][2] != " " and 
        matrix[1][1] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[1][1], maxPawn) != -1 or matrixOfCurrerntPlayer[1][1] == " ")):
            matrixOfBlocks[1][1] = "X"
            if gge.size_cmp(maxPawn, matrix[0][0]) == 1:
                    matrixOfBlocks[0][0] = "X"
            if gge.size_cmp(maxPawn, matrix[2][2]) == 1:
                matrixOfBlocks[2][2] = "X"
    if (matrix[1][1] != " " and 
        matrix[2][2] != " " and 
        matrix[0][0] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[0][0], maxPawn) != -1 or matrixOfCurrerntPlayer[0][0] == " ")):
            matrixOfBlocks[0][0] = "X"
            if gge.size_cmp(maxPawn, matrix[1][1]) == 1:
                    matrixOfBlocks[1][1] = "X"
            if gge.size_cmp(maxPawn, matrix[2][2]) == 1:
                matrixOfBlocks[2][2] = "X"
            
    if (matrix[0][2] != " " and 
        matrix[1][1] != " " and 
        matrix[2][0] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[2][0], maxPawn) != -1 or matrixOfCurrerntPlayer[2][0] == " ")):
            matrixOfBlocks[2][0] = "X"
            if gge.size_cmp(maxPawn, matrix[0][2]) == 1:
                    matrixOfBlocks[0][2] = "X"
            if gge.size_cmp(maxPawn, matrix[1][1]) == 1:
                matrixOfBlocks[1][1] = "X"
    if (matrix[1][1] != " " and 
        matrix[2][0] != " " and 
        matrix[0][2] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[0][2], maxPawn) != -1 or matrixOfCurrerntPlayer[0][2] == " ")):
            matrixOfBlocks[0][2] = "X"
            if gge.size_cmp(maxPawn, matrix[1][1]) == 1:
                    matrixOfBlocks[1][1] = "X"
            if gge.size_cmp(maxPawn, matrix[2][0]) == 1:
                matrixOfBlocks[2][0] = "X"
    if (matrix[0][2] != " " and 
        matrix[2][0] != " " and 
        matrix[1][1] == " " and 
        (gge.size_cmp(matrixOfCurrerntPlayer[1][1], maxPawn) != -1 or matrixOfCurrerntPlayer[1][1] == " ")):
            matrixOfBlocks[1][1] = "X"
            if gge.size_cmp(maxPawn, matrix[0][2]) == 1:
                    matrixOfBlocks[0][2] = "X"
            if gge.size_cmp(maxPawn, matrix[2][0]) == 1:
                matrixOfBlocks[2][0] = "X"
    
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
        
    if maxPawn == maxOpponentPawn:#maxOpponentPawn is equal to maxPawn
        return 40
    if maxPawn == "S": #maxOpponentPawn is "M" or "B"
        return 10
    if maxPawn == "B": #maxOpponentPawn is "M"
        return 40
    if maxOpponentPawn == "S": #maxPawn is "M"
        return 30
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
#to do - greedy_improved_aux
def greedy_improved(curr_state, agent_id, time_limit):
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

class RBMinimax:
    
    def __init__(self, state, agent_id, timeout, depth, actionToThisState, alpha, beta, heuristic = smart_heuristic):
        self.state = state
        self.agent_id = agent_id
        self.timeout = timeout
        self.depth = depth
        self.actionToThisState = actionToThisState
        self.alpha = alpha
        self.beta = beta
        self.heuristic = heuristic
        self.best_max_action_value = (None, -math.inf) #((pawn, location), value)
        self.best_min_action_value = (None, math.inf) #((pawn, location), value)
        self.start_time = time.time()
        self.is_done_flag = False
     
    #neighbor[0] = action = (pawn, location())
    #neighbor[1] = next_state
    def run_rb_minimax(self):            
        if self.is_done() or gge.is_final_state(self.state) or self.depth == 0:
            return (self.actionToThisState, self.heuristic(self.state, self.agent_id))
         
        neighbor_list = self.state.get_neighbors() #list of (action, next_state) #also handle the turn
        if self.state.turn == self.agent_id:
            for neighbor in neighbor_list:
                action, curr_heuristic = RBMinimax(neighbor[1], self.agent_id, self.timeout - (time.time() - self.start_time), self.depth - 1, 
                                                   neighbor[0], self.alpha, self.beta, self.heuristic).run_rb_minimax()
                # if (not gge.is_legal_step(action, self.state)):
                #     continue
                if self.is_done():
                    return self.best_max_action_value
                if curr_heuristic >= self.best_max_action_value[1]:
                    #print("max, depth: {}, pawn: {}, location: {}, heuristic: {}, next location:{}".format(self.depth, neighbor[0][0], neighbor[0][1], curr_heuristic, gge.find_curr_location(neighbor[1], neighbor[0][0], self.agent_id)))
                    self.best_max_action_value = (neighbor[0], curr_heuristic)
                if(self.alpha != None and self.beta != None):
                    self.alpha = max(self.alpha, self.best_max_action_value[1])
                    if(self.best_max_action_value[1] >= self.beta):
                        return (self.actionToThisState, math.inf)
            return self.best_max_action_value
        else:#turn != self.state.turn
            for neighbor in neighbor_list:
                action, curr_heuristic = RBMinimax(neighbor[1], self.agent_id, self.timeout - (time.time() - self.start_time), self.depth - 1, 
                                                   neighbor[0], self.alpha, self.beta, self.heuristic).run_rb_minimax()
                # if (not gge.is_legal_step(action, self.state)):
                #     continue
                if self.is_done():
                    return self.best_min_action_value                
                if curr_heuristic <= self.best_min_action_value[1]:
                    #print("min, depth: {}, pawn: {}, location: {}, heuristic: {}, next location:{}".format(self.depth, neighbor[0][0], neighbor[0][1], curr_heuristic, gge.find_curr_location(neighbor[1], neighbor[0][0], self.agent_id)))
                    self.best_min_action_value = (neighbor[0], curr_heuristic)
                if(self.alpha != None and self.beta != None):
                    self.beta = min(self.beta, self.best_min_action_value[1])
                    if(self.best_min_action_value[1] <= self.alpha):
                        return (self.actionToThisState, -math.inf)
            return self.best_min_action_value          
        
    def checkTime(self):
        end_time = time.time()
        if(end_time - self.start_time + 0.2 > self.timeout): #-0.2 to make sure we don't go over the time limit
            self.is_done_flag = True
        self.timeout -= (end_time - self.start_time)
        if(self.timeout < end_time - self.start_time):
            self.is_done_flag = True
        
    def is_done(self):
        self.checkTime()
        return self.is_done_flag
    
    def get_best_action(self):
        return self.best_max_action_value[0]

def alpha_beta_minimax(curr_state, agent_id, time_limit, alpha = None, beta = None):
    rb_minimax = RBMinimax(curr_state, agent_id, time_limit, 1, None, alpha, beta)
    while not rb_minimax.is_done():
        rb_minimax.run_rb_minimax()
        # print(rb_minimax.best_max_action_value)
        # print(rb_minimax.best_min_action_value)
        # print(rb_minimax.depth)
        rb_minimax.depth += 1
    #print(rb_minimax.get_best_action())
    return rb_minimax.get_best_action()
    
def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    #print(agent_id)
    #print(curr_state.turn)
    action = alpha_beta_minimax(curr_state, agent_id, time_limit)
    return action

def alpha_beta(curr_state, agent_id, time_limit):
    action = alpha_beta_minimax(curr_state, agent_id, time_limit, -math.inf, math.inf)
    return action

  

class RB_Expectimax:
    
    def __init__(self, state, agent_id, timeout, depth, actionToThisState, heuristic = smart_heuristic):
        self.state = state
        self.agent_id = agent_id
        self.timeout = timeout
        self.depth = depth
        self.actionToThisState = actionToThisState
        self.heuristic = heuristic
        self.best_max_action_value = (None, -math.inf) #((pawn, location), value)
        self.best_probability_action_value = (None, 0) #((pawn, location), value)
        self.start_time = time.time()
        self.is_done_flag = False
     
    #neighbor[0] = action = (pawn, location())
    #neighbor[1] = next_state
    def run_rb_expectimax(self):            
        if self.is_done() or gge.is_final_state(self.state) or self.depth == 0:
            return (self.actionToThisState, self.heuristic(self.state, self.agent_id))
        
        neighbor_list = self.state.get_neighbors() #list of (action, next_state) #also handle the turn
        if self.state.turn == self.agent_id:
            for neighbor in neighbor_list:
                action, curr_heuristic = RB_Expectimax(neighbor[1], self.agent_id, self.timeout - (time.time() - self.start_time), self.depth - 1, 
                                                   neighbor[0], self.heuristic).run_rb_expectimax()
                # if (not gge.is_legal_step(action, self.state)):
                #     continue
                if self.is_done():
                    return self.best_max_action_value
                if curr_heuristic >= self.best_max_action_value[1]:
                    self.best_max_action_value = (neighbor[0], curr_heuristic)
            return self.best_max_action_value
        else:#turn != self.state.turn
            utilities = []
            probabilities = []
            for neighbor in neighbor_list:
                action, curr_utility = RB_Expectimax(neighbor[1], self.agent_id, self.timeout - (time.time() - self.start_time), self.depth - 1, 
                                                   neighbor[0], self.heuristic).run_rb_expectimax()
                utilities.append(curr_utility)
                probabilities.append(self.probabilityOfState(self.state, neighbor[1], self.agent_id))
                if self.is_done():
                    return self.best_probability_action_value                        
            #after we have all the utilities and probabilities
            sumOfProbabilities = 0
            for probability in probabilities:
                sumOfProbabilities += probability #sumOfProbabilities is at least in the length of neighbor_list                            
            chance = 0
            for probability, curr_utility in zip(probabilities, utilities):
                chance += probability/sumOfProbabilities * curr_utility
                
            self.best_probability_action_value = (self.state, chance)                
            return self.best_probability_action_value
        
    def checkTime(self):
        end_time = time.time()
        if(end_time - self.start_time + 0.2 > self.timeout): #-0.2 to make sure we don't go over the time limit
            self.is_done_flag = True
        self.timeout -= (end_time - self.start_time)
        if(self.timeout < end_time - self.start_time):
            self.is_done_flag = True
        
    def is_done(self):
        self.checkTime()
        return self.is_done_flag
    
    def get_best_action(self):
        return self.best_max_action_value[0]
    
    def probabilityOfState(self, curr_state, next_state, agent_id):
        probability = 1
        if(self.pawnOnPawn(curr_state, next_state, agent_id) or self.smallPawnMove(curr_state, next_state, agent_id)):
            probability = 2
        return probability
    
    #for inner use only
    def pawnOnPawn(self, curr_state, next_state, curr_agent_id):    
        opponent_pawns_curr_state = curr_state.player2_pawns.values() if curr_agent_id == 0 else curr_state.player1_pawns.values()    
        curr_agent_pawns_next_state = next_state.player1_pawns.values() if curr_agent_id == 0 else next_state.player2_pawns.values()
        opponent_pawns_next_state = next_state.player2_pawns.values() if curr_agent_id == 0 else next_state.player1_pawns.values()
        opponent_id = 1 if curr_agent_id == 0 else 0
        for oppenent_location, oppenent_pawn in opponent_pawns_next_state:
            if not np.array_equal(oppenent_location, not_on_board) and not is_hidden(next_state, opponent_id, oppenent_pawn): #pawn is on board and is not hidden
                previous_location = gge.find_curr_location(curr_state, oppenent_pawn, opponent_id)                    
                if not np.array_equal(oppenent_location, previous_location):#pawn has moved in this step
                    for curr_agent_location, curr_agent_pawn in curr_agent_pawns_next_state:
                        if np.array_equal(curr_agent_location, oppenent_location):#pawn is on an opponent's pawn
                            return 1
                    for hidden_location, hidden_pawn in opponent_pawns_curr_state:
                        if np.array_equal(oppenent_location, hidden_location):#pawn is on a pawn of the same agent
                            return 1            
                                
        return 0 #no pawn is on pawn
 
    #for inner use only
    def smallPawnMove(self, curr_state, next_state, curr_agent_id):
        opponent_pawns_next_state = next_state.player2_pawns.values() if curr_agent_id == 0 else next_state.player1_pawns.values()
        opponent_id = 1 if curr_agent_id == 0 else 0
        for oppenent_location, oppenent_pawn in opponent_pawns_next_state:
            if oppenent_pawn == "S" and not np.array_equal(oppenent_location, not_on_board) and not is_hidden(next_state, opponent_id, oppenent_pawn): #pawn is on board and is not hidden
                previous_location = gge.find_curr_location(curr_state, oppenent_pawn, opponent_id)
                if not np.array_equal (oppenent_location, previous_location):#"S" pawn has moved in this step
                    return 1                        
                                
        return 0 #no pawn is on pawn 

def expectimax(curr_state, agent_id, time_limit):
    rb_expectimax = RB_Expectimax(curr_state, agent_id, time_limit, 1, None)
    while not rb_expectimax.is_done():
        rb_expectimax.run_rb_expectimax()
        rb_expectimax.depth += 1
    return rb_expectimax.get_best_action()

# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    
    raise NotImplementedError()
