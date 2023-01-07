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

# count the numbers of pawns that i have on the board that aren't hidden
# player1_pawns = "B1": (not_on_board, "B")
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    if agent_id == 0:
        for key, value in state.player1_pawns.items():#key = pawn, value = (not_on_board, "B")
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1

    return sum_pawns

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

#neighbor_list = all possible states that are possible from this state
#neighbor = (action = (pawn, location(), next_state)
def greedy_improved(curr_state, agent_id, time_limit):
    max_heuristic = 0
    max_neighbor = None
    neighbor_list = curr_state.get_neighbors()
    for neighbor in neighbor_list:
        next_state = neighbor[1]
        curr_heuristic = smart_heuristic(next_state, agent_id)
        # print("pawn: {}, location: {}, heuristic: {}, next location:{}".format(neighbor[0][0], neighbor[0][1], curr_heuristic, gge.find_curr_location(neighbor[1], neighbor[0][0], agent_id)))
        # print("###")
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    
    if max_neighbor != None:
        return max_neighbor[0]
    return max_neighbor

def alpha_beta_minimax(curr_state, agent_id, time_limit, alpha = None, beta = None):
    rb_minimax = RBMinimax(curr_state, agent_id, time_limit, 0, None, alpha, beta)
    max_action = None
    while not rb_minimax.is_done():
        try:
            action, value = rb_minimax.run_rb_minimax()
            print("action: {}, value: {}".format(action, value))
            max_action = action
            print("max_action: {}".format(max_action))
            print("depth: {}".format(rb_minimax.depth))
            rb_minimax.depth += 1
        except Exception as e:
            rb_minimax.is_done_flag = True
    print("#####max_action: {}".format(max_action))        
    return max_action
    
def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    action = alpha_beta_minimax(curr_state, agent_id, time_limit)
    return action

def alpha_beta(curr_state, agent_id, time_limit):
    action = alpha_beta_minimax(curr_state, agent_id, time_limit, -math.inf, math.inf)
    return action

def expectimax(curr_state, agent_id, time_limit):
    rb_expectimax = RB_Expectimax(curr_state, agent_id, time_limit, 0, None)
    max_action = None
    action = None
    while not rb_expectimax.is_done():
        try:
            action, value = rb_expectimax.run_rb_expectimax()
            max_action = action
            rb_expectimax.depth += 1 
        except Exception:
            rb_expectimax.is_done_flag = True
    return max_action

# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    action = alpha_beta_minimax(curr_state, agent_id, time_limit, -math.inf, math.inf)
    return action


def printMatrix(matrix):
    print("     *******     ")
    for i in range(0, 3):
        print(" | ", matrix[i, 0], " | ", matrix[i, 1], " | ", matrix[i, 2], " | ")
    print("     *******     ")

#convert a state to a matrix of exposed pawns on board of a specific player
def stateToMatrix(curr_state, agent_id):
    matrix = np.full((3, 3), " ")
    if(agent_id == 0):
        for pawn_key in curr_state.player1_pawns.keys():
            curr_location = gge.find_curr_location(curr_state, pawn_key, 0)
            if(curr_location[0] == -1):
                continue
            if is_hidden(curr_state, agent_id, pawn_key):
                continue
            matrix[curr_location[0]][curr_location[1]] = pawn_key
    
    if(agent_id == 1):
        for pawn_key in curr_state.player2_pawns.keys():
            curr_location = gge.find_curr_location(curr_state, pawn_key, 1)
            if(curr_location[0] == -1):
                continue
            if is_hidden(curr_state, agent_id, pawn_key):
                continue
            matrix[curr_location[0]][curr_location[1]] = pawn_key
            
    return matrix

#gives priority for a state in which we are definatlly going to win
def definitelyWin(matrix):
    for i in range(3):
        if(matrix[i][0] != " " and matrix[i][1] != " " and matrix[i][2] != " "):
            return 1
            
    # check columns
    for j in range(3):
        if (matrix[0][j] != " " and matrix[1][j] != " " and matrix[2][j] != " "):
            return 1

    # check main diagonal
    if (matrix[0][0] != " " and matrix[1][1] != " " and matrix[2][2] != " "):
        return 1
    
    # check secondary diagonal        
    if (matrix[0][2] != " " and matrix[1][1] != " " and matrix[2][0] != " "):
            return 1
    
    #no win    
    return 0

#potential win - winning after the opponent makes a move
def evaluatePotentialWins(state, agent_id, matrixCurrPlayer, matrixOpponent, matrixOfPotentialBlocksOpponent):
    #matrixOfPotentialBlocks = createMatrixOfPotentialBlocks(matrixCurrPlayer)
    
    availablePawns = getAvailablePawnsToBlock(matrixOfPotentialBlocksOpponent, matrixCurrPlayer)
    maxAvailablePawn = getMaxPawn(availablePawns)#in order to know if he would be able to put a pawn on maxAavailablePawn

    availableOpponentPawns = getAvailablePawns(state, agent_id)#getAvailablePawnsToBlock(matrixOfPotentialBlocksOpponent, matrixOpponent)
    maxAvailableOpponentPawnForBlocking = getMaxPawn(availableOpponentPawns)
    
    # print("availablePawns: ", availablePawns)
    # print("aviailableOpponentPawns: ", availableOpponentPawns)
    
    matrixOfPotentialWins = createMatrixOfPotentialWins(matrixCurrPlayer)
    return evaluateRemainingSpots(matrixOpponent, matrixOfPotentialWins, maxAvailablePawn, maxAvailableOpponentPawnForBlocking)

def createMatrixOfPotentialWins(matrixCurrPlayer):
    matrixOfWins = np.full((3, 3), 0)
    row, col = -1, -1
    for i in range(3):
        row, col = checkIthRowForWin(matrixCurrPlayer, i)
        if row != -1 and col != -1:
            matrixOfWins[row][col] = 1
    
    # check columns
    row, col = -1, -1
    for i in range(3):
        row, col = checkIthColForWin(matrixCurrPlayer, i)
        if row != -1 and col != -1:
            matrixOfWins[row][col] = 1
            
    # check diagonals
    row, col = -1, -1
    row, col = checkMainDiagonalForWin(matrixCurrPlayer)
    if row != -1 and col != -1:
        matrixOfWins[row][col] = 1
        
    row, col = -1, -1
    row, col = checkSecondaryDiagonalForWin(matrixCurrPlayer)
    if row != -1 and col != -1:
        matrixOfWins[row][col] = 1
    
    return matrixOfWins

def checkIthRowForWin(matrixCurrPlayer, row):
    colomns = [0, 1, 2]
    for col in range(3):
        if matrixCurrPlayer[row][col] != " " and col in colomns:
            colomns.remove(col)
    if len(colomns) == 1:#only 1 spot is empty
        return row, colomns[0]
    else:
        return -1, -1
    
def checkIthColForWin(matrixCurrPlayer, col):
    rows = [0, 1, 2]
    for row in range(3):
        if matrixCurrPlayer[row][col] != " " and row in rows:
            rows.remove(row)
    if len(rows) == 1:#only 1 spot is empty
        return rows[0], col
    else:
        return -1, -1
    
def checkMainDiagonalForWin(matrixCurrPlayer):
    options = [0, 1, 2]
    for i in range(3):
        if matrixCurrPlayer[i][i] != " " and i in options:
            options.remove(i)
    if len(options) == 1:#only 1 spot is empty
        return options[0], options[0]
    else:
        return -1, -1

def checkSecondaryDiagonalForWin(matrixCurrPlayer):
    options = [(0, 2), (1, 1), (2, 0)]
    if matrixCurrPlayer[0][2] != " ":
        options.remove((0, 2))        
    if matrixCurrPlayer[1][1] != " ":
        options.remove((1, 1))        
    if matrixCurrPlayer[2][0] != " ":
        options.remove((2, 0))
    if len(options) == 1:#only 1 spot is empty
        return options[0][0], options[0][1]
    else:
        return -1, -1
    
def evaluateRemainingSpots(matrixOpponent, matrixOfWins, maxAvailablePawn, maxAvailableOpponentPawnForBlocking):
    #print("win")
    value = 0
    for i in range(3):
        for j in range(3):          
            if matrixOfWins[i][j] == 1 and (matrixOpponent[i][j] == " " or gge.size_cmp(maxAvailablePawn, matrixOpponent[i][j])):
                value += evaluateSpot(maxAvailablePawn, maxAvailableOpponentPawnForBlocking)
    #print("end win")
    return value    

def createMatrixOfPotentialBlocks(matrixOpponent):
    matrixOfBlocks = np.full((3, 3), 0)    
    for i in range(3):
        row = checkIthRow(matrixOpponent, i)
        if row != -1:#need to check all options to block in the i_th row
            for i in range(3):
                matrixOfBlocks[row][i] = 1
    
    # check columns
    for i in range(3):
        col = checkIthCol(matrixOpponent, i)
        if col != -1:#need to check all option to block in the i_th colomn
            for i in range(3):
                matrixOfBlocks[i][col] = 1
            
    # check diagonals
    mainDiagonal = checkMainDiagonal(matrixOpponent)
    if mainDiagonal != -1:#need to check all option to block in the main diagonal
        for i in range(3):
            matrixOfBlocks[i][i] = 1
           
    secondaryDiagonal = checkSecondaryDiagonal(matrixOpponent)
    if secondaryDiagonal != -1: #need to check all option to block in the secondary diagonal
        for i in range(3):
            matrixOfBlocks[i][2-i] = 1
    
    return matrixOfBlocks

def evaluatePotentialBlocks(matrixOpponent, matrixCurrPlayer, matrixOfPotentialBlocks):
    #matrixOfPotentialBlocks = createMatrixOfPotentialBlocks(matrixOpponent)
                    
    availableOpponentPawns = getAvailablePawnsToBlock(matrixOfPotentialBlocks, matrixOpponent)
    maxAvailableOpponentPawnForBlocking = getMaxPawn(availableOpponentPawns)#in order to know if he would be able to put a pawn on maxAavailablePawn
        
    return evaluateBlocks(matrixCurrPlayer, matrixOfPotentialBlocks, maxAvailableOpponentPawnForBlocking)
            
#get not hidden pawns
#return dictionary of kind of pawn and the number of unhidden pawns from that kind
def getAvailablePawns(state, agent_id):
    pawns = {"B": 2, "M": 2, "S": 2}
    if(agent_id == 0):
        for key, value in state.player1_pawns.items():#key = pawn, value = (not_on_board, "B")
            if is_hidden(state, agent_id, key):
                pawns[value[1]] -= 1
    if(agent_id == 1):
        for key, value in state.player2_pawns.items():
            if is_hidden(state, agent_id, key):
                pawns[value[1]] -= 1
            
    return pawns

class SPOTS:
    ROW = 0
    COL = 1
    MAIN_DIAGONAL = 2
    SECONDARY_DIAGONAL = 3

def checkIthRow(matrixOpponent, row):
    colomns = [0, 1, 2]
    for col in range(3):
        if matrixOpponent[row][col] != " " and col in colomns:
            colomns.remove(col)
    if len(colomns) == 1:#only 1 spot is empty
        return row
    else:
        return -1
    
def checkIthCol(matrixOpponent, col):
    rows = [0, 1, 2]
    for row in range(3):
        if matrixOpponent[row][col] != " " and row in rows:
            rows.remove(row)
    if len(rows) == 1:#only 1 spot is empty
        return col
    else:
        return -1
    
def checkMainDiagonal(matrixOpponent):
    options = [0, 1, 2]
    for i in range(3):
        if matrixOpponent[i][i] != " " and i in options:
            options.remove(i)
    if len(options) == 1:#only 1 spot is empty
        return options[0]
    else:
        return -1

def checkSecondaryDiagonal(matrixOpponent):
    options = [(0, 2), (1, 1), (2, 0)]
    if matrixOpponent[0][2] != " ":
        options.remove((0, 2))        
    if matrixOpponent[1][1] != " ":
        options.remove((1, 1))        
    if matrixOpponent[2][0] != " ":
        options.remove((2, 0))
    if len(options) == 1:#only 1 spot is empty
        return options[0][0]
    else:
        return -1
    
#paws is a dictionary    
def getMaxPawn(pawns):
    maxPawn = 'S'
    if(pawns['B'] > 0):
        maxPawn = 'B'
    elif(pawns['M'] > 0):
        maxPawn = 'M'
    else:
        maxPawn = 'S'
    return maxPawn    
  
def evaluateSpot(currPawn, maxAvailableOpponentPawn):
    #print("BLOCK: currPawn: " + currPawn + ", maxAvailableOpponentPawn: " + maxAvailableOpponentPawn)
    value = 0
    isBigger = gge.size_cmp(currPawn, maxAvailableOpponentPawn)
    if(isBigger == 1):#availableMaxPawn is bigger than availableMaxPawnOpponent - can block the opponent, must put the max pawn
        value = 150
    elif(isBigger == 0):#availableMaxPawn is equal to availableMaxPawnOpponent - can block the opponent, must put the max pawn
        value = 100
    else:
        value = 10
    # print("value: {}".format(value))
    # print("###")
    return value
  
#check if the remaining spot is empty or if it is smaller than the max pawn of the current player    
def evaluateBlocks(matrixCurrPlayer, matrixOfBlocks, maxAvailableOpponentPawnForBlocking):
    value = 0
    #print("block")
    for i in range(3):
        for j in range(3):          
            if matrixCurrPlayer[i][j] != " " and matrixOfBlocks[i][j] == 1:
                value += evaluateSpot(matrixCurrPlayer[i][j], maxAvailableOpponentPawnForBlocking)                    
    #print("end evaluateBlocks")
    return value  
    
#remove pawns that are on the same row/col/diagonal that brings to win
#matrixCurrPlayer is the matrix of the current player of pawns that are not hidden
def getAvailablePawnsToBlock(matrix, matrixOpponent):
    pawns = {"B": 2, "M": 2, "S": 2}
    for i in range(3):
        for j in range(3):
            if matrix[i][j] == 1 and matrixOpponent[i][j] != " ":
                pawns[matrixOpponent[i][j]] -= 1  
    
    return pawns

def cornerMove(matrixCurrPlayer):
    cornereMoves = [[0,0], [0,2], [2,0], [2,2]]
    num = 0
    for location in cornereMoves:
        if matrixCurrPlayer[location[0], location[1]] == "S":
            num += 1
        if matrixCurrPlayer[location[0], location[1]] == "M":
            num += 3
        if matrixCurrPlayer[location[0], location[1]] == "B":
            num += 5
    return num    

def centerMove(matrixCurrPlayer):
    num = 0
    if matrixCurrPlayer[1, 1] == "S":
        num = 1
    if matrixCurrPlayer[1, 1] == "M":
        num = 3
    if matrixCurrPlayer[1, 1] == "B":
        num = 5
    return num

#did not use it in the end
def edgeMove(matrixCurrPlayer):
    edgeMoves = [[0,1], [1,0], [1,2], [2,1]]
    num = 0
    for location in edgeMoves:
        if matrixCurrPlayer[location[0], location[1]] == "S":
            num = 1
        if matrixCurrPlayer[location[0], location[1]] == "M":
            num = 3
        if matrixCurrPlayer[location[0], location[1]] == "B":
            num = 5
    return num    

def calculateHeuristic(valuesState, agent_id, matrixCurrPlayer, opponent_agent_id):
    heuristic = 0
    
    heuristic += valuesState[agent_id]["potentialBlocks"]
    #צheuristic -= valuesState[opponent_agent_id]["potentialBlocks"]
    
    heuristic += valuesState[agent_id]["potentialWins"]
    #heuristic -= valuesState[opponent_agent_id]["potentialWins"]    
    
    heuristic += valuesState[agent_id]["exposedPawns"] #need to consider the kind of pawn?
    heuristic -= valuesState[opponent_agent_id]["exposedPawns"] #need to consider the kind of pawn?
    
    if (valuesState[agent_id]["exposedPawns"] == 1 and valuesState[opponent_agent_id]["exposedPawns"] == 1) or (valuesState[agent_id]["exposedPawns"] == 1 and valuesState[opponent_agent_id]["exposedPawns"] == 0):
        heuristic += centerMove(matrixCurrPlayer)
        heuristic += cornerMove(matrixCurrPlayer)   
    
    return heuristic

#returns dictionary of values [win, num of potential wins in one step, num of exposed pawns] for each player for a given state
#potential wins - number of ways to win in one step
def smart_heuristic(state, agent_id):            
    winner = gge.is_final_state(state)
    if winner == 0:
        return -100
    elif winner != None and int(winner) - 1 == agent_id:
        return 1000
    elif  winner != None and int(winner) - 1 != agent_id:
        return -1000
    else:
        opponent_agent_id = (agent_id + 1) % 2
        matrixCurrPlayer = stateToMatrix(state, agent_id)
        matrixOpponent = stateToMatrix(state, opponent_agent_id)
        matrixOfPotentialBlocks = createMatrixOfPotentialBlocks(matrixOpponent)
        matrixOfPotentialBlocksOpponent = createMatrixOfPotentialBlocks(matrixCurrPlayer)

        # printMatrix(matrixCurrPlayer)
        # printMatrix(matrixOpponent)
        # printMatrix(matrixOfPotentialBlocks)
        # printMatrix(matrixOfPotentialBlocksOpponent)
        
        values = {}
        values[agent_id] = {"potentialBlocks": 0, "potentialWins": 0, "exposedPawns": 0}
        values[opponent_agent_id] = {"potentialBlocks": 0, "potentialWins": 0, "exposedPawns": 0}

        values[agent_id]["potentialBlocks"] = evaluatePotentialBlocks(matrixOpponent, matrixCurrPlayer, matrixOfPotentialBlocks)
        values[agent_id]["potentialWins"] = evaluatePotentialWins(state, agent_id, matrixCurrPlayer, matrixOpponent, matrixOfPotentialBlocksOpponent)#in 2 steps
        values[agent_id]["exposedPawns"] = dumb_heuristic2(state, agent_id)
        #print("\n")
        values[opponent_agent_id]["potentialBlocks"] = evaluatePotentialBlocks(matrixCurrPlayer, matrixOpponent, matrixOfPotentialBlocksOpponent)
        values[opponent_agent_id]["potentialWins"] = evaluatePotentialWins(state, opponent_agent_id, matrixOpponent, matrixCurrPlayer, matrixOfPotentialBlocks)#in 2 steps
        values[opponent_agent_id]["exposedPawns"] = dumb_heuristic2(state, opponent_agent_id)
        
        # print(values[agent_id])
        # print(values[opponent_agent_id])

        return calculateHeuristic(values, agent_id, matrixCurrPlayer, opponent_agent_id)

##########################################class RBMinimax###################################################
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
        self.start_time = time.time()
        self.is_done_flag = False
     
    def run_rb_minimax(self):            
        if self.is_done():
           raise Exception("timeout") 
        if gge.is_final_state(self.state) or self.depth == 0:
            return (self.actionToThisState, self.heuristic(self.state, self.agent_id) + self.depth) # + self.depth to make sure that the agent will choose the shortest path to win
         
        neighbor_list = self.state.get_neighbors()
        if self.state.turn == self.agent_id:
            curr_max = -math.inf
            action_max = None            
            for neighbor in neighbor_list:
                actionToNextState = neighbor[0]
                next_state = neighbor[1]                                
                action, curr_heuristic = RBMinimax(next_state, self.agent_id, self.timeout - (time.time() - self.start_time), 
                                                   self.depth - 1, actionToNextState, self.alpha, self.beta).run_rb_minimax()
                if self.is_done():
                    raise Exception("timeout")
                
                if curr_heuristic > curr_max:
                    curr_max = curr_heuristic                        
                    action_max = actionToNextState
                if(self.alpha != None and self.beta != None):
                    self.alpha = max(self.alpha, curr_max)
                    if(curr_max >= self.beta):
                        return (actionToNextState, curr_max)
            
            return action_max, curr_max
        else:#turn != self.state.turn
            curr_min = math.inf
            action_min = None
            for neighbor in neighbor_list:
                actionToNextState = neighbor[0]
                next_state = neighbor[1]
                action, curr_heuristic = RBMinimax(next_state, self.agent_id, self.timeout - (time.time() - self.start_time), 
                                                   self.depth - 1, actionToNextState, self.alpha, self.beta).run_rb_minimax()
                
                if self.is_done():
                    raise Exception("timeout")
                
                if curr_heuristic < curr_min:
                    curr_min = curr_heuristic                        
                    action_min = actionToNextState
                if(self.alpha != None and self.beta != None):
                    self.beta = min(self.beta, curr_min)
                    if(curr_min <= self.alpha):
                        return (actionToNextState, curr_min)
            return action_min, curr_min         
        
    def checkTime(self):
        end_time = time.time()
        duration = end_time - self.start_time
        if(duration + 0.2 > self.timeout):
            self.is_done_flag = True
        self.timeout -= duration
        if(self.timeout < duration):
            self.is_done_flag = True
        
    def is_done(self):
        self.checkTime()
        return self.is_done_flag
    

############################################################################################################  
class RB_Expectimax:
    
    def __init__(self, state, agent_id, timeout, depth, actionToThisState, heuristic = smart_heuristic):
        self.state = state
        self.agent_id = agent_id
        self.timeout = timeout
        self.depth = depth
        self.actionToThisState = actionToThisState
        self.heuristic = heuristic
        self.start_time = time.time()
        self.is_done_flag = False
     
    def run_rb_expectimax(self):            
        if self.is_done():
           raise Exception("timeout")
        if gge.is_final_state(self.state) or self.depth == 0:
            return (self.actionToThisState, self.heuristic(self.state, self.agent_id) + self.depth) # + self.depth to make sure that the agent will choose the shortest path to win
        
        neighbor_list = self.state.get_neighbors()
        if self.state.turn == self.agent_id:
            curr_max = -math.inf
            action_max = None            
            for neighbor in neighbor_list:
                actionToNextState = neighbor[0]
                next_state = neighbor[1]
                action, curr_heuristic = RB_Expectimax(next_state, self.agent_id, self.timeout - (time.time() - self.start_time), 
                                                        self.depth - 1, actionToNextState, self.heuristic).run_rb_expectimax()
                if self.is_done():
                    raise Exception("timeout")

                if curr_heuristic > curr_max:
                    curr_max = curr_heuristic                        
                    action_max = actionToNextState                
            return action_max, curr_max
        else:#turn != self.state.turn
            specialCases = []
            for neighbor in neighbor_list:
                next_state = neighbor[1]
                specialCases.append(self.probabilityOfState(self.state, next_state, self.agent_id))
            sumSpecialCases = 0
            for case in specialCases:
                sumSpecialCases += case #sumOfProbabilities is at least in the length of neighbor_list                            
            
            sum = 0
            curr_min = math.inf
            action_min = None
            for i, neighbor in enumerate(neighbor_list):
                actionToNextState = neighbor[0]
                next_state = neighbor[1]
                action, curr_heuristic = RB_Expectimax(next_state, self.agent_id, self.timeout - (time.time() - self.start_time), 
                                                            self.depth - 1, actionToNextState, self.heuristic).run_rb_expectimax()
                if self.is_done():
                    raise Exception("timeout")
                
                curr_heuristic = (specialCases[i]/sumSpecialCases) * curr_heuristic    
                if curr_heuristic < curr_min:
                    curr_min = curr_heuristic                        
                    action_min = actionToNextState
                    
                sum += curr_heuristic
            return action_min, sum #doesn't matter what action we return here
                              
    def checkTime(self):
        end_time = time.time()
        duration = end_time - self.start_time
        if(duration + 0.2 > self.timeout):
            self.is_done_flag = True
        self.timeout -= duration
        if(self.timeout < duration):
            self.is_done_flag = True
        
    def is_done(self):
        self.checkTime()
        return self.is_done_flag
    
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
############################################################################################################

