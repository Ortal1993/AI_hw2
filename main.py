import game

if __name__ == '__main__':
    print("YAY LET'S START RUNNING STUFF")

    # PART1
    # game.play_game("human", "human")
    #game.play_game("greedy_improved", "human")

    # PART2
    #game.play_tournament("greedy", "human", 10)
    #game.play_tournament("greedy", "random", 10)
    #game.play_tournament("greedy_improved", "random", 10)
    #game.play_tournament("greedy", "greedy_improved", 1)
    
    # PART3
    #game.play_tournament("human", "minimax", 1)
    #game.play_tournament("minimax", "random", 1)
    #game.play_tournament("minimax", "greedy_improved", 1)
    #game.play_tournament("minimax", "greedy", 50)
    
    # PART4
    #game.play_tournament("alpha_beta", "human", 1)
    #game.play_tournament("alpha_beta", "random", 1)    
    #game.play_tournament("alpha_beta", "greedy_improved", 1)
    #game.play_tournament("alpha_beta", "greedy", 1)
    #game.play_tournament("alpha_beta", "minimax", 1)
    
    # PART5
    game.play_tournament("expectimax", "human", 1)
    #game.play_tournament("expectimax", "random", 1)
    #game.play_tournament("expectimax", "greedy_improved", 1)
    #game.play_tournament("expectimax", "greedy", 1)
    #game.play_tournament("expectimax", "minimax", 1)
    #game.play_tournament("expectimax", "alpha_beta", 1)

    # PART6 - BONUS