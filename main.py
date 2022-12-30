import game

if __name__ == '__main__':
    print("YAY LET'S START RUNNING STUFF")

    # PART1
    # game.play_game("human", "human")

    # PART2
    #game.play_tournament("greedy", "random", 50)
    #game.play_tournament("greedy_improved", "random", 10)
    #game.play_tournament("greedy", "greedy_improved", 50)
    
    # PART3
    game.play_tournament("minimax", "random", 1)
    #game.play_tournament("minimax", "greedy_improved", 50)
    #game.play_tournament("minimax", "greedy", 50)
    
    # PART4
    # game.play_tournament("alpha_beta", "random", 1)
    # game.play_tournament("alpha_beta", "greedy_improved", 1)
    # game.play_tournament("alpha_beta", "greedy", 1)
    # game.play_tournament("alpha_beta", "minimax", 1)
    
    # PART5
    # game.play_tournament("expectimax", "random", 1)
    # game.play_tournament("expectimax", "greedy_improved", 1)
    # game.play_tournament("expectimax", "greedy", 1)
    # game.play_tournament("expectimax", "minimax", 1)
    # game.play_tournament("expectimax", "alpha_beta", 1)
