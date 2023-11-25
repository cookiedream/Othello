from othello.OthelloGame import *
from othello.bots.Random import BOT as RandomBOT
from othello.bots.mcts import MCTSPlayer as mctspalyer
from othello.bots.mcts_1 import MCTS_Player as mctspalyer1
n = 12

def self_play(black, white, verbose=True):
    g = OthelloGame(n=n)
    result = g.play(black, white, verbose)
    return result

def main():
    n_game = 5
    bot1_win = 0
    bot2_win = 0
    # bot1_name = "RandomBOT"
    # bot1 = RandomBOT()
    # bot1_name = "mctspalyer"
    # bot1 = mctspalyer(n_playout=30, n=n , start_time=0.5)
    bot1_name = "mctspalyer1"
    bot1 = mctspalyer1(n_playout=300, n=n)
    
    bot2_name = "MCTS_PureBOT"
    # bot2 = MCTS_PureBOT(n_playout=20, n=n)
    bot2 = RandomBOT()
    for i in range(n_game):
        print("Game {}".format(i+1))
        result = self_play(bot1, bot2, verbose=False)
        if result == BLACK:
            bot1_win += 1
        elif result == WHITE:
            bot2_win += 1

        result = self_play(bot2, bot1, verbose=False)
        if result == BLACK:
            bot2_win += 1
        elif result == WHITE:
            bot1_win += 1
        with open("result.txt", "a") as f:
            f.write("Game {}\n".format(i+1))
            f.write("{} win: {}\n".format(bot1_name, bot1_win))
            f.write("{} win: {}\n".format(bot2_name, bot2_win))
            f.write("----------------------------------------------------------------------------\n")
        print("{} win: {}".format(bot1_name, bot1_win))
        print("{} win: {}".format(bot2_name, bot2_win))
        print("----------------------------------------------------------------------------")
if __name__ == '__main__':

    main()
