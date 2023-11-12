from othello.OthelloGame import *
from othello.bots.Random import BOT as RandomBOT
from othello.bots.mcts_pure import MCTS_BOT as MCTS_PureBOT
from othello.bots.alpha_beta import Alpha_Beta_BOT

n = 12

def self_play(black, white, verbose=True):
    g = OthelloGame(n=n)
    result = g.play(black, white, verbose)
    return result

def main():
    n_game = 100
    bot1_win = 0
    bot2_win = 0
    bot1_name = "RandomBOT"
    bot1 = RandomBOT()
    # bot1_name = "mcts_bot_100"
    # bot1 = MCTS_PureBOT(n_playout=100, n=n)
    
    bot2_name = "mcts_bot_100"
    bot2 = MCTS_PureBOT(n_playout=1, n=n)

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