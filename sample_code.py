from AIGamePlatform import Othello
from othello.bots.Random import BOT
from othello.bots.mcts_pure import MCTS_BOT
from othello.bots.mcts import MCTSPlayer
from othello.bots.mcts_1 import MCTS_Player
app=Othello() # 和平台建立WebSocket連線
# bot_mcts = MCTS_BOT(n_playout=30, n=12)
# mcts = MCTSPlayer(n_playout=27, n=12)
mcts1 = MCTS_Player(n_playout=30000, n=12)
bot = BOT()

@app.competition(competition_id='test_12x12') # 競賽ID
def _callback_(board, color): # 當需要走步會收到盤面及我方棋種
    # print(board, color)
    return mcts1.getAction(board, color) # bot回傳落子座標
    # return mcts1.getAction(board,color)

