# 參考資料來自 ｢蒙特卡洛树搜索（MCTS）代码详解【python】｣ https://blog.csdn.net/windowsyun/article/details/88770799  
# 參考資料來自 ｢AlphaZero五子棋网络模型【python】｣ https://blog.csdn.net/windowsyun/article/details/88855277


import numpy as np
# from othello.OthelloUtil import *
from othello.OthelloGame import *
from othello.Cy_OthelloUtil import *
import copy
import time


def rpf(board: OthelloGame):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    # action_probs = np.random.rand(len(board.availables))
    # return zip(board.availables, action_probs)
    valids = getValidMoves(board, board.current_player) # 取得合法步列表
    # print(board)
    # print(valids)
    if len(valids) == 0:
        board.current_player = -1. * board.current_player
        valids = getValidMoves(board, board.current_player)
    position = np.random.choice(range(len(valids)), size=1)[0] # 隨機選擇其中一合法步
    position = valids[position]
    return position


def pvf(board: OthelloGame):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    # print(board)
    # valids = getValidMoves(board, 1)
    # print(valids)
    # print(board.availables())
    return zip(board.availables(), np.ones(len(board.availables())) / len(board.availables())), 0



class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, curr_player, player_color):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self.curr_player = curr_player
        self.player_color = player_color

    def expand(self, action_priors, next_player):
        """
            action_priors: (action, prob)， 所有可以出的牌和其機率
            next_player: 下一個出牌的人
            根據以上參數擴展節點
        """
        for action, prob in action_priors:
            if np_2_str(action) not in self._children:
                self._children[np_2_str(action)] = TreeNode(self, prob, next_player, self.player_color)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """

        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))

        return self._Q + self._u

    def back_update(self, leaf_value):
        """
            反向去更所有直系節點的Q值和訪問次數
            leaf_value: 從葉節點到當前節點的分數，leaf_value自己的分數，若當前節點為對手，則leaf_value為13-leaf_value
        """
        curr_value = leaf_value
        self._n_visits += 1

        if self.curr_player != self.player_color:
            curr_value = -1. * curr_value

        # 更新Q值
        self._Q += 2.9 * (curr_value - self._Q) / self._n_visits

        if self._parent:
            self._parent.back_update(leaf_value)

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, pvf, c_puct=5, n_playout=1000, time_limit=2.5):
        """
        pvf: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = None
        self.player_color = None
        self._policy = pvf
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.board = None
        self.n = None
        self.time_limit = time_limit
        
    def reset_root(self, player_color):
        self.player_color = player_color
        self._root = TreeNode(None, 2.9, -player_color, player_color) # 第一個節點一定是對手的節點
    
    
    def _playout(self, state: OthelloGame):
        """
        一次模擬, 會選定一個葉節點, 並從葉節點開始模擬, 會先擴展一次, 直到遊戲結束, 最後更新分數。
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root

        action_probs, _ = self._policy(state)
        # Check for end of game
        end = isEndGame(state)
        if not end:
            # print("expand")
            node.expand(action_probs, state.current_player)
        # Evaluate the leaf node by random rollout
        leaf_value = self.evaluate(state)
        # Update value and visit count of nodes in this traversal.
        node.back_update(leaf_value)

    def evaluate(self, state: OthelloGame, limit=1000): 
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        start_time = time.time()
        for i in range(limit):
            end = isEndGame(state)
            if end is not None:
                break
            position = rpf(state)
            # max_action = max(action_probs, key=itemgetter(1))[0]
            state.move(position)
            if time.time() - start_time > self.time_limit:
                break
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if end == 0:  # tie
            return 0
        else:
            return 1 if end == self.player_color else -1


    # def get_move(self, new_board: OthelloGame , mycolor):
    #     start_time = time.time() # 計算模擬時間
    #     self.player_color = mycolor # 設定我方顏色
    #     self.mcts = MCTS(pvf , self._c_puct, self._n_playout)

    #     if self.mcts._root is None:
    #         self.mcts.reset_root(mycolor) # 重置tree
    #     else:
    #         oppo_move = find_opp_move(self.board, new_board, mycolor=mycolor, to_str=True) # 取得對手落子(str)   
    #         self.mcts.update_with_move(oppo_move, mycolor) # 更新tree至當前狀態
            
    #     self.board[:] = new_board[:] # 更新盤面
        
    #     self.board.curr_player = self.player_color # 設定當前玩家
    #     move = self.mcts._get_move(self.board, start_time) # 取得MCTS的落子(str)
        
    #     move_position = (move // self.n, move % self.n) # 轉換成np.array
    #     self.board.move(move_position)  # 更新盤面(我方落子)
    #     self.mcts.update_with_move([move], mycolor) # 更新tree至當前狀態
    #     return move_position
    
    def get_move(self, state): # 修正這裡，傳遞 mycolor 參數
            """Runs all playouts sequentially and returns the most visited action.
            state: the current game state

            Return: the selected action
            """
            start_time = time.time()
            for n in range(self._n_playout): # 進行n_playout次模擬
                # state_copy = copy.deepcopy(state)
                # print(state_copy.current_player) 
                self._playout(copy.deepcopy(state)) # 進行一次模擬
                # print(n)
                if time.time() - start_time > self.time_limit:
                    break
            return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0] # 取得最佳落子

    def update_with_move(self, last_move, player_color):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        last_move: 更新tree至當前狀態(str) (list)(可能不只一步)
        """
        update = False
        for _ in range(len(last_move)):
            for child in self._root._children:
                if child in last_move:
                    update = True
                    self._root = self._root._children[child]
                    self._root._parent = None
                    last_move.remove(child)
                    break
        if (not update) or (update and len(last_move) > 0):
            self.reset_root(player_color)

    def __str__(self):
        return "MCTS"


class MCTS_Player(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=50, n=12, time_limit=2.5):
        self.mcts = MCTS(pvf, c_puct, n_playout, time_limit)
        self.game = OthelloGame(n)
        self.player = None
        self.n = n

    def getAction(self, board, color):
        self.player = color # 設定我方顏色
        # 重置tree
        self.mcts.reset_root(color)
        # 判斷是否已經進行下一局
        if judge_next_game(self.game, board):
            self.game = OthelloGame(n=self.n)
        # 更新tree至當前狀態
        # self.mcts.update_with_move(find_opp_move(self.game, board, to_str=True), color)
        self.game[:] = board[:] # 更新盤面
        # print(type(self.game))

        print(self.game.showBoard())
        # 更新盤面(對手落子直到我方有合法步)
        # print(find_opp_move(self.game, board), self.player)
        # for opp_position in find_opp_move(self.game, board):
        #     self.game.do_move(opp_position, -self.player)

        self.game.current_player = self.player # 設定當前玩家
        # print(self.game.current_player)

        move_str = self.mcts.get_move(self.game) # 取得MCTS的落子(str)
        # print(move_str)
        move = str_2_np(move_str) # 轉換成np.array
        self.game.move(move) # 更新盤面(我方落子)
        # print(move)
        print(self.game.showBoard())
        print(move)
        print("=========================================================")

        # self.mcts.update_with_move([move_str], color) # 更新tree至當前狀態
        return move

    def __str__(self):
        return "MCTS {}".format(self.player)
