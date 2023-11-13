# 參考資料來自 ｢蒙特卡洛树搜索（MCTS）代码详解【python】｣ https://blog.csdn.net/windowsyun/article/details/88770799  
# ｢AlphaZero五子棋网络模型【python】｣ https://blog.csdn.net/windowsyun/article/details/88855277
import numpy as np
import copy 
from othello.OthelloUtil import *
from othello.OthelloGame import *
import time


# 設定棋盤大小
n = 12
show_detail = False

def rollout(board:OthelloGame):
    valids = getValidMoves(board, board.current_player)
    
    if len(valids) == 0:
        board.current_player = -board.current_player
        # 所有合法步列
        valids = getValidMoves(board, board.current_player)
    # 隨機給他一個合法步
    position = np.random.array.chioce(board, board.current_player)
    position = np_2_str(position)
    return position


def rollout_fn(board:OthelloGame):
    '''
    這個 rollout_fn 函數是模擬過程中使用的策略函數，它決定了在模擬過程中如何進行隨機探索。在這裡，它簡單地分配了均勻的機率給所有可行的行動。

    讓我們逐一解釋函數的結構：

    python
    Copy code
    def rollout_fn(board: OthelloGame):
        action_probs = np.ones(len(board.availables())) / len(board.availables())
        return zip(board.availables(), action_probs)
    rollout_fn 函數接受一個 OthelloGame 對象作為參數，代表當前的遊戲狀態。

    np.ones(len(board.availables())) / len(board.availables()) 這一部分創建一個機率分佈，其中所有合法行動的機率均相等。np.ones 生成一個元素均為 1 的 NumPy 陣列，然後除以合法行動的總數，以確保機率總和為 1。

    zip(board.availables(), action_probs) 這一部分將合法行動和對應的機率配對起來，返回一個可迭代對象。這樣，模擬過程中就可以從這個配對中進行隨機選擇行動。

    總體而言，這個 rollout_fn 函數是一個簡單的策略，對所有合法行動採用相同的均勻機率。這樣做的目的是在模擬過程中進行隨機探索，以更全面地探索可能的遊戲狀態。
    '''
    action_probs = np.ones(len(board.availables())) / len(board.availables())
    return zip(board.availables(), action_probs)


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, player_color, current_player):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self.player_color = player_color
        self._current_player = current_player
    
        

    def expand(self, action_priors, next_player):
        """Expand tree by creating new children.
        action_priors -- output from policy function - a list of tuples of actions
            and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if np_2_str(action) not in self._children:
                self._children[np_2_str(action)] = TreeNode(self, prob, next_player, self.player_color)

    def select(self, c_puct):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    # def update(self, leaf_value):
    #     """Update node values from leaf evaluation.
    #     """
    #     # Count visit.
    #     self._n_visits += 1
    #     # Update Q, a running average of values for all visits.
    #     self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    # def update_recursive(self, leaf_value):
    #     """Like a call to update(), but applied recursively for all ancestors.
    #     """
    #     # If it is not root, this node's parent should be updated first.

    #     if self._parent:
    #         self._parent.update_recursive(-leaf_value)
    #     self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000):
        """Arguments:
        policy_value_fn -- a function that takes in a board state and outputs a list of (action, probability)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from 
            the current player's perspective) for the current player.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        

    def _playout(self, state: OthelloGame):
        """Run a single playout from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        node = self._root
        
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
        action_probs, _ = self._policy(state)
        # Check for end of game
        end = isEndGame(state)
        if not end:
            # print("expand")
            node.expand(action_probs, state.current_player)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.back_update(leaf_value)

        # # Evaluate the leaf using a network which outputs a list of (action, probability)
        # # tuples p and also a score v in [-1, 1] for the current player.
        # action_probs, leaf_value = self._policy(state)

        # # Check for end of game.
        # end, winner = state.game_end()
        # if not end:
        #     node.expand(action_probs)
        # else:
        #     # for end state，return the "true" leaf_value
        #     if winner == -1:  # tie
        #         leaf_value = 0.0
        #     else:
        #         leaf_value = 1.0 if winner == state.get_current_player() else -1.0
            
        # # Update value and visit count of nodes in this traversal.
        # node.update_recursive(-leaf_value)
        
    def _evaluate_rollout(self, state: OthelloGame, limit=1000): 
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        for i in range(limit):
            end = isEndGame(state)
            if end is not None:
                break
            position = rollout_fn(state)
            # max_action = max(action_probs, key=itemgetter(1))[0]
            state.move(position)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if end == 0:  # tie
            return 0
        else:
            return 1 if end == self.player_color else -1
        
    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the available actions and their corresponding probabilities 
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities 
        """        
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
            
        # calc the move probabilities based on the visit counts at the root node
        # act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        # acts, visits = zip(*act_visits)
        # act_probs = softmax(1.0/temp * np.log(visits))       
         
        # return acts, act_probs
        



    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        # print("OK")
        for n in range(self._n_playout):
            # state_copy = copy.deepcopy(state)
            # print(state_copy.current_player)
            self._playout(copy.deepcopy(state))
            # print(n)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]
    

    # def update_with_move(self, last_move):
    #     """Step forward in the tree, keeping everything we already know about the subtree.
    #     """
    #     if last_move in self._root._children:
    #         self._root = self._root._children[last_move]
    #         self._root._parent = None
    #     else:
    #         self._root = TreeNode(None, 1.0)
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


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                c_puct=5, n_playout=1000, is_selfplay=12):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self.player = None
        self.game = OthelloGame(n)
        self.n = n
        
    def set_player_ind(self, p):
        self.player = p

    # def reset_player(self):
    #     self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
                # location = board.move_to_location(move)
                # print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        def __str__(self):
            return "MCTS {}".format(self.player)