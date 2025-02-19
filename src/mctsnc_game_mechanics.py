"""
Set of five CUDA device functions defining the mechanics of a certain game (Connect 4, Gomoku, etc.) 
required by the class ``MCTSNC`` (callable by its kernel functions) from :doc:`mctsnc`. 
The five functions are: ``is_action_legal``, ``take_action``, ``legal_actions_playout``, ``take_action_playout``, ``compute_outcome``.
To define a new custom game or a search problem the user should provide his implementations either directly as bodies of the aforementioned functions, 
or write his own device functions and forward the calls.
Currently, the module contains examples of how those functions are implemented for the games of Connect 4 and Gomoku.

Function ``is_action_legal`` is called by each of ``_expand_1_*`` kernel functions from ``MCTSNC`` class;
function ``take_action`` is called by each of ``_expand_2_*`` kernel functions; 
functions ``legal_actions_playout`` and ``take_action_playout`` are called interchangeably by each of ``_playout_*`` kernel functions;
function ``compute_outcome`` is called by each of ``_expand_2_*`` and ``_playout_*`` kernel functions.

The following arguments are common for all the functions:

    m (int): 
        number of rows in board.
    n (int): 
        number of columns in board.
    board (array[int8, ndim=2] shared or local):
        two-dimensional array of bytes representing the board of a state.
    extra_info (array[int8, ndim=1] shared or local):
        one-dimensional array with any additional information associated with a state (not implied by the contents of the board itself).
    turn {-1, 1}:
        indicator of the player, minimizing or maximizing, to act now.
        
The following arguments are function-specific:

    legal_actions (array[boolean] shared): 
        one-dimensional array of boolean flags indicating legal actions; becomes populated by multiple calls of ``is_action_legal`` made by an ``_expand_*`` kernel function of ``MCTSNC`` class.
    action (int16):
        index of action to be taken
    legal_actions_with_count (array[int16] local):
        array storing legal actions with their count to be applied within a playout; 
        its last entry contains the count of legal actions, its leftmost entries (in the number equal to that count) contain indexes of legal actions (possibly unordered); 
        becomes established within calls of ``legal_actions_playout`` or just the first such a call made by a ``_playout_*`` kernel function of ``MCTSNC`` class; 
        can be updated (but does not have to) within calls of ``take_action_playout`` to avoid future costs of legal moves regeneration during ``legal_actions_playout``.
    action_ord (int16):
        ordinal index of entry in array ``legal_actions_with_count``, picked on random by a ``xoroshiro128p`` generator within a ``_playout_*`` kernel functions of ``MCTSNC`` class;
        this entry defines the index of action to be currently taken during a playout, i.e., ``legal_actions_with_count[action_ord] == action``;
        can be used (but does not have to) within calls of ``take_action_playout`` to avoid future costs of legal moves regeneration during ``legal_actions_playout`` calls
        by placing the last available legal action (rightmost) under ``action_ord`` index just after the current action is taken.  
                            

Link to project repository
--------------------------
`https://github.com/pklesk/mcts_numba_cuda <https://github.com/pklesk/mcts_numba_cuda>`_ 
"""

from numba import cuda

__version__ = "1.0.1"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl" 

@cuda.jit(device=True)
def is_action_legal(m, n, board, extra_info, turn, action, legal_actions):
    """Checks whether action defined by index ``action`` is legal and leaves the result (a boolean indicator) in array ``legal_actions`` under that index."""
    is_action_legal_c4(m, n, board, extra_info, turn, action, legal_actions)
    #is_action_legal_gomoku(m, n, board, extra_info, turn, action, legal_actions)    

@cuda.jit(device=True)
def take_action(m, n, board, extra_info, turn, action):
    """Takes action defined by index ``action`` during an expansion - modifies the ``board`` and possibly ``extra_info`` arrays."""    
    take_action_c4(m, n, board, extra_info, turn, action)
    #take_action_gomoku(m, n, board, extra_info, turn, action)

@cuda.jit(device=True)
def legal_actions_playout(m, n, board, extra_info, turn, legal_actions_with_count):
    """Establishes legal actions and their count during a playout; leaves the results in array ``legal_actions_with_count``."""
    legal_actions_playout_c4(m, n, board, extra_info, turn, legal_actions_with_count)    
    #legal_actions_playout_gomoku(m, n, board, extra_info, turn, legal_actions_with_count)    

@cuda.jit(device=True)    
def take_action_playout(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count):
    """Takes action defined by index ``action`` during a playout - modifies the ``board`` and possibly arrays: ``extra_info``, ``legal_actions_with_count``."""
    take_action_playout_c4(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count)
    #take_action_playout_gomoku(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count)
    
@cuda.jit(device=True)
def compute_outcome(m, n, board, extra_info, turn, last_action): # any outcome other than {-1, 0, 1} implies status: game ongoing
    """
    Computes and returns the outcome of game state represented by ``board`` and ``extra_info`` arrays. 
    Outcomes ``{-1, 1}`` denote a win by minimizing or maximizing player, respectively. ``0`` denotes a tie. Any other outcome denotes an ongoing game.
    """    
    return compute_outcome_c4(m, n, board, extra_info, turn, last_action)
    #return compute_outcome_gomoku(m, n, board, extra_info, turn, last_action)    

@cuda.jit(device=True)
def is_action_legal_c4(m, n, board, extra_info, turn, action, legal_actions):
    """Functionality of function ``is_action_legal`` for the game of Connect 4.""" 
    legal_actions[action] = True if extra_info[action] < m else False
    
@cuda.jit(device=True)
def take_action_c4(m, n, board, extra_info, turn, action):
    """Functionality of function ``take_action`` for the game of Connect 4."""
    extra_info[action] += 1
    row = m - extra_info[action]
    board[row, action] = turn

@cuda.jit(device=True)
def legal_actions_playout_c4(m, n, board, extra_info, turn, legal_actions_with_count):
    """Functionality of function ``legal_actions_playout`` for the game of Connect 4."""    
    count = 0
    for j in range(n):
        if extra_info[j] < m:            
            legal_actions_with_count[count] = j
            count += 1
    legal_actions_with_count[-1] = count

@cuda.jit(device=True)
def take_action_playout_c4(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count):
    """Functionality of function ``take_action_playout`` for the game of Connect 4."""
    extra_info[action] += 1
    row = m - extra_info[action]
    board[row, action] = turn

@cuda.jit(device=True)
def compute_outcome_c4(m, n, board, extra_info, turn, last_action):
    """Functionality of function ``compute_outcome`` for the game of Connect 4."""    
    last_token = -turn    
    j = last_action            
    i = m - extra_info[j]
    # N-S
    total = 0
    for k in range(1, 4):
        if i -  k < 0 or board[i - k, j] != last_token:
            break
        total += 1
    for k in range(1, 4):
        if i + k >= m or board[i + k, j] != last_token:
            break            
        total += 1
    if total >= 3:            
        return last_token
    # E-W
    total = 0
    for k in range(1, 4):
        if j + k >= n or board[i, j + k] != last_token:
            break
        total += 1
    for k in range(1, 4):
        if j - k < 0 or board[i, j - k] != last_token:
            break            
        total += 1
    if total >= 3:        
        return last_token            
    # NE-SW
    total = 0
    for k in range(1, 4):
        if i - k < 0 or j + k >= n or board[i - k, j + k] != last_token:
            break
        total += 1
    for k in range(1, 4):
        if i + k >= m or j - k < 0 or board[i + k, j - k] != last_token:
            break
        total += 1
    if total >= 3:
        return last_token            
    # NW-SE
    total = 0
    for k in range(1, 4):
        if i - k < 0 or j - k < 0 or board[i - k, j - k] != last_token:
            break
        total += 1
    for k in range(1, 4):
        if i + k >= m or j + k >= n or board[i + k, j + k] != last_token:
            break
        total += 1            
    if total >= 3:
        return last_token
    draw = True                                    
    for j in range(n):
        if extra_info[j] < m:
            draw = False
            break
    if draw:
        return 0
    return 2 # anything other than {-1, 0, 1} implies 'game ongoing'

@cuda.jit(device=True)
def is_action_legal_gomoku(m, n, board, extra_info, turn, action, legal_actions):
    """Functionality of function ``is_action_legal`` for the game of Gomoku."""
    i = action // n
    j = action % n
    legal_actions[action] = (board[i, j] == 0)
    
@cuda.jit(device=True)
def take_action_gomoku(m, n, board, extra_info, turn, action):
    """Functionality of function ``take_action`` for the game of Gomoku."""
    i = action // n
    j = action % n
    board[i, j] = turn

@cuda.jit(device=True)
def legal_actions_playout_gomoku(m, n, board, extra_info, turn, legal_actions_with_count):
    """Functionality of function ``legal_actions_playout`` for the game of Gomoku."""
    if legal_actions_with_count[-1] == 0: # time-consuming board scan only if legal actions not established yet
        count = 0 
        k = 0
        for i in range(m):
            for j in range(n):            
                if board[i, j] == 0:                
                    legal_actions_with_count[count] = k
                    count += 1
                k += 1
        legal_actions_with_count[-1] = count

@cuda.jit(device=True)
def take_action_playout_gomoku(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count):
    """Functionality of function ``take_action_playout`` for the game of Gomoku."""
    i = action // n
    j = action % n
    board[i, j] = turn    
    last_legal_action = legal_actions_with_count[legal_actions_with_count[-1] - 1]
    legal_actions_with_count[action_ord] = last_legal_action
    legal_actions_with_count[-1] -= 1            

@cuda.jit(device=True)
def compute_outcome_gomoku(m, n, board, extra_info, turn, last_action):
    """Functionality of function ``compute_outcome`` for the game of Gomoku."""    
    last_token = -turn    
    i = last_action // n
    j = last_action % n
    # N-S
    total = 0
    for k in range(1, 6):
        if i -  k < 0 or board[i - k, j] != last_token:
            break
        total += 1
    for k in range(1, 6):
        if i + k >= m or board[i + k, j] != last_token:
            break            
        total += 1
    if total == 4:         
        return last_token            
    # E-W
    total = 0
    for k in range(1, 6):
        if j + k >= n or board[i, j + k] != last_token:
            break
        total += 1
    for k in range(1, 6):
        if j - k < 0 or board[i, j - k] != last_token:
            break            
        total += 1
    if total == 4:        
        return last_token            
    # NE-SW
    total = 0
    for k in range(1, 6):
        if i - k < 0 or j + k >= n or board[i - k, j + k] != last_token:
            break
        total += 1
    for k in range(1, 6):
        if i + k >= m or j - k < 0 or board[i + k, j - k] != last_token:
            break
        total += 1
    if total == 4:
        return last_token            
    # NW-SE
    total = 0
    for k in range(1, 6):
        if i - k < 0 or j - k < 0 or board[i - k, j - k] != last_token:
            break
        total += 1
    for k in range(1, 6):
        if i + k >= m or j + k >= n or board[i + k, j + k] != last_token:
            break
        total += 1            
    if total == 4:
        return last_token
    draw = True
    for i in range(m):                                    
        for j in range(n):
            if board[i, j] == 0:
                draw = False
                break
    if draw:
        return 0
    return 2 # anything other than {-1, 0, 1} implies 'game ongoing'