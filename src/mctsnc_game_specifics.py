from numba import cuda

@cuda.jit(device=True)
def is_action_legal(m, n, board, extra_info, turn, action, legal_actions):
    #is_action_legal_gomoku(m, n, board, extra_info, turn, action, legal_actions)
    is_action_legal_c4(m, n, board, extra_info, turn, action, legal_actions)

@cuda.jit(device=True)
def take_action(m, n, board, extra_info, turn, action):
    #take_action_gomoku(m, n, board, extra_info, turn, action)
    take_action_c4(m, n, board, extra_info, turn, action)

@cuda.jit(device=True)
def legal_actions_playout(m, n, board, extra_info, turn, legal_actions_with_count):
    #legal_actions_playout_gomoku(m, n, board, extra_info, turn, legal_actions_with_count)
    legal_actions_playout_c4(m, n, board, extra_info, turn, legal_actions_with_count)

@cuda.jit(device=True)    
def take_action_playout(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count):
    #take_action_playout_gomoku(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count)    
    take_action_playout_c4(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count)
    
@cuda.jit(device=True)
def compute_outcome(m, n, board, extra_info, turn, last_action): # any outcome other than {-1, 0, 1} implies status: game ongoing
    #return compute_outcome_gomoku(m, n, board, extra_info, turn, last_action)
    return compute_outcome_c4(m, n, board, extra_info, turn, last_action)

@cuda.jit(device=True)
def is_action_legal_c4(m, n, board, extra_info, turn, action, legal_actions):
    legal_actions[action] = True if extra_info[action] < m else False
    
@cuda.jit(device=True)
def take_action_c4(m, n, board, extra_info, turn, action):
    extra_info[action] += 1
    row = m - extra_info[action]
    board[row, action] = turn

@cuda.jit(device=True)
def legal_actions_playout_c4(m, n, board, extra_info, turn, legal_actions_with_count):
    count = 0
    for j in range(n):
        if extra_info[j] < m:            
            legal_actions_with_count[count] = j
            count += 1
    legal_actions_with_count[-1] = count

@cuda.jit(device=True)
def take_action_playout_c4(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count):
    extra_info[action] += 1
    row = m - extra_info[action]
    board[row, action] = turn

@cuda.jit(device=True)
def compute_outcome_c4(m, n, board, extra_info, turn, last_action):    
    last_token = -turn    
    j = last_action            
    i = m - extra_info[j]
    # N-S
    total = 0
    for k in range(1, 4 + 1):
        if i -  k < 0 or board[i - k, j] != last_token:
            break
        total += 1
    for k in range(1, 4 + 1):
        if i + k >= m or board[i + k, j] != last_token:
            break            
        total += 1
    if total >= 3:            
        return last_token            
    # E-W
    total = 0
    for k in range(1, 4 + 1):
        if j + k >= n or board[i, j + k] != last_token:
            break
        total += 1
    for k in range(1, 4 + 1):
        if j - k < 0 or board[i, j - k] != last_token:
            break            
        total += 1
    if total >= 3:        
        return last_token            
    # NE-SW
    total = 0
    for k in range(1, 4 + 1):
        if i - k < 0 or j + k >= n or board[i - k, j + k] != last_token:
            break
        total += 1
    for k in range(1, 4 + 1):
        if i + k >= m or j - k < 0 or board[i + k, j - k] != last_token:
            break
        total += 1
    if total >= 3:
        return last_token            
    # NW-SE
    total = 0
    for k in range(1, 4 + 1):
        if i - k < 0 or j - k < 0 or board[i - k, j - k] != last_token:
            break
        total += 1
    for k in range(1, 4 + 1):
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
    i = action // n
    j = action % n
    legal_actions[action] = (board[i, j] == 0)
    
@cuda.jit(device=True)
def take_action_gomoku(m, n, board, extra_info, turn, action):
    i = action // n
    j = action % n
    board[i, j] = turn

@cuda.jit(device=True)
def legal_actions_playout_gomoku(m, n, board, extra_info, turn, legal_actions_with_count):
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
    i = action // n
    j = action % n
    board[i, j] = turn    
    last_legal_action = legal_actions_with_count[legal_actions_with_count[-1] - 1]
    legal_actions_with_count[action_ord] = last_legal_action
    legal_actions_with_count[-1] -= 1            

@cuda.jit(device=True)
def compute_outcome_gomoku(m, n, board, extra_info, turn, last_action):    
    last_token = -turn    
    i = last_action // n
    j = last_action % n
    # N-S
    total = 0
    for k in range(1, 6 + 1):
        if i -  k < 0 or board[i - k, j] != last_token:
            break
        total += 1
    for k in range(1, 6 + 1):
        if i + k >= m or board[i + k, j] != last_token:
            break            
        total += 1
    if total == 4:         
        return last_token            
    # E-W
    total = 0
    for k in range(1, 6 + 1):
        if j + k >= n or board[i, j + k] != last_token:
            break
        total += 1
    for k in range(1, 6 + 1):
        if j - k < 0 or board[i, j - k] != last_token:
            break            
        total += 1
    if total == 4:        
        return last_token            
    # NE-SW
    total = 0
    for k in range(1, 6 + 1):
        if i - k < 0 or j + k >= n or board[i - k, j + k] != last_token:
            break
        total += 1
    for k in range(1, 6 + 1):
        if i + k >= m or j - k < 0 or board[i + k, j - k] != last_token:
            break
        total += 1
    if total == 4:
        return last_token            
    # NW-SE
    total = 0
    for k in range(1, 6 + 1):
        if i - k < 0 or j - k < 0 or board[i - k, j - k] != last_token:
            break
        total += 1
    for k in range(1, 6 + 1):
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