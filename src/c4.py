import numpy as np
from mcts import State
from numba import jit
from numba import int8

class C4(State):
    M = 6
    N = 7 
    SYMBOLS = ['\u25CB', '.', '\u25CF'] # or: ["O", ".", "X"]    
    
    def __init__(self, parent=None):
        super().__init__(parent)
        if self.parent:
            self.board = np.copy(self.parent.board)
            self.column_fills = np.copy(self.parent.column_fills)
        else:
            self.board = np.zeros((C4.M, C4.N), dtype=np.int8)
            self.column_fills = np.zeros(C4.N, dtype=np.int8)
        self.last_move = None
            
    def __str__(self):
        s = ""
        for i in range(C4.M):
            s += "|"
            for j in range(C4.N):
                s += C4.SYMBOLS[self.board[i, j] + 1]
                s += "|"
            s += "\n"
        s += " "
        for j in range(C4.N):
            s += f"{j} "
        return s
    
    def __repr__(self):
        s = str(self)
        s += f"\n[n: {self.n}, n_wins: {self.n_wins}]"
        s += f"\n[turn: {self.turn}, outcome: {self.outcome}, n_children: {len(self.children)}]"
        return s    
    
    def move(self, move_index):
        j_index = move_index 
        if self.column_fills[j_index] == C4.M:
            return False
        i_index = C4.M - 1 - self.column_fills[j_index] 
        self.board[i_index, j_index] = self.turn
        self.column_fills[j_index] += 1
        self.turn *= -1
        self.last_move = (i_index, j_index)
        return True
    
    @staticmethod
    def move_down_tree_via(c4, move_index):
        if len(c4.children) > 0:
            return c4.children[move_index]
        child = C4(c4)
        move_valid = child.move(move_index) 
        if not move_valid:
            return None
        return child
    
    def get_outcome(self):
        if self.outcome_computed:
            return self.outcome
        if not self.last_move:
            return None
        self.outcome_computed = True # will be soon        
        if True: # a bit faster outcome via numba
            numba_outcome = C4.get_outcome_numba_jit(C4.M, C4.N, self.turn, self.last_move[0], self.last_move[1], self.board)
            if numba_outcome != 0:
                self.outcome = numba_outcome
                return self.outcome 
        else:
            last_token = -self.turn        
            i, j = self.last_move            
            # N-S
            total = 0
            for k in range(1, 4 + 1):
                if i -  k < 0 or self.board[i - k, j] != last_token:
                    break
                total += 1
            for k in range(1, 4 + 1):
                if i + k >= C4.M or self.board[i + k, j] != last_token:
                    break            
                total += 1
            if total >= 3:
                self.outcome = last_token            
                return last_token            
            # E-W
            total = 0
            for k in range(1, 4 + 1):
                if j + k >= C4.N or self.board[i, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 4 + 1):
                if j - k < 0 or self.board[i, j - k] != last_token:
                    break            
                total += 1
            if total >= 3:
                self.outcome = last_token
                return last_token            
            # NE-SW
            total = 0
            for k in range(1, 4 + 1):
                if i - k < 0 or j + k >= C4.N or self.board[i - k, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 4 + 1):
                if i + k >= C4.M or j - k < 0 or self.board[i + k, j - k] != last_token:
                    break
                total += 1            
            if total >= 3:
                self.outcome = last_token
                return last_token            
            # NW-SE
            total = 0
            for k in range(1, 4 + 1):
                if i - k < 0 or j - k < 0 or self.board[i - k, j - k] != last_token:
                    break
                total += 1
            for k in range(1, 4 + 1):
                if i + k >= C4.M or j + k >= C4.N or self.board[i + k, j + k] != last_token:
                    break
                total += 1            
            if total >= 3:
                self.outcome = last_token
                return last_token                                    
        if np.sum(self.board == 0) == 0: # draw
            self.outcome = 0
        return self.outcome    
    
    @staticmethod
    @jit(int8(int8, int8, int8, int8, int8, int8[:, :]), nopython=True, cache=True)  
    def get_outcome_numba_jit(M, N, turn, last_i, last_j, board):
        last_token = -turn        
        i, j = last_i, last_j
        # N-S
        total = 0
        for k in range(1, 4 + 1):
            if i - k < 0 or board[i - k, j] != last_token:
                break
            total += 1
        for k in range(1, 4 + 1):
            if i + k >= M or board[i + k, j] != last_token:
                break            
            total += 1
        if total >= 3:
            return last_token        
        # E-W
        total = 0
        for k in range(1, 4 + 1):
            if j + k >= N or board[i, j + k] != last_token:
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
            if i - k < 0 or j + k >= N or board[i - k, j + k] != last_token:
                break
            total += 1
        for k in range(1, 4 + 1):
            if i + k >= M or j - k < 0 or board[i + k, j - k] != last_token:
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
            if i + k >= M or j + k >= N or board[i + k, j + k] != last_token:
                break
            total += 1            
        if total >= 3:
            return last_token        
        return 0
    
    def expand(self):
        if len(self.children) == 0 and self.get_outcome() is None:
            for j in range(self.N):
                child = C4(self)
                if child.move(j):
                    self.children[j] = child
                        
    def expand_one_random_child(self):
        j_indexes = np.where(self.column_fills < C4.M)[0]
        j = np.random.choice(j_indexes) 
        child = C4(self)
        child.move(j)
        self.children[j] = child
        return child
    
    def get_board(self):
        return self.board
    
    def get_extra_info(self):
        return self.column_fills    
    
    def move_name_to_index(self, name):
        return int(name)

    def move_index_to_name(self, move_index):
        return str(move_index)
    
    @staticmethod
    def get_board_shape():
        return (C4.M, C4.N)

    @staticmethod
    def get_extra_info_memory():
        return C4.N

    @staticmethod
    def get_max_actions():
        return C4.N    