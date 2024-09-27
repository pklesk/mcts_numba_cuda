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

    @staticmethod
    def class_repr():
        return f"{C4.__name__}_{C4.M}x{C4.N}"
            
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
    
    def take_action_job(self, action_index):
        j = action_index 
        if self.column_fills[j] == C4.M:
            return False
        i = C4.M - 1 - self.column_fills[j] 
        self.board[i, j] = self.turn
        self.column_fills[j] += 1
        self.turn *= -1
        return True
    
    def compute_outcome_job(self):
        j = self.last_action_index
        i = C4.M - self.column_fills[j]     
        if True: # a bit faster outcome via numba
            numba_outcome = C4.compute_outcome_job_numba_jit(C4.M, C4.N, self.turn, i, j, self.board)
            if numba_outcome != 0:
                return numba_outcome 
        else: # a bit slower outcome via pure Python (inactive now)
            last_token = -self.turn        
            # N-S
            total = 0
            for k in range(1, 4):
                if i -  k < 0 or self.board[i - k, j] != last_token:
                    break
                total += 1
            for k in range(1, 4):
                if i + k >= C4.M or self.board[i + k, j] != last_token:
                    break            
                total += 1
            if total >= 3:            
                return last_token            
            # E-W
            total = 0
            for k in range(1, 4):
                if j + k >= C4.N or self.board[i, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 4):
                if j - k < 0 or self.board[i, j - k] != last_token:
                    break            
                total += 1
            if total >= 3:
                return last_token            
            # NE-SW
            total = 0
            for k in range(1, 4):
                if i - k < 0 or j + k >= C4.N or self.board[i - k, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 4):
                if i + k >= C4.M or j - k < 0 or self.board[i + k, j - k] != last_token:
                    break
                total += 1            
            if total >= 3:
                return last_token            
            # NW-SE
            total = 0
            for k in range(1, 4):
                if i - k < 0 or j - k < 0 or self.board[i - k, j - k] != last_token:
                    break
                total += 1
            for k in range(1, 4):
                if i + k >= C4.M or j + k >= C4.N or self.board[i + k, j + k] != last_token:
                    break
                total += 1            
            if total >= 3:
                return last_token                                    
        if np.sum(self.board == 0) == 0: # draw
            return 0
        return None    
    
    @staticmethod
    @jit(int8(int8, int8, int8, int8, int8, int8[:, :]), nopython=True, cache=True)  
    def compute_outcome_job_numba_jit(M, N, turn, last_i, last_j, board):
        last_token = -turn        
        i, j = last_i, last_j
        # N-S
        total = 0
        for k in range(1, 4):
            if i - k < 0 or board[i - k, j] != last_token:
                break
            total += 1
        for k in range(1, 4):
            if i + k >= M or board[i + k, j] != last_token:
                break            
            total += 1
        if total >= 3:
            return last_token        
        # E-W
        total = 0
        for k in range(1, 4):
            if j + k >= N or board[i, j + k] != last_token:
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
            if i - k < 0 or j + k >= N or board[i - k, j + k] != last_token:
                break
            total += 1
        for k in range(1, 4):
            if i + k >= M or j - k < 0 or board[i + k, j - k] != last_token:
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
            if i + k >= M or j + k >= N or board[i + k, j + k] != last_token:
                break
            total += 1            
        if total >= 3:
            return last_token        
        return 0
    
    # TODO remove if unnecessary (default implementation of expand was moved to State class)
    # def expand(self):
    #     if len(self.children) == 0 and self.compute_outcome() is None:
    #         for j in range(self.N):
    #             self.take_action(j)
                        
    def take_random_action_playout(self):
        j_indexes = np.where(self.column_fills < C4.M)[0]
        j = np.random.choice(j_indexes) 
        child = self.take_action(j)
        return child
    
    def get_board(self):
        return self.board
    
    def get_extra_info(self):
        return self.column_fills    
    
    @staticmethod    
    def action_name_to_index(action_name):
        return int(action_name)

    @staticmethod
    def action_index_to_name(action_index):
        return str(action_index)
    
    @staticmethod
    def get_board_shape():
        return (C4.M, C4.N)

    @staticmethod
    def get_extra_info_memory():
        return C4.N

    @staticmethod
    def get_max_actions():
        return C4.N