import numpy as np
from mcts import State
from numba import jit
from numba import int8

class Gomoku(State):
    M = 15
    N = 15
    SYMBOLS = ['\u25CB', '+', '\u25CF'] # or: [['O', '+', 'X']
    
    def __init__(self, parent=None):
        super().__init__(parent)
        if self.parent:
            self.board = np.copy(self.parent.board)
        else:
            self.board = np.zeros((Gomoku.M, Gomoku.N), dtype=np.int8)
    
    @staticmethod
    def class_repr():
        """        
        Returns a string representation of class ``Gomoku`` (meant to instantiate states of Gomoku game), informing about the size of board.
        
        Returns:
            str: string representation of class ``Gomoku`` (meant to instantiate states of Gomoku game), informing about the size of board. 
        """        
        return f"{Gomoku.__name__}_{Gomoku.M}x{Gomoku.N}"    
            
    def __str__(self):
        s = "  "
        for j in range(Gomoku.N):
            s += f"{chr(j + ord('A'))}"
        s += "\n"
        for i in range(Gomoku.M, 0, -1):
            s += str(i).rjust(2)
            for j in range(Gomoku.N):
                s += Gomoku.SYMBOLS[self.board[i - 1, j] + 1]
            s += str(i).ljust(2)
            s += "\n"
        s += "  "
        for j in range(Gomoku.N):
            s += f"{chr(j + ord('A'))}"
        return s         
    
    def take_action_job(self, action_index):
        i = action_index // Gomoku.N
        j = action_index % Gomoku.N
        if i < 0 or i >= Gomoku.M or j < 0 or j >= Gomoku.N:
            return False
        if self.board[i, j] != 0:
            return False
        self.board[i, j] = self.turn
        self.turn *= -1
        return True
    
    def compute_outcome_job(self):
        i = self.last_action_index // Gomoku.N
        j = self.last_action_index % Gomoku.N
        if True: # a bit faster outcome via numba
            numba_outcome = Gomoku.compute_outcome_job_numba_jit(Gomoku.M, Gomoku.N, self.turn, i, j, self.board)
            if numba_outcome != 0:
                return numba_outcome 
        else:
            last_token = -self.turn        
            # N-S
            total = 0
            for k in range(1, 6):
                if i -  k < 0 or self.board[i - k, j] != last_token:
                    break
                total += 1
            for k in range(1, 6):
                if i + k >= Gomoku.M or self.board[i + k, j] != last_token:
                    break            
                total += 1
            if total == 4:
                return last_token                        
            # E-W
            total = 0
            for k in range(1, 6):
                if j + k >= Gomoku.N or self.board[i, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 6):
                if j - k < 0 or self.board[i, j - k] != last_token:
                    break            
                total += 1
            if total == 4:
                return last_token            
            # NE-SW
            total = 0
            for k in range(1, 6):
                if i - k < 0 or j + k >= Gomoku.N or self.board[i - k, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 6):
                if i + k >= Gomoku.M or j - k < 0 or self.board[i + k, j - k] != last_token:
                    break
                total += 1            
            if total == 4:
                return last_token            
            # NW-SE
            total = 0
            for k in range(1, 6):
                if i - k < 0 or j - k < 0 or self.board[i - k, j - k] != last_token:
                    break
                total += 1
            for k in range(1, 6):
                if i + k >= Gomoku.M or j + k >= Gomoku.N or self.board[i + k, j + k] != last_token:
                    break
                total += 1            
            if total == 4:
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
        for k in range(1, 6):
            if i - k < 0 or board[i - k, j] != last_token:
                break
            total += 1
        for k in range(1, 6):
            if i + k >= M or board[i + k, j] != last_token:
                break            
            total += 1
        if total == 4:
            return last_token        
        # E-W
        total = 0
        for k in range(1, 6):
            if j + k >= N or board[i, j + k] != last_token:
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
            if i - k < 0 or j + k >= N or board[i - k, j + k] != last_token:
                break
            total += 1
        for k in range(1, 6):
            if i + k >= M or j - k < 0 or board[i + k, j - k] != last_token:
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
            if i + k >= M or j + k >= N or board[i + k, j + k] != last_token:
                break
            total += 1            
        if total == 4:
            return last_token        
        return 0        
                            
    def take_random_action_playout(self):
        indexes = np.where(np.ravel(self.board) == 0)[0]
        action_index = np.random.choice(indexes) 
        child = self.take_action(action_index)
        return child    
    
    def get_board(self):
        return self.board
    
    def get_extra_info(self):
        return None
   
    @staticmethod
    def action_name_to_index(action_name):
        letter = action_name.upper()[0]
        j = ord(letter) - ord('A')
        i = int(action_name[1:]) - 1
        return i * Gomoku.N + j

    @staticmethod
    def action_index_to_name(action_index):
        i = action_index // Gomoku.N
        j = action_index % Gomoku.N
        return f"{chr(ord('A') + j)}{i + 1}"
   
    @staticmethod
    def get_board_shape():
        return (Gomoku.M, Gomoku.N)

    @staticmethod
    def get_extra_info_memory():
        return 0

    @staticmethod
    def get_max_actions():
        return Gomoku.M * Gomoku.N