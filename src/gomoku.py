import numpy as np
from mcts import State
from numba import jit
from numba import int8

class Gomoku(State):
    M = 9
    N = 9
    SYMBOLS = ['\u25CB', '+', '\u25CF'] # or: [['O', '+', 'X']
    
    def __init__(self, parent=None):
        super().__init__(parent)
        if self.parent:
            self.board = np.copy(self.parent.board)
        else:
            self.board = np.zeros((Gomoku.M, Gomoku.N), dtype=np.int8)
        self.last_move = None
            
    def __str__(self):
        s = ""
        for i in range(Gomoku.M, 0, -1):
            s += str(i).rjust(2)
            for j in range(Gomoku.N):
                s += Gomoku.SYMBOLS[self.board[i - 1, j] + 1]
            s += "\n"
        s += "  "
        for j in range(Gomoku.N):
            s += f"{chr(j + ord('A'))}"
        return s
    
    def __repr__(self):
        s = str(self)
        s += f"\n[n: {self.n}, n_wins: {self.n_wins}]"
        s += f"\n[turn: {self.turn}, outcome: {self.outcome}, n_children: {len(self.children)}]"
        return s    
    
    @staticmethod
    def move_name_to_index(name):
        letter = name.upper()[0]
        j = ord(letter) - ord('A')
        i = int(name[1:]) - 1
        return i * Gomoku.N + j

    @staticmethod
    def move_index_to_name(index):
        i = index // Gomoku.N
        j = index % Gomoku.N
        return f"{chr(ord('A') + j)}{i + 1}" 
    
    def move(self, move_index):
        i = move_index // Gomoku.N
        j = move_index % Gomoku.N
        if i < 0 or i >= Gomoku.M or j < 0 or j >= Gomoku.N:
            return False
        if self.board[i, j] != 0:
            return False
        self.board[i, j] = self.turn
        self.turn *= -1
        self.last_move = (i, j)
        return True
    
    @staticmethod
    def move_down_tree_via(state, move_index):
        if len(state.children) > 0:
            return state.children[move_index]
        child = Gomoku(state)
        move_valid = child.move(move_index) 
        if not move_valid:
            return None
        return child
    
    def expand(self):
        if len(self.children) == 0 and self.get_outcome() is None:
            m_n = self.M * self.N
            for index in range(m_n):
                child = Gomoku(self)
                if child.move(index):
                    self.children[index] = child
                        
    def expand_one_random_child(self):
        indexes = np.where(np.ravel(self.board) == 0)[0]
        move_index = np.random.choice(indexes) 
        child = Gomoku(self)
        child.move(move_index)
        self.children[move_index] = child
        return child    
    
    def get_outcome(self):
        if self.outcome_computed:
            return self.outcome
        if not self.last_move:
            return None
        self.outcome_computed = True # will be soon        
        if True: # a bit faster outcome via numba
            numba_outcome = Gomoku.get_outcome_numba_jit(Gomoku.M, Gomoku.N, self.turn, self.last_move[0], self.last_move[1], self.board)
            if numba_outcome != 0:
                self.outcome = numba_outcome
                return self.outcome 
        else:
            last_token = -self.turn        
            i, j = self.last_move            
            # N-S
            total = 0
            for k in range(1, 6 + 1):
                if i -  k < 0 or self.board[i - k, j] != last_token:
                    break
                total += 1
            for k in range(1, 6 + 1):
                if i + k >= Gomoku.M or self.board[i + k, j] != last_token:
                    break            
                total += 1
            if total == 4:
                self.outcome = last_token            
                return last_token            
            # E-W
            total = 0
            for k in range(1, 6 + 1):
                if j + k >= Gomoku.N or self.board[i, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 6 + 1):
                if j - k < 0 or self.board[i, j - k] != last_token:
                    break            
                total += 1
            if total == 4:
                self.outcome = last_token
                return last_token            
            # NE-SW
            total = 0
            for k in range(1, 6 + 1):
                if i - k < 0 or j + k >= Gomoku.N or self.board[i - k, j + k] != last_token:
                    break
                total += 1
            for k in range(1, 6 + 1):
                if i + k >= Gomoku.M or j - k < 0 or self.board[i + k, j - k] != last_token:
                    break
                total += 1            
            if total == 4:
                self.outcome = last_token
                return last_token            
            # NW-SE
            total = 0
            for k in range(1, 6 + 1):
                if i - k < 0 or j - k < 0 or self.board[i - k, j - k] != last_token:
                    break
                total += 1
            for k in range(1, 6 + 1):
                if i + k >= Gomoku.M or j + k >= Gomoku.N or self.board[i + k, j + k] != last_token:
                    break
                total += 1            
            if total == 4:
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
        for k in range(1, 6 + 1):
            if i - k < 0 or board[i - k, j] != last_token:
                break
            total += 1
        for k in range(1, 6 + 1):
            if i + k >= M or board[i + k, j] != last_token:
                break            
            total += 1
        if total == 4:
            return last_token        
        # E-W
        total = 0
        for k in range(1, 6 + 1):
            if j + k >= N or board[i, j + k] != last_token:
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
            if i - k < 0 or j + k >= N or board[i - k, j + k] != last_token:
                break
            total += 1
        for k in range(1, 6 + 1):
            if i + k >= M or j - k < 0 or board[i + k, j - k] != last_token:
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
            if i + k >= M or j + k >= N or board[i + k, j + k] != last_token:
                break
            total += 1            
        if total == 4:
            return last_token        
        return 0    
   
    @staticmethod
    def get_board_shape():
        return (Gomoku.M, Gomoku.N)

    @staticmethod
    def get_extra_info_memory():
        return 0

    @staticmethod
    def get_max_actions():
        return Gomoku.M * Gomoku.N  
    
    def get_board(self):
        return self.board
    
    def get_extra_info(self):
        return None    