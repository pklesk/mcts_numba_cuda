import numpy as np
from mcts import State
from numba import jit
from numba import int8

__version__ = "1.0.1"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl" 

class C4(State):
    """
    Class for states of Connect 4 game.
    
    Attributes:
        M (int): 
            number of rows in the board, defaults to ``6``.            
        N (int): 
            number of columns in the board, defaults to ``7``.
        SYMBOLS (List):
            list of strings representing disc symbols (black, white) or ``"."`` for empty cell. 
    """        
    M = 6 
    N = 7 
    SYMBOLS = ["\u25CB", ".", "\u25CF"] # or: ["O", ".", "X"]    
    
    def __init__(self, parent=None):
        """
        Constructor (ordinary or copying) of ``C4`` instances - states of Connect 4 game.
         
        Args:
            parent (State): 
                reference to parent state object.            
        """
        super().__init__(parent)
        if self.parent:
            self.board = np.copy(self.parent.board)
            self.column_fills = np.copy(self.parent.column_fills)
        else:
            self.board = np.zeros((C4.M, C4.N), dtype=np.int8)
            self.column_fills = np.zeros(C4.N, dtype=np.int8)

    @staticmethod
    def class_repr():
        """        
        Returns a string representation of class ``C4`` (meant to instantiate states of Connect 4 game), informing about the size of board.
        
        Returns:
            str: string representation of class ``C4`` (meant to instantiate states of Connect 4 game), informing about the size of board. 
        """
        return f"{C4.__name__}_{C4.M}x{C4.N}"
            
    def __str__(self):
        """        
        Returns a string representation of this ``C4`` state - the contents of its game board.
        
        Returns:
            str: string representation of this ``C4`` state - the contents of its game board.
        """        
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
        """
        Drops a disc into column indicated by the action_index and returns ``True`` if the action is legal (column not full yet).
        Otherwise, does no changes and returns ``False``.

        Args:
            action_index (int): 
                index of column where to drop a disc.
        
        Returns:
            action_legal (bool):
                boolean flag indicating if the specified action was legal and performed.
        """
        j = action_index 
        if self.column_fills[j] == C4.M:
            return False
        i = C4.M - 1 - self.column_fills[j] 
        self.board[i, j] = self.turn
        self.column_fills[j] += 1
        self.turn *= -1
        return True
    
    def compute_outcome_job(self):
        """        
        Computes and returns the game outcome for this state in compliance with rules of Connect 4 game: 
        {-1, 1} denoting a win for the minimizing or maximizing player, respectively, if he connected at least 4 his discs; 
        0 denoting a tie, when the board is filled and no line of 4 exists;   
        ``None`` when the game is ongoing.
        
        Returns:
            outcome ({-1, 0, 1} or ``None``)
                game outcome for this state.
        """        
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
        """Called by ``compute_outcome_job`` for faster outcomes."""
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
                        
    def take_random_action_playout(self):
        """        
        Picks a uniformly random action from actions available in this state and returns the result of calling ``take_action`` with the action index as argument.
        
        Returns:
            child (State): 
                result of ``take_action`` call for the random action.          
        """        
        j_indexes = np.where(self.column_fills < C4.M)[0]
        j = np.random.choice(j_indexes) 
        child = self.take_action(j)
        return child
    
    def get_board(self):
        """                
        Returns the board of this state (a two-dimensional array of bytes).
        
        Returns:
            board (ndarray[np.int8, ndim=2]):
                board of this state (a two-dimensional array of bytes).
        """        
        return self.board
    
    def get_extra_info(self):
        """
        Returns additional information associated with this state, as one-dimensional array of bytes,
        informing about fills of columns (how many discs have been dropped in each column). 
        
        Returns:
            extra_info (ndarray[np.int8, ndim=1] or ``None``):
                one-dimensional array with additional information associated with this state - fills of columns.        
        """
        return self.column_fills    
    
    @staticmethod    
    def action_name_to_index(action_name):
        """
        Returns an action's index (numbering from 0) based on its name. E.g., name ``"0"``, denoting a drop into the leftmost column, maps to index ``0``.
        
        Args:
            action_name (str):
                name of an action.
        Returns:
            action_index (int):
                index corresponding to the given name.   
        """        
        return int(action_name)

    @staticmethod
    def action_index_to_name(action_index):
        """        
        Returns an action's name based on its index (numbering from 0). E.g., index ``0`` maps to name ``"0"``, denoting a drop into the leftmost column.
        
        Args:
            action_index (int):
                index of an action.
        Returns:
            action_name (str):
                name corresponding to the given index.          
        """        
        return str(action_index)
    
    @staticmethod
    def get_board_shape():
        """
        Returns a tuple with shape of boards for Connect 4 game.
        
        Returns:
            shape (tuple(int, int)):
                shape of boards related to states of this class.
        """        
        return (C4.M, C4.N)

    @staticmethod
    def get_extra_info_memory():
        """        
        Returns amount of memory (in bytes) needed to memorize additional information associated with Connect 4 states, i.e., the memory for fills of columns.
        That number is equal to the number of columns.
        
        Returns:
            extra_info_memory (int):
                number of bytes required to memorize fills of columns.         
        """        
        return C4.N

    @staticmethod
    def get_max_actions():
        """
        Returns the maximum number of actions (the largest branching factor) equal to the number of columns.
        
        Returns:
            max_actions (int):
                maximum number of actions (the largest branching factor) equal to the number of columns.
        """                
        return C4.N