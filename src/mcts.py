"""
Auxiliary module with a referential standard implementation of MCTS algorithm (for CPU, single-threaded).
The module contains:

- ``State``: class representing an arbitrary state of some game or sequential decision problem (meant to inherit from when searches using ``MCTS`` class are planned);
    current examples of subclasses are: `C4`` in :doc:`c4` (representation of Connect 4 game), ``Gomoku`` in :doc:`gomoku` (representation of Gomoku game).  

- ``MCTS``: class representing the referential MCTS algorithm.


Link to project repository
--------------------------
`https://github.com/pklesk/mcts_numba_cuda <https://github.com/pklesk/mcts_numba_cuda>`_

Notes
-----
Private functions of ``State`` and ``MCTS`` classes are named with a single leading underscore.
For public methods full docstrings are provided (with arguments and returns described). For private functions short docstrings are provided.
"""

import numpy as np
import time
from utils import dict_to_str

__version__ = "1.0.1"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl" 

class State:
    """
    Arbitrary abstract state of some game or sequential decision problem. Meant to be inherited - extended to subclasses 
    and then applicable in searches conducted by ``MCTS`` class.
    For actual inheritance examples see:
    
    -  ``C4`` class in :doc:`c4` (representation of Connect 4 game),
    
    -  ``Gomoku`` class in :doc:`gomoku` (representation of Gomoku game).
    
    When searches using ``MCTS`` class are planned, the programmer, while inheriting from ``State``, must provide implementations for the following non-static methods: 
    ``take_action_job``, ``compute_outcome_job``, ``take_random_action_playout``, ``__str__``; 
    and one static method ``class_repr``.
    When searches using ``MCTSNC`` class are planned, the programmer, while inheriting from ``State``, must provide the following non-static methods:    
    ``get_board``, ``get_extra_info``
    and the following static ones:
    ``get_board_shape``, ``get_extra_info_memory``, ``get_max_actions``.
    """        
            
    def __init__(self, parent=None):
        """
        Constructor of ``State`` instances. Should be called (in the first line) in subclasses' constructors as: ``super().__init__(parent)``.
         
        Args:
            parent (State): 
                reference to parent state object.            
        """
        self.win_flag = False
        self.n = 0
        self.n_wins = 0
        self.parent = parent
        self.children = {}
        self.outcome_computed = False # has outcome value been already prepared within last call of get_outcome  
        self.outcome = None # None - ongoing, or {-1, 0, 1} - win for min player, draw, win for max player        
        self.turn = 1 if self.parent is None else self.parent.turn
        self.last_action_index = None        

    def __str__(self):
        """
        [To be implemented in subclasses.]
        
        Should return a string representation of this state, e.g., game board with its current contents.
        
        Returns:
            str: string representation of this state.
        """
        pass

    @staticmethod        
    def class_repr():
        """
        [To be implemented in subclasses.]
        
        Should return a string representation of this class of states (e.g., game name, its variation, board size if configurable, etc.).
        
        Returns:
            str: string representation of this class of states. 
        """
        pass
            
    def _subtree_size(self):
        """Returns size of the subtree rooted by this state (number of tree nodes including this one)."""
        size = 1
        for key in self.children:
            size += self.children[key]._subtree_size()
        return size
    
    def _subtree_max_depth(self):
        """Returns (maximum) depth of the subtree rooted by this state."""
        d = 0
        for key in self.children:
            temp_d = self.children[key]._subtree_max_depth()
            if 1 + temp_d > d:
                d = 1 + temp_d 
        return d
    
    def _subtree_depths(self, d=0, depths=[]):
        """Returns a list of depths for nodes in the subtree rooted by this state."""
        depths.append(d)
        for key in self.children:
            self.children[key]._subtree_depths(d + 1, depths)
        return depths
    
    def get_turn(self):
        """
        Returns {-1, 1} indicating whose turn it is: -1 for the minimizing player, 1 for the maximizing player.
        
        Returns:
            self.turn ({-1, 1}): 
                indicating whose turn it is: -1 for the minimizing player, 1 for the maximizing player.
        """
        return self.turn            
        
    def take_action(self, action_index):
        """
        Takes action (specified by its index) and returns the child-state implied by the action. 
        If such a child already existed (prior to the call) among children, returns it immediately. 
        Otherwise, creates a new child-state object and tries to call on it the function ``take_action_job``, assumed as implemented in the subclass.
        If ``None`` is returned in the latter case (interpreted as illegal action), then forwards ``None`` as the result .
        
        Args:
            action_index (int): 
                index of action to be taken.
            
        Returns:
            child (State): 
                reference to child state implied by the action or ``None`` if action illegal.
        """
        if action_index in self.children:            
            return self.children[action_index]
        child = type(self)(self) # copying constructor
        action_legal = child.take_action_job(action_index) 
        if not action_legal:
            return None # no effect takes place
        child.last_action_index = action_index
        self.children[action_index] = child
        return child
    
    def take_action_job(self, action_index):
        """
        [To be implemented in subclasses.]
        
        Should performs changes on this state implied by the given action, and return ``True`` if the action is legal.
        Otherwise, should do no changes and return ``False``.

        Args:
            action_index (int): 
                index of action to be taken.
        
        Returns:
            action_legal (bool):
                boolean flag indicating if the specified action was legal and performed.
        """
        pass            
    
    def compute_outcome(self):
        """
        If called before on this state, returns the outcome already computed and memorized.
        Otherwise, tries to call on this state the function ``compute_outcome_job`` (assumed as implemented in the subclass) and return its result.
        Possible outcomes for terminal states are {-1, 0, 1}, indicating respectively: 
        a win for the minimizing player, a draw, a win for the maximizing player.
        For an ongoing game the outcome should be ``None``. 
        
        Returns:
            outcome ({-1, 0, 1} or ``None``):
                outcome of the game represented by this state.
        """        
        if self.outcome_computed:
            return self.outcome
        if self.last_action_index is None:
            return None
        self.outcome = self.compute_outcome_job()
        self.outcome_computed = True
        if self.outcome == -self.turn:
            self.win_flag = True
        return self.outcome

    def compute_outcome_job(self):
        """
        [To be implemented in subclasses.]
        
        Should compute and return the game outcome for this state in compliance with rules of the game it represents.
        Possible results for terminal states are {-1, 0, 1}, indicating respectively: 
        a win for the minimizing player, a draw, a win for the maximizing player.
        For an ongoing game the result should be ``None``.
        
        Returns:
            outcome ({-1, 0, 1} or ``None``)
                game outcome for this state.
        """        
        pass
                
    def get_board(self):
        """
        [To be implemented in subclasses only when a search using ``MCTSNC`` is planned. Not required for ``MCTS`` searches.]
        
        Should return the representation of this state as a two-dimensional array of bytes - in a board-like form (e.g., chessboard, backgammon board, etc.),
        even if no board naturally exists in the related game (e.g., bridge, Nim, etc.).
        
        Returns:
            board (ndarray[np.int8, ndim=2]):
                two-dimensional array with representation of this state.
        """
        pass    

    def get_extra_info(self):
        """
        [To be implemented in subclasses only when a search using ``MCTSNC`` is planned. Not required for ``MCTS`` searches.]
        
        Should return additional information associated with this state (as a one-dimensional array of bytes)
        not implied by the contents of the board itself (e.g., possibilities of castling or en-passant 
        captures in chess, the contract in double dummy bridge, etc.), or any technical information useful to generate 
        legal actions faster. If no additional information is needed should return ``None``.
        
        Returns:
            extra_info (ndarray[np.int8, ndim=1] or ``None``):
                one-dimensional array with any additional information associated with this state.        
        """        
        return None
            
    def expand(self):
        """        
        Expands this state to generate its children by calling ``take_action`` multiple times for all possible action indexes. 
        """
        if len(self.children) == 0 and self.compute_outcome() is None:
            for action_index in range(self.__class__.get_max_actions()):
                self.take_action(action_index)
    
    def take_random_action_playout(self):
        """
        [To be implemented in subclasses.]
        
        Should pick a uniformly random action from actions available in this state and return the result of calling ``take_action`` with the action index as argument.
        
        Returns:
            child (State): 
                result of ``take_action`` call for the random action.          
        """
        pass  
    
    @staticmethod
    def action_name_to_index(action_name):
        """
        [To be optionally implemented by programmer in subclasses.]
        
        Returns an action's index (numbering from 0) based on its name. E.g., name ``"B4"`` for 15 x 15 Gomoku maps to index ``18``.
        
        Args:
            action_name (str):
                name of an action.
        Returns:
            action_index (int):
                index corresponding to the given name.   
        """
        pass

    @staticmethod
    def action_index_to_name(action_index):
        """
        [To be optionally implemented by programmer in subclasses.]
        
        Returns an action's name based on its index (numbering from 0). E.g., index ``18`` for 15 x 15 Gomoku maps to name ``"B4"``.
        
        Args:
            action_index (int):
                index of an action.
        Returns:
            action_name (str):
                name corresponding to the given index.          
        """       
        pass
    
    @staticmethod
    def get_board_shape():
        """
        [To be implemented in subclasses only when a search using ``MCTSNC`` is planned. Not required for ``MCTS`` searches.]
        
        Returns a tuple with shape of boards for the game (or sequential decision problem) represented by this class.
        
        Returns:
            shape (tuple(int, int)):
                shape of boards related to states of this class.
        """
        pass

    @staticmethod
    def get_extra_info_memory():
        """
        [To be implemented in subclasses only when a search using ``MCTSNC`` is planned. Not required for ``MCTS`` searches.]
        
        Returns amount of memory (in bytes) needed to memorize additional information associated with states of the game (or sequential decision problem) represented by this class.
        
        Returns:
            extra_info_memory (int):
                number of bytes required to memorize additional information relate to states of this class.         
        """
        pass

    @staticmethod
    def get_max_actions():
        """
        [To be implemented in subclasses.]
        
        Returns the maximum number of actions (the largest branching factor) possible in the game represented by this class.
        
        Returns:
            max_actions (int):
                maximum number of actions (the largest branching factor) possible in the game represented by this class.
        """        
        pass
    
                                 
class MCTS:
    """
    Monte Carlo Tree Search - standard, referential implementation (for CPU, single-threaded).
    """
    
    DEFAULT_SEARCH_TIME_LIMIT = 5.0 # [s], np.inf possible
    DEFAULT_SEARCH_STEPS_LIMIT = np.inf # integer, np.inf possible
    DEFAULT_VANILLA = True
    DEFAULT_UCB_C = 2.0
    DEFAULT_SEED = 0
    DEFAULT_VERBOSE_DEBUG = False
    DEFAULT_VERBOSE_INFO = True
    
    def __init__(self, 
                 search_time_limit=DEFAULT_SEARCH_TIME_LIMIT, search_steps_limit=DEFAULT_SEARCH_STEPS_LIMIT,
                 vanilla=DEFAULT_VANILLA,                  
                 ucb_c=DEFAULT_UCB_C, seed=DEFAULT_SEED,
                 verbose_debug=DEFAULT_VERBOSE_DEBUG, verbose_info=DEFAULT_VERBOSE_INFO):
        """
        Constructor of ``MCTS`` instances.
         
        Args:
            search_time_limit (float):
                time limit in seconds (computational budget), ``np.inf`` if no limit, defaults to ``5.0``.             
            search_steps_limit (float): 
                steps limit (computational budget), ``np.inf`` if no limit, defaults to ``np.inf``.
            vanilla (bool):
                flag indicating whether information (partial tree, action-value estimates, etc.) from previous searches is ignored, defaults to ``True``.
            verbose_debug (bool):
                debug verbosity flag, if ``True`` then detailed information about each kernel invocation are printed to console (in each iteration), defaults to ``False``.
            verbose_info (bool): 
                verbosity flag, if ``True`` then standard information on actions and performance are printed to console (after a full run), defaults to ``True``.            
        """        
        self.search_time_limit = search_time_limit
        self.search_steps_limit = search_steps_limit
        self.vanilla = vanilla # if True, statistics from previous runs (searches) are not reused         
        self.ucb_c = ucb_c                 
        self.seed = seed
        np.random.seed(self.seed)
        self.verbose_debug = verbose_debug
        self.verbose_info = verbose_info

    def __str__(self):         
        """
        Returns a string representation of this ``MCTS`` instance.
        
        Returns:
            str: string representation of this ``MCTS`` instance.
        """           
        return f"MCTS(search_time_limit={self.search_time_limit}, search_steps_limit={self.search_steps_limit}, vanilla={self.vanilla}, ucb_c={self.ucb_c}, seed: {self.seed})"
        
    def __repr__(self):
        """
        Returns a string representation of this ``MCTS`` instance (equivalent to ``__str__`` method).
        
        Returns:
            str: string representation of this ``MCTSNC`` instance.
        """                
        return self.__str__() 

    def _make_performance_info(self):
        """
        Prepares and returns a dictionary with information on performance during the last run. 
        After the call, available via ``performance_info`` attribute.
        """        
        performance_info = {}
        performance_info["steps"] = self.steps
        performance_info["steps_per_second"] = self.steps / self.time_total                
        performance_info["playouts"] = self.root.n
        performance_info["playouts_per_second"] = performance_info["playouts"] / self.time_total           
        ms_factor = 10.0**3
        times_info = {}
        times_info["total"] = ms_factor * self.time_total
        times_info["loop"] = ms_factor * self.time_loop
        times_info["reduce_over_actions"] = ms_factor * self.time_reduce_over_actions
        times_info["mean_loop"] = times_info["loop"] / self.steps
        times_info["mean_select"] = ms_factor * self.time_select / self.steps
        times_info["mean_expand"] = ms_factor * self.time_expand / self.steps
        times_info["mean_playout"] = ms_factor * self.time_playout / self.steps
        times_info["mean_backup"] = ms_factor * self.time_backup / self.steps
        performance_info["times_[ms]"] = times_info
        tree_info = {}
        tree_info["initial_n_root"] = self.initial_n_root
        tree_info["initial_mean_depth"] = self.initial_mean_depth        
        tree_info["initial_max_depth"] = self.initial_max_depth
        tree_info["initial_size"] = self.initial_size            
        tree_info["n_root"] = self.root.n
        tree_info["mean_depth"] = np.mean(self.root._subtree_depths(0, []))
        tree_info["max_depth"] = self.root._subtree_max_depth()
        tree_info["size"] = self.root._subtree_size()              
        performance_info["tree"] = tree_info
        self.performance_info = performance_info
        return performance_info

                
    def _make_actions_info(self, children, best_action_entry=False):
        """
        Prepares and returns a dictionary with information on root actions implied by the last run, in particular: estimates of action values, their UCBs, counts of times actions were taken, etc.
        After the call, available via ``actions_info`` attribute.
        """
        actions_info = {}
        for key in children.keys():
            n_root = children[key].parent.n
            win_flag = children[key].win_flag
            n = children[key].n
            n_wins = children[key].n_wins        
            q = n_wins / n if n > 0 else 0.0 # 2nd case does not affect ucb
            ucb = q + self.ucb_c * np.sqrt(np.log(n_root) / n) if n > 0 else np.inf 
            entry = {}
            entry["name"] = children[key].__class__.action_index_to_name(key)
            entry["n_root"] = n_root
            entry["win_flag"] = win_flag
            entry["n"] = n
            entry["n_wins"] = n_wins
            entry["q"] = n_wins / n if n > 0 else np.nan
            entry["ucb"] = ucb
            actions_info[key] = entry
        if best_action_entry:
            best_key = self._best_action(children, actions_info)
            best_entry = {"index": best_key, **actions_info[best_key]}
            actions_info["best"] = best_entry
        self.actions_info = actions_info
        return actions_info
    
    def _best_action_ucb(self, children, actions_info):
        """Returns the best action for selection stage purposes, i.e. the action with the largest UCB value."""  
        best_key = None
        best_ucb = -1.0
        for key in children.keys():
            ucb = actions_info[key]["ucb"]
            if ucb > best_ucb:
                best_ucb = ucb
                best_key = key                        
        return best_key    
    
    def _best_action(self, root_children, root_actions_info):
        """
        Returns the best action among the root actions for the final decision.
        Actions' comparison is a three-step process: 
        (1) in the first order, the win flag is decisive (attribute ``win_flag`` of a child state), 
        (2) if there is a tie (win flags equal), the number of times an action was taken becomes decisive (attribute ``n`` of a child state), 
        (3) if there still is a tie (both win flags and action execution counts equal), the number of wins becomes decisive (attribute ``n_wins`` of a child state).
        """ 
        self.best_action = None
        self.best_win_flag = False
        self.best_n = -1
        self.best_n_wins = -1
        for key in root_children.keys():            
            win_flag = root_actions_info[key]["win_flag"]
            n = root_actions_info[key]["n"]
            n_wins = root_actions_info[key]["n_wins"]
            if (win_flag > self.best_win_flag) or\
             ((win_flag == self.best_win_flag) and (n > self.best_n)) or\
             ((win_flag == self.best_win_flag) and (n == self.best_n) and (n_wins > self.best_n_wins)):
                self.best_win_flag = win_flag
                self.best_n = n
                self.best_n_wins = n_wins
                self.best_action = key
        self.best_q = self.best_n_wins / self.best_n if self.best_n > 0 else np.nan                      
        return self.best_action
        
    def run(self, root, forced_search_steps_limit=np.inf):
        """
        Runs the standard, referential implementation of Monte Carlo Tree Search (on CPU, single-threaded).
        
        Args:
            root (State):
                root state from which the search starts.            
            forced_search_steps_limit (int):
                steps limit used only when reproducing results of a previous experiment; if less than``np.inf`` then has a priority over the standard computational budget given by ``search_time_limit`` and ``search_steps_limit``.            
        Returns:
            self.best_action (int):
                best action resulting from search.                        
        """
        print("MCTS RUN...")
        t1 = time.time()
        self.root = root
        self.root.parent = None
        if self.vanilla:
            self.root.n = 0                       
            self.root.children = {}
        
        if self.verbose_info:
            self.initial_n_root = self.root.n                    
            self.initial_mean_depth = np.mean(self.root._subtree_depths(0, []))
            self.initial_max_depth = self.root._subtree_max_depth()            
            self.initial_size = self.root._subtree_size()                         
            
        self.time_select = 0.0
        self.time_expand = 0.0        
        self.time_playout = 0.0
        self.time_backup = 0.0    
        self.steps = 0
                
        t1_loop = time.time()
        while True:
            t2_loop = time.time()
            if forced_search_steps_limit < np.inf:
                if self.steps >= forced_search_steps_limit:
                    break
            elif self.steps >= self.search_steps_limit or t2_loop - t1_loop >= self.search_time_limit:
                break            
            state = self.root
            
            # selection
            if self.verbose_debug:
                print(f"[MCTS._select()...]")            
            t1_select = time.time()
            state = self._select(state)
            t2_select = time.time()
            if self.verbose_debug:
                print(f"[MCTS._select() done; time: {t2_select - t1_select} s]")            
            self.time_select += t2_select - t1_select
            
            # expansion
            if self.verbose_debug:
                print(f"[MCTS._expand()...]")
            t1_expand = time.time()
            state = self._expand(state)
            t2_expand = time.time()
            if self.verbose_debug:
                print(f"[MCTS._expand() done; time: {t2_expand - t1_expand} s]")            
            self.time_expand += t2_expand - t1_expand            
            
            # playout
            if self.verbose_debug:
                print(f"[MCTS._playout()...]")
            t1_playout = time.time()
            playout_root = state
            state = self._playout(state)
            t2_playout = time.time()
            if self.verbose_debug:
                print(f"[MCTS._playout() done; time: {t2_playout - t1_playout} s]")                        
            self.time_playout += t2_playout - t1_playout                            
            
            # backup
            if self.verbose_debug:
                print(f"[MCTS._backup()...]")           
            t1_backup = time.time()
            self._backup(state, playout_root)
            t2_backup = time.time()
            if self.verbose_debug:
                print(f"[MCTS._backup() done; time: {t2_backup - t1_backup} s]")            
            self.time_backup += t2_backup - t1_backup                                
            
            self.steps += 1  
        self.time_loop = time.time() - t1_loop

        if self.verbose_debug:
            print(f"[MCTS._reduce_over_actions()...]")        
        t1_reduce_over_actions = time.time()        
        self._reduce_over_actions()
        best_action_label = str(self.best_action)
        best_action_label += f" ({type(self.root).action_index_to_name(self.best_action)})"
        t2_reduce_over_actions = time.time()
        if self.verbose_debug:
            print(f"[MCTS._reduce_over_actions() done; time: {t2_reduce_over_actions - t1_reduce_over_actions} s]")        
        self.time_reduce_over_actions = t2_reduce_over_actions - t1_reduce_over_actions     
        
        t2 = time.time()    
        self.time_total = t2 - t1
        
        if self.verbose_info:
            print(f"[actions info:\n{dict_to_str(self.root_actions_info)}]")
            print(f"[performance info:\n{dict_to_str(self._make_performance_info())}]")
                                             
        print(f"MCTS RUN DONE. [time: {self.time_total} s; best action: {best_action_label}, best win_flag: {self.best_win_flag}, best n: {self.best_n}, best n_wins: {self.best_n_wins}, best q: {self.best_q}]")                      
        return self.best_action
    
    def _select(self, state):
        """Performs the selection stage and returns the selected state."""
        while len(state.children) > 0:
            actions_info = self._make_actions_info(state.children)
            best_ucb_action = self._best_action_ucb(state.children, actions_info)
            state = state.children[best_ucb_action]
        return state     
    
    def _expand(self, state):
        """Performs the expansion stage and returns the child (picked on random) on which to carry out the playout."""
        state.expand()
        if len(state.children) > 0:
            random_child_key = np.random.choice(list(state.children.keys()))
            state = state.children[random_child_key]
        return state
    
    def _playout(self, state):
        """Performs the playout stage and returns the reached terminal state."""
        while True:
            outcome = state.compute_outcome()
            if outcome is not None:
                break        
            state = state.take_random_action_playout()
        return state        
    
    def _backup(self, state, playout_root):
        """Calls ``compute_outcome`` method on the terminal state (``state``), and suitably backs up the outcome to ancestors of the playout root."""
        outcome = state.compute_outcome()
        state = playout_root
        del state.children # getting rid of playout branch
        state.children = {}
        while state:
            state.n += 1
            if state.turn == -outcome:
                state.n_wins += 1
            state = state.parent
            
    def _reduce_over_actions(self):
        """Calls ``_make_actions_info`` and ``_best_action`` using children states of the root to finds the best available action."""
        self.root_actions_info = self._make_actions_info(self.root.children, best_action_entry=True)
        self._best_action(self.root.children, self.root_actions_info)