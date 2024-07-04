import numpy as np
import time
from utils import dict_to_str

class State:
    def __init__(self, parent=None):
        self.n = 0
        self.n_wins = 0
        self.parent = parent
        self.children = {}
        self.outcome = None # None - ongoing, or {-1, 0, 1} - win for min player, draw, win for max player
        self.outcome_computed = False
        self.turn = 1 if self.parent is None else self.parent.turn
    
    def move(self, move_index):            
        pass
    
    @staticmethod
    def move_down_tree_via(state, move_index):
        pass
    
    def get_outcome(self):
        pass
        
    def expand(self):
        pass
    
    def expand_one_random_child(self):
        pass
    
    def get_board(self):
        pass    

    def get_extra_info(self):
        pass
    
    @staticmethod
    def move_name_to_index(name):
        pass

    @staticmethod
    def move_index_to_name(move_index):
        pass
    
    @staticmethod
    def get_board_shape():
        pass

    @staticmethod
    def get_extra_info_memory():
        pass

    @staticmethod
    def get_max_actions():
        pass
            
    def states_total(self):
        t = 1
        for key in self.children:
            t += self.children[key].states_total()
        return t
    
    def depth(self):
        d = 0
        for key in self.children:
            temp_d = self.children[key].depth()
            if 1 + temp_d > d:
                d = 1 + temp_d 
        return d
    
class MCTS:
    
    SEARCH_TIME_LIMIT = 5.0
    SEARCH_STEPS_LIMIT = np.inf
    UCB1_C = 2.0
    SEED = 0 
    
    VERBOSE_DEBUG = True
    
    def __init__(self, search_time_limit=SEARCH_TIME_LIMIT, search_steps_limit=SEARCH_STEPS_LIMIT, ucb1_c=UCB1_C, seed=SEED):
        self.search_time_limit = search_time_limit
        self.search_steps_limit = search_steps_limit
        self.ucb1_c = ucb1_c        
        self.seed = seed
        np.random.seed(self.seed)

    def __str__(self):         
        return f"{self.__class__.__name__}(search_time_limit={self.search_time_limit}, search_steps_limit={self.search_steps_limit}, ucb1_c={self.ucb1_c}, seed: {self.seed})"
        
    def __repr__(self):
        return self.__str__() 
    
    def values(self, children):
        values = {}
        ucb1_max = -np.inf
        key_max = None
        for key in children.keys():
            n = children[key].n
            n_wins = children[key].n_wins
            n_parent = children[key].parent.n
            q = n_wins / n if n > 0 else 0.0 # 2nd case does not affect ucb1
            bound_width = self.ucb1_c * np.sqrt(np.log(n_parent) / n) if n > 0 else np.inf
            ucb1 = q + bound_width  
            entry = {}
            entry["name"] = children[key].__class__.move_index_to_name(key)
            entry["n"] = n
            entry["n_wins"] = n_wins
            entry["n_parent"] = n_parent
            entry["q"] = q
            entry["bound_width"] = bound_width
            entry["ucb1"] = ucb1
            values[key] = entry
            if ucb1 > ucb1_max:
                ucb1_max = ucb1
                key_max = key                  
        return values, key_max
        
    def best_move(self, children, values):
        best_key = None
        best_n = -1
        best_q = -np.inf
        for key in children.keys():
            n = values[key]["n"]
            q = values[key]["q"]
            if (n > best_n) or (n == best_n and q > best_q):
                best_n = n
                best_q = q
                best_key = key                        
        return best_key
                
    def run(self, root):
        print("MCTS RUN...")
        t1 = time.time()
        self.root = root
        self.root.parent = None
        step = 0
        total_time_selection = 0.0
        total_time_expansion = 0.0
        total_time_playout = 0.0
        total_time_backup = 0.0
        if self.VERBOSE_DEBUG:
            print(f"[initial states total: {self.root.states_total()}]")
            print(f"[initial depth: {self.root.depth()}]")
            print(f"[initial root n: {self.root.n}]")
        while True:
            t2 = time.time()
            if step >= self.search_steps_limit or t2 - t1 >= self.search_time_limit:
                break            
            state = self.root
            # MCTS selection
            t1_selection = time.time()
            while len(state.children) > 0:
                _, arg_max = self.values(state.children)
                state = state.children[arg_max]
            t2_selection = time.time()
            total_time_selection += t2_selection - t1_selection
            # MCTS expansion
            t1_expansion = time.time()
            state.expand()
            if len(state.children) > 0:
                random_child_key = np.random.choice(list(state.children.keys()))
                state = state.children[random_child_key]
            t2_expansion = time.time()
            total_time_expansion += t2_expansion - t1_expansion            
            # MCTS playout
            playout_root = state
            t1_playout = time.time()
            while True:
                outcome = state.get_outcome()
                if outcome is not None:
                    break        
                state = state.expand_one_random_child()
            t2_playout = time.time()            
            total_time_playout += t2_playout - t1_playout                            
            # MCTS backup
            t1_backup = time.time()
            outcome = state.get_outcome()
            state = playout_root
            del state.children # getting rid of playout
            state.children = {}
            while state:
                state.n += 1
                if state.turn == -outcome:
                    state.n_wins += 1
                state = state.parent               
            t2_backup = time.time()
            total_time_backup += t2_backup - t1_backup                                
            step += 1  
        vs, _ = self.values(self.root.children)
        best_move = self.best_move(self.root.children, vs)        
        print(f"[action values:\n{dict_to_str(vs)}]")        
        if self.VERBOSE_DEBUG:
            print(f"[steps (=playouts) performed: {step}; mean times -> selection: {total_time_selection / step} s,  expansion: {total_time_expansion / step}, playout: {total_time_playout / step} s, backup: {total_time_backup / step}]")            
            print(f"[states total: {self.root.states_total()}]")
            print(f"[depth: {self.root.depth()}]")
            print(f"[root n: {self.root.n}]") 
        print(f"[best move: {best_move}, q: {vs[best_move]['q']}]")                           
        print(f"MCTS RUN DONE. [time: {t2 - t1} s]")                        
        return best_move