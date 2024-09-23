import numpy as np
import time
from utils import dict_to_str

class State:
    
    def __init__(self, parent=None):
        self.win_flag = False
        self.n = 0
        self.n_wins = 0
        self.parent = parent
        self.children = {}
        self.outcome_computed = False # has outcome value been already prepared within last call of get_outcome  
        self.outcome = None # None - ongoing, or {-1, 0, 1} - win for min player, draw, win for max player        
        self.turn = 1 if self.parent is None else self.parent.turn
        self.last_action_index = None

    @staticmethod        
    def class_repr():
        return f"{State.__name__}()"
            
    def _subtree_size(self):
        size = 1
        for key in self.children:
            size += self.children[key]._subtree_size()
        return size
    
    def _subtree_max_depth(self):
        d = 0
        for key in self.children:
            temp_d = self.children[key]._subtree_max_depth()
            if 1 + temp_d > d:
                d = 1 + temp_d 
        return d
    
    def _subtree_depths(self, d=0, depths=[]):
        depths.append(d)
        for key in self.children:
            self.children[key]._subtree_depths(d + 1, depths)
        return depths
    
    def get_turn(self):
        return self.turn            
        
    def take_action(self, action_index):        
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
        pass            
    
    def compute_outcome(self):
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
        pass
                
    def get_board(self):
        pass    

    def get_extra_info(self):
        pass
            
    def expand(self):
        pass
    
    def expand_one_random_child(self):
        pass  
    
    @staticmethod
    def action_name_to_index(action_name):
        pass

    @staticmethod
    def action_index_to_name(action_index):
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
    
                                 
class MCTS:
    
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
        self.search_time_limit = search_time_limit
        self.search_steps_limit = search_steps_limit
        self.vanilla = vanilla # if True, statistics from previous runs (searches) are not reused         
        self.ucb_c = ucb_c                 
        self.seed = seed
        np.random.seed(self.seed)
        self.verbose_debug = verbose_debug
        self.verbose_info = verbose_info

    def __str__(self):         
        return f"{self.__class__.__name__}(search_time_limit={self.search_time_limit}, search_steps_limit={self.search_steps_limit}, vanilla={self.vanilla}, ucb_c={self.ucb_c}, seed: {self.seed})"
        
    def __repr__(self):
        return self.__str__() 
                
    def _make_actions_info(self, children, best_action_entry=False):
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

    def _make_performance_info(self):
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
    
    def _best_action_ucb(self, children, actions_info): 
        best_key = None
        best_ucb = -1.0
        for key in children.keys():
            ucb = actions_info[key]["ucb"]
            if ucb > best_ucb:
                best_ucb = ucb
                best_key = key                        
        return best_key    
    
    def _best_action(self, root_children, root_actions_info): 
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
        while len(state.children) > 0:
            actions_info = self._make_actions_info(state.children)
            best_ucb_action = self._best_action_ucb(state.children, actions_info)
            state = state.children[best_ucb_action]
        return state     
    
    def _expand(self, state):
        state.expand()
        if len(state.children) > 0:
            random_child_key = np.random.choice(list(state.children.keys()))
            state = state.children[random_child_key]
        return state
    
    def _playout(self, state):
        while True:
            outcome = state.compute_outcome()
            if outcome is not None:
                break        
            state = state.expand_one_random_child()
        return state        
    
    def _backup(self, state, playout_root):
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
        self.root_actions_info = self._make_actions_info(self.root.children, best_action_entry=True)
        self._best_action(self.root.children, self.root_actions_info)