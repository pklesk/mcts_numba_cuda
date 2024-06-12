import numpy as np
from numpy import inf
from numba import cuda
from numba import void, int8, int16, int32, int64, float32, boolean
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_type 
import time
import math
from numba.core.errors import NumbaPerformanceWarning
import warnings
from mcts_cuda_game_specifics import is_action_legal, take_action, legal_actions_playout, take_action_playout, compute_outcome

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

class MCTSCuda:
    
    SEARCH_TIME_LIMIT = 5.0
    SEARCH_STEPS_LIMIT = np.inf
    UCB1_C = 2.0
    SEED = 0
        
    VERBOSE_DEBUG = False
    VERBOSE_INFO = True

    MAX_STATE_BOARD_SHAPE = (32, 32)
    MAX_STATE_EXTRA_INFO_MEMORY = 1024
    MAX_STATE_MAX_ACTIONS = 1024
    MAX_DEVICE_MEMORY = 1.0 * 1024**3 # to be consumed by device-side multiple trees for MCTS (and related information)        
    MAX_TREE_SIZE = 2**24
    MAX_N_TREES = 512
    
    MAX_N_PLAYOUTS = 512
    MAX_TREE_DEPTH = 2048 # to memorize selected paths  
        
    def __init__(self, state_board_shape, state_extra_info_memory, state_max_actions, 
                 n_trees=16, n_playouts=128, kind="scpo", 
                 search_time_limit=SEARCH_TIME_LIMIT, search_steps_limit=SEARCH_STEPS_LIMIT, ucb1_c=UCB1_C, 
                 seed=SEED, max_device_memory=MAX_DEVICE_MEMORY,
                 action_to_name_function=None):
        self.state_board_shape = state_board_shape
        if self.state_board_shape[0] > self.MAX_STATE_BOARD_SHAPE[0] or self.state_board_shape[1] > self.MAX_STATE_BOARD_SHAPE[1]:
            sys.exit(f"[MCTSCuda.__init__() -> exiting due to allowed state board shape exceeded]")            
        self.state_extra_info_memory = max(state_extra_info_memory, 1)
        if self.state_extra_info_memory > self.MAX_STATE_EXTRA_INFO_MEMORY:
            sys.exit(f"[MCTSCuda.__init__() -> exiting due to allowed state extra info memory exceeded]")        
        self.state_max_actions = state_max_actions
        if self.state_max_actions > self.MAX_STATE_MAX_ACTIONS:
            sys.exit(f"[MCTSCuda.__init__() -> exiting due to allowed state max actions memory exceeded]")        
        self.n_trees = min(n_trees, self.MAX_N_TREES)
        self.n_playouts = min(n_playouts, self.MAX_N_PLAYOUTS)
        self.kind = kind
        self._run = None # function pointer assignment postponed        
        self.search_time_limit = search_time_limit
        self.search_steps_limit = search_steps_limit
        self.ucb1_c = ucb1_c
        self.seed = seed
        self.max_device_memory = max_device_memory
        self.action_to_name_function = action_to_name_function         
        self._set_cuda_constants()
        if not self._cuda_available:
            sys.exit(f"[MCTSCuda.__init__() -> exiting due to cuda computations not available]")
        self._search_time_limit_epsilon = 0.01 + 0.5 * (self.n_trees / self.MAX_N_TREES + self.state_max_actions / self.MAX_STATE_MAX_ACTIONS) * 0.04         
    
    def _set_cuda_constants(self):    
        self._cuda_available = cuda.is_available() 
        self._cuda_tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2 if self._cuda_available else None
        
    def _init_device_side_arrays(self):        
        per_state_additional_memory = 1 + 1 + 1 + 1 + 4 + 4 # turns (1), leaves (1), terminals (1), outcomes (1), ns (4), ns_wins (4)
        per_tree_additional_memory = 4 + 4 + (self.state_max_actions + 2) * 4 + 2 * 4 # tree sizes (4), nodes selected (4), nodes expanded ((self.state_max_actions + 2) * 4), playout outcomes (2 * 4)
        per_tree_additional_memory += (MAX_TREE_DEPTH + 2) * 4 
        if self.kind == "acpo":
            per_tree_additional_memory += self.state_max_actions * 2 * 4 # more playout outcomes for the expanded level (self.state_max_actions * 2 * 4)            
        self._per_state_memory = np.prod(self.state_board_shape) + self.state_extra_info_memory + (1 + self.state_max_actions) * 4 + per_state_additional_memory 
        self._max_tree_size = (int(self.max_device_memory) - self.n_trees * per_tree_additional_memory) // (self._per_state_memory * self.n_trees)
        self._max_tree_size = min(self._max_tree_size, self.MAX_TREE_SIZE)         
        if MCTSCuda.VERBOSE_INFO:
            print(f"[MCTSCuda._init_device_side_arrays() for {self}]")
            print(f"[per_state_memory: {self._per_state_memory} B,  calculated _max_tree_size: {self._max_tree_size}]")
        t1_dev_arrays = time.time()        
        self._dev_trees = cuda.device_array((self.n_trees, self._max_tree_size, 1 + self.state_max_actions), dtype=np.int32) # each row of a tree represents a node consisting of: parent indexes and indexes of all children (associated with actions), -1 index for none parent or child 
        self._dev_trees_sizes = cuda.device_array(self.n_trees, dtype=np.int32)
        self._dev_trees_depths = cuda.device_array((self.n_trees, self._max_tree_size), dtype=np.int16)
        self._dev_trees_turns = cuda.device_array((self.n_trees, self._max_tree_size), dtype=np.int8)
        self._dev_trees_leaves = cuda.device_array((self.n_trees, self._max_tree_size), dtype=bool)
        self._dev_trees_terminals = cuda.device_array((self.n_trees, self._max_tree_size), dtype=bool)
        self._dev_trees_outcomes = cuda.device_array((self.n_trees, self._max_tree_size), dtype=np.int8)        
        self._dev_trees_ns = cuda.device_array((self.n_trees, self._max_tree_size), dtype=np.int32)
        self._dev_trees_ns_wins = cuda.device_array((self.n_trees, self._max_tree_size), dtype=np.int32)
        self._dev_trees_boards = cuda.device_array((self.n_trees, self._max_tree_size, self.state_board_shape[0], self.state_board_shape[1]), dtype=np.int8)
        self._dev_trees_extra_infos = cuda.device_array((self.n_trees, self._max_tree_size, self.state_extra_info_memory), dtype=np.int8)
        self._dev_trees_nodes_selected = cuda.device_array(self.n_trees, dtype=np.int32)
        self._dev_trees_selected_paths = cuda.device_array((self.n_trees, MAX_TREE_DEPTH + 2), dtype=np.int32)
        self._dev_trees_actions_expanded = cuda.device_array((self.n_trees, self.state_max_actions + 2), dtype=np.int16) # +2 because 2 last entries inform about: child picked randomly for playout, number of actions (children) expanded            
        board_tpb = int(2**np.ceil(np.log2(np.prod(self.state_board_shape))))
        extra_info_tbp = int(2**np.ceil(np.log2(self.state_extra_info_memory))) if self.state_extra_info_memory > 0 else 1
        max_actions_tpb = int(2**np.ceil(np.log2(self.state_max_actions)))
        self._tpb_reset = min(max(board_tpb, extra_info_tbp), self._cuda_tpb_default)
        self._tpb_select = self._cuda_tpb_default
        self._tpb_expand_stage1 = min(max(self._tpb_reset, max_actions_tpb), self._cuda_tpb_default)
        self._dev_random_generators_expand_stage1 = create_xoroshiro128p_states(self.n_trees * self._tpb_expand_stage1, seed=self.seed)
        self._tpb_expand_stage2 = self._tpb_reset        
        self._dev_random_generators_playout = None
        if self.kind == "scpo":
            self._dev_random_generators_playout =  create_xoroshiro128p_states(self.n_trees * self.n_playouts, seed=0)
        elif self.kind == "acpo":
            self._dev_random_generators_playout =  create_xoroshiro128p_states(self.n_trees * self.state_max_actions * self.n_playouts, seed=self.seed)            
        self._dev_trees_playout_outcomes = cuda.device_array((self.n_trees, 2), dtype=np.int32) # each row stores counts of: -1 wins and +1 wins, respectively (for given tree) 
        self._dev_trees_playout_outcomes_children = None
        if self.kind == "acpo":
            self._dev_trees_playout_outcomes_children = cuda.device_array((self.n_trees, self.state_max_actions, 2), dtype=np.int32) # for each (playable) action, each row stores counts of: -1 wins and +1 wins, respectively (for given tree)            
        self._tpb_backup = self._cuda_tpb_default
        self._tbp_reduce_over_actions = min(max_actions_tpb, self._cuda_tpb_default)
        t2_dev_arrays = time.time()
        print(f"[device arrays initialized; time: {t2_dev_arrays - t1_dev_arrays} s; press any key]") 

    def __str__(self):         
        return f"{self.__class__.__name__}(n_trees={self.n_trees}, n_playouts={self.n_playouts}, kind='{self.kind}', search_time_limit={self.search_time_limit} s, search_steps_limit={self.search_steps_limit}, ucb1_c={self.ucb1_c}, seed: {self.seed}, max_device_memory={np.round(self.max_device_memory / 1024**3, 2)} GB)"
        
    def __repr__(self):
        repr_str = f"{str(self)}, "
        repr_str += f"state_board_shape={self.state_board_shape}, state_extra_info_memory={self.state_extra_info_memory}, state_max_actions={self.state_max_actions})"
        return repr_str 
    
    def run(self, root_board, root_extra_info, root_turn):
        self._run = getattr(self, "_run_" + self.kind)
        return self._run(root_board, root_extra_info, root_turn)
    
    def _run_scpo(self, root_board, root_extra_info, root_turn):
        print("MCTS_CUDA RUN SCPO...")
        print(f"[{self}]")
        t1 = time.time()            
        # MCTS reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self._tpb_reset
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSCuda._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                  self._dev_trees, self._dev_trees_sizes, self._dev_trees_depths, self._dev_trees_turns, self._dev_trees_leaves, self._dev_trees_terminals, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                  self._dev_trees_boards, self._dev_trees_extra_infos)    
        t2_reset = time.time()
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reset() done; time: {t2_reset - t1_reset} s]")
        total_time_select = 0.0
        total_time_expand = 0.0
        total_time_playout = 0.0
        total_time_backup = 0.0    
        step = 0
        trees_actions_expanded = np.empty((self.n_trees, self.state_max_actions + 2), dtype=np.int16)
        root_actions_expanded = np.empty(self.state_max_actions + 2, dtype=np.int16)
        while True:
            t2 = time.time()            
            if step >= self.search_steps_limit or t2 - t1 >= self.search_time_limit * (1.0 - self._search_time_limit_epsilon):
                break
            if self.VERBOSE_DEBUG:
                print(f"[step: {step + 1} starting, time used so far: {t2 - t1} s]")     
            # MCTS select
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self._tpb_select
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._select()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._select[bpg, tpb](self.state_max_actions, self.ucb1_c, 
                                       self._dev_trees, self._dev_trees_leaves, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                       self._dev_trees_nodes_selected)
            t2_select = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._select() done; time: {t2_select - t1_select} s]")
            total_time_select += t2_select - t1_select
            # MCTS expand            
            t1_expand = time.time()
            t1_expand_stage1 = time.time()
            bpg = self.n_trees
            tpb = self._tpb_expand_stage1
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_scpo_stage1()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSCuda._expand_scpo_stage1[bpg, tpb](self.state_max_actions, self._max_tree_size, 
                                                   self._dev_trees, self._dev_trees_sizes, self._dev_trees_turns, self._dev_trees_leaves, self._dev_trees_terminals,
                                                   self._dev_trees_boards, self._dev_trees_extra_infos, 
                                                   self._dev_trees_nodes_selected, self._dev_random_generators_expand_stage1, self._dev_trees_actions_expanded)                                                    
            self._dev_trees_actions_expanded.copy_to_host(ary=trees_actions_expanded)
            cuda.synchronize()
            if step == 0:
                root_actions_expanded = np.copy(trees_actions_expanded[0])
            t2_expand_stage1 = time.time()            
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_scpo_stage1() done; time: {t2_expand_stage1 - t1_expand_stage1} s]")
            t1_expand_stage2 = time.time()
            actions_expanded_cumsum = np.cumsum(trees_actions_expanded[:, -1])
            trees_actions_expanded_flat = np.empty((actions_expanded_cumsum[-1], 2), dtype=np.int16)
            shift = 0
            for ti in range(self.n_trees):
                s = slice(shift, actions_expanded_cumsum[ti])
                trees_actions_expanded_flat[s, 0] = ti
                trees_actions_expanded_flat[s, 1] = trees_actions_expanded[ti, :trees_actions_expanded[ti, -1]]
                shift = actions_expanded_cumsum[ti]                                        
            bpg = actions_expanded_cumsum[-1]            
            tpb = self._tpb_expand_stage2
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            dev_trees_actions_expanded_flat = cuda.to_device(trees_actions_expanded_flat)
            MCTSCuda._expand_stage2[bpg, tpb](self._dev_trees, self._dev_trees_depths, self._dev_trees_turns, self._dev_trees_leaves, self._dev_trees_terminals, self._dev_trees_outcomes, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                              self._dev_trees_boards, self._dev_trees_extra_infos,                                               
                                              self._dev_trees_nodes_selected, dev_trees_actions_expanded_flat)
            t2_expand_stage2 = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_stage2() done; time: {t2_expand_stage2 - t1_expand_stage2} s]")
            t2_expand = time.time()
            total_time_expand += t2_expand - t1_expand
            # MCTS playout
            t1_playout = time.time()
            bpg = self.n_trees
            tpb = self.n_playouts
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._playout_scpo()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._playout_scpo[bpg, tpb](self._dev_trees, self._dev_trees_turns, self._dev_trees_terminals, self._dev_trees_outcomes, 
                                             self._dev_trees_boards, self._dev_trees_extra_infos, 
                                             self._dev_trees_nodes_selected, self._dev_trees_actions_expanded, self._dev_random_generators_playout, self._dev_trees_playout_outcomes)
            t2_playout = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._playout_scpo() done; time: {t2_playout - t1_playout} s]")
            total_time_playout += t2_playout - t1_playout
            # MCTS backup
            t1_backup = time.time()
            tpb = self._tpb_backup
            bpg = (self.n_trees + tpb - 1) // tpb         
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_scpo[bpg, tpb](self.n_playouts,
                                            self._dev_trees, self._dev_trees_turns, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                            self._dev_trees_nodes_selected, self._dev_trees_actions_expanded, self._dev_trees_playout_outcomes)            
            t2_backup = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup() done; time: {t2_backup - t1_backup} s]")
            total_time_backup += t2_backup - t1_backup                                        
            step += 1        
        # MCTS sum reduction over trees for each root action        
        t1_reduce_over_trees = time.time()
        n_root_actions = int(root_actions_expanded[-1]) 
        bpg = n_root_actions
        tpb = int(2**np.ceil(np.log2(self.n_trees)))
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_trees()...; bpg: {bpg}, tpb: {tpb}]")
        dev_root_actions_expanded = cuda.to_device(root_actions_expanded)        
        dev_root_ns = cuda.device_array(n_root_actions, dtype=np.int64)
        dev_actions_ns = cuda.device_array(n_root_actions, dtype=np.int64)
        dev_actions_ns_wins = cuda.device_array(n_root_actions, dtype=np.int64)
        MCTSCuda._reduce_over_trees[bpg, tpb](self._dev_trees, self._dev_trees_ns, self._dev_trees_ns_wins, dev_root_actions_expanded, dev_root_ns, dev_actions_ns, dev_actions_ns_wins)
        root_ns = np.empty(n_root_actions, dtype=np.int64)
        actions_ns = np.empty(n_root_actions, dtype=np.int64)
        actions_ns_wins = np.empty(n_root_actions, dtype=np.int64)        
        dev_root_ns.copy_to_host(ary=root_ns)
        dev_actions_ns.copy_to_host(ary=actions_ns)
        dev_actions_ns_wins.copy_to_host(ary=actions_ns_wins)
        cuda.synchronize()
        t2_reduce_over_trees = time.time()
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_trees() done; time: {t2_reduce_over_trees - t1_reduce_over_trees} s]")                
        t2 = time.time()
        qs = -np.ones(self.state_max_actions)
        if self.VERBOSE_INFO:
            print("[action values:")
        for i in range(n_root_actions):
            q = actions_ns_wins[i] / actions_ns[i] if actions_ns[i] > 0 else np.nan
            ucb1 = q + self.ucb1_c * np.sqrt(np.log(root_ns[i]) / actions_ns[i]) if actions_ns[i] > 0 else np.nan
            qs[root_actions_expanded[i]] = q
            if self.VERBOSE_INFO:
                action_str = f"action: {root_actions_expanded[i]}, "
                if self.action_to_name_function:
                    action_str += f"name: {self.action_to_name_function(root_actions_expanded[i])}, "
                action_str += f"root_n: {root_ns[i]}, n: {actions_ns[i]}, n_wins: {actions_ns_wins[i]}, q: {q}, ucb1: {ucb1}"
                print(action_str) 
        if self.VERBOSE_INFO:
            print("]")            
        # MCTS sum reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self._tbp_reduce_over_actions
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_actions()...; bpg: {bpg}, tpb: {tpb}]")        
        dev_best_score = cuda.device_array(1, dtype=np.float32)
        dev_best_action = cuda.device_array(1, dtype=np.int16)
        MCTSCuda._reduce_over_actions[bpg, tpb](dev_actions_ns, dev_actions_ns_wins, dev_best_score, dev_best_action)                 
        best_score = dev_best_score.copy_to_host()[0]
        best_action = dev_best_action.copy_to_host()[0]        
        cuda.synchronize()
        best_action = root_actions_expanded[best_action]
        t2_reduce_over_actions = time.time()
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_actions() done; time: {t2_reduce_over_actions - t1_reduce_over_actions} s]")                
        t2 = time.time()
        # TODO suitable if-verbose around depths below (so that efficiency is increased) 
        cuda.synchronize()
        depths = self._dev_trees_depths.copy_to_host()
        sizes = self._dev_trees_sizes.copy_to_host()
        i_sizes = np.zeros(self.n_trees, dtype=np.int32)
        max_depth = -1        
        for i in range(self.n_trees):            
            i_depth = np.max(depths[i, :sizes[i]])
            i_sizes[i] = sizes[i]
            # print(f"[tree {i} -> size: {sizes[i]}, depth: {i_depth}]")
            max_depth = max(i_depth, max_depth)
        print(f"[steps performed: {step}]")
        print(f"[trees -> max depth: {max_depth}, max size: {np.max(i_sizes)}, mean size: {np.mean(i_sizes)}]")                    
        mus_factor = 10.0**6                    
        print(f"[mean times of stages [us] -> selection: {mus_factor * total_time_select / step}, expansion: {mus_factor * total_time_expand / step}, playout: {mus_factor * total_time_playout / step}, backup: {mus_factor * total_time_backup / step}]")
        print(f"[best action: {best_action}, best score: {best_score}, best q: {qs[best_action]}]")                        
        print(f"MCTS_CUDA RUN SCPO DONE. [time: {t2 - t1} s, steps: {step}]")        
        return best_action

    def _run_acpo_old(self, root_board, root_extra_info, root_turn):
        print("MCTS_CUDA RUN ACPO...")
        print(f"[{self}]")
        t1 = time.time()            
        # MCTS reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self._tpb_reset
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array        
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSCuda._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                  self._dev_trees, self._dev_trees_sizes, self._dev_trees_depths, self._dev_trees_turns, self._dev_trees_leaves, self._dev_trees_terminals, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                  self._dev_trees_boards, self._dev_trees_extra_infos)    
        t2_reset = time.time()
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reset() done; time: {t2_reset - t1_reset} s]")
        total_time_select = 0.0
        total_time_expand = 0.0        
        total_time_playout = 0.0
        total_time_backup = 0.0
        total_time_backup_1 = 0.0
        total_time_backup_2 = 0.0
        total_time_copying = 0.0        
        step = 0
        trees_actions_expanded = np.empty((self.n_trees, self.state_max_actions + 2), dtype=np.int16)
        root_actions_expanded = np.empty(self.state_max_actions + 2, dtype=np.int16)
        t1_loop = time.time()
        while True:
            t2 = time.time()
            if step >= self.search_steps_limit or t2 - t1 >= self.search_time_limit:
                break
            if self.VERBOSE_DEBUG:
                print(f"[step: {step + 1} starting, time used so far: {t2 - t1} s]")     
            # MCTS select
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self._tpb_select
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._select()...; bpg: {bpg}, tpb: {tpb}]")
            # MCTSCuda._select[bpg, tpb](self.state_max_actions, self.ucb1_c, 
            #                            self._dev_trees, self._dev_trees_leaves, self._dev_trees_ns, self._dev_trees_ns_wins, 
            #                            self._dev_trees_nodes_selected)
            MCTSCuda._select[bpg, tpb](self.state_max_actions, self.ucb1_c, 
                                       self._dev_trees, self._dev_trees_leaves, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                       self._dev_trees_nodes_selected, self._dev_trees_selected_paths)                     
            t2_select = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._select() done; time: {t2_select - t1_select} s]")
            total_time_select += t2_select - t1_select
                                        
            # MCTS expand                        
            t1_expand_stage1 = time.time()
            bpg = self.n_trees
            tpb = self._tpb_expand_stage1
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_acpo_stage1()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSCuda._expand_acpo_stage1[bpg, tpb](self.state_max_actions, self._max_tree_size, 
                                                   self._dev_trees, self._dev_trees_sizes, self._dev_trees_turns, self._dev_trees_leaves, self._dev_trees_terminals,
                                                   self._dev_trees_boards, self._dev_trees_extra_infos, 
                                                   self._dev_trees_nodes_selected, self._dev_trees_actions_expanded)
            t2_expand_stage1 = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_acpo_stage1() done; time: {t2_expand_stage1 - t1_expand_stage1} s]")
            total_time_expand += t2_expand_stage1 - t1_expand_stage1          
            t1_copy_after_expand_stage1 = time.time()                                                            
            self._dev_trees_actions_expanded.copy_to_host(ary=trees_actions_expanded)
            cuda.synchronize()
            if step == 0:
                root_actions_expanded = np.copy(trees_actions_expanded[0])
            t2_copy_after_expand_stage1 = time.time()            
            if self.VERBOSE_DEBUG:
                print(f"[copying after _expand_stage1 done; time: {t2_copy_after_expand_stage1 - t1_copy_after_expand_stage1} s, bytes: {self._dev_trees_actions_expanded.nbytes}, shape: {self._dev_trees_actions_expanded.shape}]")
            total_time_copying += t2_copy_after_expand_stage1 - t1_copy_after_expand_stage1            
            actions_expanded_cumsum = np.cumsum(trees_actions_expanded[:, -1])
            trees_actions_expanded_flat = np.empty((actions_expanded_cumsum[-1], 2), dtype=np.int16)
            shift = 0
            for ti in range(self.n_trees):
                s = slice(shift, actions_expanded_cumsum[ti])
                trees_actions_expanded_flat[s, 0] = ti
                trees_actions_expanded_flat[s, 1] = trees_actions_expanded[ti, :trees_actions_expanded[ti, -1]]
                shift = actions_expanded_cumsum[ti]                                                    
            t1_expand_stage2 = time.time()
            bpg = actions_expanded_cumsum[-1]            
            tpb = self._tpb_expand_stage2
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            dev_trees_actions_expanded_flat = cuda.to_device(trees_actions_expanded_flat)
            MCTSCuda._expand_stage2[bpg, tpb](self._dev_trees, self._dev_trees_depths, self._dev_trees_turns, self._dev_trees_leaves, self._dev_trees_terminals, self._dev_trees_outcomes, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                              self._dev_trees_boards, self._dev_trees_extra_infos,                                               
                                              self._dev_trees_nodes_selected, dev_trees_actions_expanded_flat)
            t2_expand_stage2 = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_stage2() done; time: {t2_expand_stage2 - t1_expand_stage2} s]")
            total_time_expand += t2_expand_stage2 - t1_expand_stage2
            # MCTS playout
            t1_playout = time.time()
            bpg = actions_expanded_cumsum[-1]
            tpb = self.n_playouts
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._playout_acpo()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._playout_acpo[bpg, tpb](self._dev_trees, self._dev_trees_turns, self._dev_trees_terminals, self._dev_trees_outcomes, 
                                             self._dev_trees_boards, self._dev_trees_extra_infos, 
                                             self._dev_trees_nodes_selected, self._dev_trees_actions_expanded, dev_trees_actions_expanded_flat,
                                             self._dev_random_generators_playout, self._dev_trees_playout_outcomes, self._dev_trees_playout_outcomes_children)
            t2_playout = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._playout_acpo() done; time: {t2_playout - t1_playout} s]")
            total_time_playout += t2_playout - t1_playout
            # MCTS backup
            t1_backup = time.time()
            t1_backup_stage1 = time.time()
            bpg = self.n_trees
            tpb = self._tpb_backup                     
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage1()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_acpo_stage1[bpg, tpb](self.n_playouts, 
                                                   self._dev_trees, self._dev_trees_turns, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                                   self._dev_trees_nodes_selected, self._dev_trees_actions_expanded, self._dev_trees_playout_outcomes, self._dev_trees_playout_outcomes_children)            
            t2_backup_stage1 = time.time()
            total_time_backup_1 += t2_backup_stage1 - t1_backup_stage1
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage1() done; time: {t2_backup_stage1 - t1_backup_stage1} s]")            
            t1_backup_stage2 = time.time()            
            # tpb = self._tpb_backup 
            # bpg = (self.n_trees + tpb) // tpb         
            # if self.VERBOSE_DEBUG:
            #     print(f"[MCTSCuda._backup_acpo_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            # MCTSCuda._backup_acpo_stage2[bpg, tpb](self.n_playouts,
            #                                        self._dev_trees, self._dev_trees_turns, self._dev_trees_ns, self._dev_trees_ns_wins, 
            #                                        self._dev_trees_nodes_selected, self._dev_trees_actions_expanded, self._dev_trees_playout_outcomes)
            tpb = self._tpb_backup
            bpg = self.n_trees         
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_acpo_stage2[bpg, tpb](self.n_playouts,
                                                   self._dev_trees_turns, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                                   self._dev_trees_nodes_selected, self._dev_trees_selected_paths, self._dev_trees_actions_expanded, self._dev_trees_playout_outcomes)                                    
            t2_backup_stage2 = time.time()
            total_time_backup_2 += t2_backup_stage2 - t1_backup_stage2
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage2() done; time: {t2_backup_stage2 - t1_backup_stage2} s]")
            t2_backup = time.time()
            total_time_backup += t2_backup - t1_backup                                        
            step += 1
        t2_loop = time.time()
                    
        # MCTS sum reduction over trees for each root action        
        t1_reduce_over_trees = time.time()
        n_root_actions = int(root_actions_expanded[-1]) 
        bpg = n_root_actions
        tpb = int(2**np.ceil(np.log2(self.n_trees)))
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_trees()...; bpg: {bpg}, tpb: {tpb}]")
        dev_root_actions_expanded = cuda.to_device(root_actions_expanded)        
        dev_root_ns = cuda.device_array(n_root_actions, dtype=np.int64)
        dev_actions_ns = cuda.device_array(n_root_actions, dtype=np.int64)
        dev_actions_ns_wins = cuda.device_array(n_root_actions, dtype=np.int64)
        MCTSCuda._reduce_over_trees[bpg, tpb](self._dev_trees, self._dev_trees_ns, self._dev_trees_ns_wins, dev_root_actions_expanded, dev_root_ns, dev_actions_ns, dev_actions_ns_wins)
        root_ns = np.empty(n_root_actions, dtype=np.int64)
        actions_ns = np.empty(n_root_actions, dtype=np.int64)
        actions_ns_wins = np.empty(n_root_actions, dtype=np.int64)        
        dev_root_ns.copy_to_host(ary=root_ns)
        dev_actions_ns.copy_to_host(ary=actions_ns)
        dev_actions_ns_wins.copy_to_host(ary=actions_ns_wins)
        cuda.synchronize()
        t2_reduce_over_trees = time.time()
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_trees() done; time: {t2_reduce_over_trees - t1_reduce_over_trees} s]")                
        t2 = time.time()
        qs = -np.ones(self.state_max_actions)                
        if self.VERBOSE_INFO:
            print("[action values:")
        for i in range(n_root_actions):
            q = actions_ns_wins[i] / actions_ns[i] if actions_ns[i] > 0 else np.nan
            ucb1 = q + self.ucb1_c * np.sqrt(np.log(root_ns[i]) / actions_ns[i]) if actions_ns[i] > 0 else np.nan
            qs[root_actions_expanded[i]] = q
            if self.VERBOSE_INFO:
                action_str = f"action: {root_actions_expanded[i]}, "
                if self.action_to_name_function:
                    action_str += f"name: {self.action_to_name_function(root_actions_expanded[i])}, "
                action_str += f"root_n: {root_ns[i]}, n: {actions_ns[i]}, n_wins: {actions_ns_wins[i]}, q: {q}, ucb1: {ucb1}"
                print(action_str) 
        if self.VERBOSE_INFO:
            print("]")                                                
        # MCTS sum reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self._tbp_reduce_over_actions
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_actions()...; bpg: {bpg}, tpb: {tpb}]")        
        dev_best_score = cuda.device_array(1, dtype=np.float32)
        dev_best_action = cuda.device_array(1, dtype=np.int16)
        MCTSCuda._reduce_over_actions[bpg, tpb](dev_actions_ns, dev_actions_ns_wins, dev_best_score, dev_best_action)                 
        best_score = dev_best_score.copy_to_host()[0]
        best_action = dev_best_action.copy_to_host()[0]        
        cuda.synchronize()
        best_action = root_actions_expanded[best_action]
        t2_reduce_over_actions = time.time()
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_actions() done; time: {t2_reduce_over_actions - t1_reduce_over_actions} s]")                
        t2 = time.time()
        # TODO suitable if-verbose around depths below (so that efficiency is increased) 
        cuda.synchronize()
        depths = self._dev_trees_depths.copy_to_host()
        sizes = self._dev_trees_sizes.copy_to_host()
        i_sizes = np.zeros(self.n_trees, dtype=np.int32)
        max_depth = -1        
        for i in range(self.n_trees):            
            i_depth = np.max(depths[i, :sizes[i]])
            i_sizes[i] = sizes[i]
            # print(f"[tree {i} -> size: {sizes[i]}, depth: {i_depth}]")
            max_depth = max(i_depth, max_depth)
        print(f"[steps performed: {step}]")
        print(f"[trees -> max depth: {max_depth}, max size: {np.max(i_sizes)}, mean size: {np.mean(i_sizes)}]")
        mus_factor = 10.0**6                    
        print(f"[loop time [us] -> total: {mus_factor * (t2_loop - t1_loop)}, mean: {mus_factor * (t2_loop - t1_loop) / step}]")
        print(f"[mean times of stages [us] -> selection: {mus_factor * total_time_select / step}, expansion: {mus_factor * total_time_expand / step}, playout: {mus_factor * total_time_playout / step}, backup: {mus_factor * total_time_backup / step}]")
        print(f"[mean times of stages [us] -> backup stage 1: {mus_factor * total_time_backup_1 / step}, backup stage 2: {mus_factor * total_time_backup_2 / step}]")
        print(f"[mean time of copying [us]: {mus_factor * total_time_copying / step}]") 
        print(f"[best action: {best_action}, best score: {best_score}, best q: {qs[best_action]}]")                        
        print(f"MCTS_CUDA RUN ACPO DONE. [time: {t2 - t1} s, steps: {step}]")        
        return best_action

    def _run_acpo(self, root_board, root_extra_info, root_turn):
        print("MCTS_CUDA RUN ACPO...")
        print(f"[{self}]")
        t1 = time.time()            
        # MCTS reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self._tpb_reset
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array        
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSCuda._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                  self._dev_trees, self._dev_trees_sizes, self._dev_trees_depths, self._dev_trees_turns, self._dev_trees_leaves, self._dev_trees_terminals, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                  self._dev_trees_boards, self._dev_trees_extra_infos)    
        t2_reset = time.time()
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reset() done; time: {t2_reset - t1_reset} s]")
        total_time_select = 0.0
        total_time_expand = 0.0        
        total_time_playout = 0.0
        total_time_backup = 0.0
        total_time_backup_1 = 0.0
        total_time_backup_2 = 0.0
        total_time_copying = 0.0        
        step = 0
        trees_actions_expanded = np.empty((self.n_trees, self.state_max_actions + 2), dtype=np.int16)
        root_actions_expanded = np.empty(self.state_max_actions + 2, dtype=np.int16)
        t1_loop = time.time()
        while True:
            t2 = time.time()
            if step >= self.search_steps_limit or t2 - t1 >= self.search_time_limit:
                break
            if self.VERBOSE_DEBUG:
                print(f"[step: {step + 1} starting, time used so far: {t2 - t1} s]")     
            # MCTS select
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self._tpb_select
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._select()...; bpg: {bpg}, tpb: {tpb}]")
            # MCTSCuda._select[bpg, tpb](self.state_max_actions, self.ucb1_c, 
            #                            self._dev_trees, self._dev_trees_leaves, self._dev_trees_ns, self._dev_trees_ns_wins, 
            #                            self._dev_trees_nodes_selected)
            MCTSCuda._select[bpg, tpb](self.state_max_actions, self.ucb1_c, 
                                       self._dev_trees, self._dev_trees_leaves, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                       self._dev_trees_nodes_selected, self._dev_trees_selected_paths)                     
            t2_select = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._select() done; time: {t2_select - t1_select} s]")
            total_time_select += t2_select - t1_select
                                        
            # MCTS expand                        
            t1_expand_stage1 = time.time()
            bpg = self.n_trees
            tpb = self._tpb_expand_stage1
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_acpo_stage1()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSCuda._expand_acpo_stage1[bpg, tpb](self.state_max_actions, self._max_tree_size, 
                                                   self._dev_trees, self._dev_trees_sizes, self._dev_trees_turns, self._dev_trees_leaves, self._dev_trees_terminals,
                                                   self._dev_trees_boards, self._dev_trees_extra_infos, 
                                                   self._dev_trees_nodes_selected, self._dev_trees_actions_expanded)
            t2_expand_stage1 = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_acpo_stage1() done; time: {t2_expand_stage1 - t1_expand_stage1} s]")
            total_time_expand += t2_expand_stage1 - t1_expand_stage1          
            t1_copy_after_expand_stage1 = time.time()                                                            
            self._dev_trees_actions_expanded.copy_to_host(ary=trees_actions_expanded)
            cuda.synchronize()
            if step == 0:
                root_actions_expanded = np.copy(trees_actions_expanded[0])
            t2_copy_after_expand_stage1 = time.time()            
            if self.VERBOSE_DEBUG:
                print(f"[copying after _expand_stage1 done; time: {t2_copy_after_expand_stage1 - t1_copy_after_expand_stage1} s, bytes: {self._dev_trees_actions_expanded.nbytes}, shape: {self._dev_trees_actions_expanded.shape}]")
            total_time_copying += t2_copy_after_expand_stage1 - t1_copy_after_expand_stage1            
            actions_expanded_cumsum = np.cumsum(trees_actions_expanded[:, -1])
            trees_actions_expanded_flat = np.empty((actions_expanded_cumsum[-1], 2), dtype=np.int16)
            shift = 0
            for ti in range(self.n_trees):
                s = slice(shift, actions_expanded_cumsum[ti])
                trees_actions_expanded_flat[s, 0] = ti
                trees_actions_expanded_flat[s, 1] = trees_actions_expanded[ti, :trees_actions_expanded[ti, -1]]
                shift = actions_expanded_cumsum[ti]                                                    
            t1_expand_stage2 = time.time()
            bpg = actions_expanded_cumsum[-1]            
            tpb = self._tpb_expand_stage2
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            dev_trees_actions_expanded_flat = cuda.to_device(trees_actions_expanded_flat)
            MCTSCuda._expand_stage2[bpg, tpb](self._dev_trees, self._dev_trees_depths, self._dev_trees_turns, self._dev_trees_leaves, self._dev_trees_terminals, self._dev_trees_outcomes, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                              self._dev_trees_boards, self._dev_trees_extra_infos,                                               
                                              self._dev_trees_nodes_selected, dev_trees_actions_expanded_flat)
            t2_expand_stage2 = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_stage2() done; time: {t2_expand_stage2 - t1_expand_stage2} s]")
            total_time_expand += t2_expand_stage2 - t1_expand_stage2
            # MCTS playout
            t1_playout = time.time()
            bpg = actions_expanded_cumsum[-1]
            tpb = self.n_playouts
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._playout_acpo()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._playout_acpo[bpg, tpb](self._dev_trees, self._dev_trees_turns, self._dev_trees_terminals, self._dev_trees_outcomes, 
                                             self._dev_trees_boards, self._dev_trees_extra_infos, 
                                             self._dev_trees_nodes_selected, self._dev_trees_actions_expanded, dev_trees_actions_expanded_flat,
                                             self._dev_random_generators_playout, self._dev_trees_playout_outcomes, self._dev_trees_playout_outcomes_children)
            t2_playout = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._playout_acpo() done; time: {t2_playout - t1_playout} s]")
            total_time_playout += t2_playout - t1_playout
            # MCTS backup
            t1_backup = time.time()
            t1_backup_stage1 = time.time()
            bpg = self.n_trees
            tpb = self._tpb_backup                     
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage1()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_acpo_stage1[bpg, tpb](self.n_playouts, 
                                                   self._dev_trees, self._dev_trees_turns, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                                   self._dev_trees_nodes_selected, self._dev_trees_actions_expanded, self._dev_trees_playout_outcomes, self._dev_trees_playout_outcomes_children)            
            t2_backup_stage1 = time.time()
            total_time_backup_1 += t2_backup_stage1 - t1_backup_stage1
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage1() done; time: {t2_backup_stage1 - t1_backup_stage1} s]")            
            t1_backup_stage2 = time.time()            
            # tpb = self._tpb_backup 
            # bpg = (self.n_trees + tpb) // tpb         
            # if self.VERBOSE_DEBUG:
            #     print(f"[MCTSCuda._backup_acpo_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            # MCTSCuda._backup_acpo_stage2[bpg, tpb](self.n_playouts,
            #                                        self._dev_trees, self._dev_trees_turns, self._dev_trees_ns, self._dev_trees_ns_wins, 
            #                                        self._dev_trees_nodes_selected, self._dev_trees_actions_expanded, self._dev_trees_playout_outcomes)
            tpb = self._tpb_backup
            bpg = self.n_trees         
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_acpo_stage2[bpg, tpb](self.n_playouts,
                                                   self._dev_trees_turns, self._dev_trees_ns, self._dev_trees_ns_wins, 
                                                   self._dev_trees_nodes_selected, self._dev_trees_selected_paths, self._dev_trees_actions_expanded, self._dev_trees_playout_outcomes)                                    
            t2_backup_stage2 = time.time()
            total_time_backup_2 += t2_backup_stage2 - t1_backup_stage2
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage2() done; time: {t2_backup_stage2 - t1_backup_stage2} s]")
            t2_backup = time.time()
            total_time_backup += t2_backup - t1_backup                                        
            step += 1
        t2_loop = time.time()
                    
        # MCTS sum reduction over trees for each root action        
        t1_reduce_over_trees = time.time()
        n_root_actions = int(root_actions_expanded[-1]) 
        bpg = n_root_actions
        tpb = int(2**np.ceil(np.log2(self.n_trees)))
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_trees()...; bpg: {bpg}, tpb: {tpb}]")
        dev_root_actions_expanded = cuda.to_device(root_actions_expanded)        
        dev_root_ns = cuda.device_array(n_root_actions, dtype=np.int64)
        dev_actions_ns = cuda.device_array(n_root_actions, dtype=np.int64)
        dev_actions_ns_wins = cuda.device_array(n_root_actions, dtype=np.int64)
        MCTSCuda._reduce_over_trees[bpg, tpb](self._dev_trees, self._dev_trees_ns, self._dev_trees_ns_wins, dev_root_actions_expanded, dev_root_ns, dev_actions_ns, dev_actions_ns_wins)
        root_ns = np.empty(n_root_actions, dtype=np.int64)
        actions_ns = np.empty(n_root_actions, dtype=np.int64)
        actions_ns_wins = np.empty(n_root_actions, dtype=np.int64)        
        dev_root_ns.copy_to_host(ary=root_ns)
        dev_actions_ns.copy_to_host(ary=actions_ns)
        dev_actions_ns_wins.copy_to_host(ary=actions_ns_wins)
        cuda.synchronize()
        t2_reduce_over_trees = time.time()
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_trees() done; time: {t2_reduce_over_trees - t1_reduce_over_trees} s]")                
        t2 = time.time()
        qs = -np.ones(self.state_max_actions)                
        if self.VERBOSE_INFO:
            print("[action values:")
        for i in range(n_root_actions):
            q = actions_ns_wins[i] / actions_ns[i] if actions_ns[i] > 0 else np.nan
            ucb1 = q + self.ucb1_c * np.sqrt(np.log(root_ns[i]) / actions_ns[i]) if actions_ns[i] > 0 else np.nan
            qs[root_actions_expanded[i]] = q
            if self.VERBOSE_INFO:
                action_str = f"action: {root_actions_expanded[i]}, "
                if self.action_to_name_function:
                    action_str += f"name: {self.action_to_name_function(root_actions_expanded[i])}, "
                action_str += f"root_n: {root_ns[i]}, n: {actions_ns[i]}, n_wins: {actions_ns_wins[i]}, q: {q}, ucb1: {ucb1}"
                print(action_str) 
        if self.VERBOSE_INFO:
            print("]")                                                
        # MCTS sum reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self._tbp_reduce_over_actions
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_actions()...; bpg: {bpg}, tpb: {tpb}]")        
        dev_best_score = cuda.device_array(1, dtype=np.float32)
        dev_best_action = cuda.device_array(1, dtype=np.int16)
        MCTSCuda._reduce_over_actions[bpg, tpb](dev_actions_ns, dev_actions_ns_wins, dev_best_score, dev_best_action)                 
        best_score = dev_best_score.copy_to_host()[0]
        best_action = dev_best_action.copy_to_host()[0]        
        cuda.synchronize()
        best_action = root_actions_expanded[best_action]
        t2_reduce_over_actions = time.time()
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_actions() done; time: {t2_reduce_over_actions - t1_reduce_over_actions} s]")                
        t2 = time.time()
        # TODO suitable if-verbose around depths below (so that efficiency is increased) 
        cuda.synchronize()
        depths = self._dev_trees_depths.copy_to_host()
        sizes = self._dev_trees_sizes.copy_to_host()
        i_sizes = np.zeros(self.n_trees, dtype=np.int32)
        max_depth = -1        
        for i in range(self.n_trees):            
            i_depth = np.max(depths[i, :sizes[i]])
            i_sizes[i] = sizes[i]
            # print(f"[tree {i} -> size: {sizes[i]}, depth: {i_depth}]")
            max_depth = max(i_depth, max_depth)
        print(f"[steps performed: {step}]")
        print(f"[trees -> max depth: {max_depth}, max size: {np.max(i_sizes)}, mean size: {np.mean(i_sizes)}]")
        mus_factor = 10.0**6                    
        print(f"[loop time [us] -> total: {mus_factor * (t2_loop - t1_loop)}, mean: {mus_factor * (t2_loop - t1_loop) / step}]")
        print(f"[mean times of stages [us] -> selection: {mus_factor * total_time_select / step}, expansion: {mus_factor * total_time_expand / step}, playout: {mus_factor * total_time_playout / step}, backup: {mus_factor * total_time_backup / step}]")
        print(f"[mean times of stages [us] -> backup stage 1: {mus_factor * total_time_backup_1 / step}, backup stage 2: {mus_factor * total_time_backup_2 / step}]")
        print(f"[mean time of copying [us]: {mus_factor * total_time_copying / step}]") 
        print(f"[best action: {best_action}, best score: {best_score}, best q: {qs[best_action]}]")                        
        print(f"MCTS_CUDA RUN ACPO DONE. [time: {t2 - t1} s, steps: {step}]")        
        return best_action

    @staticmethod
    @cuda.jit(void(int8[:, :], int8[:], int8, int32[:, :, :], int32[:], int16[:, :], int8[:, :], boolean[:, :], boolean[:, :], int32[:, :], int32[:, :], int8[:, :, :, :], int8[:, :, :]))
    def _reset(root_board, root_extra_info, root_turn, trees, trees_sizes, trees_depths, trees_turns, trees_leaves, trees_terminals, trees_ns, trees_ns_wins, trees_boards, trees_extra_infos):        
        ti = cuda.blockIdx.x # tree index 
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x                
        if t == 0:
            trees[ti, 0, 0] = int32(-1)
            trees_sizes[ti] = int32(1)
            trees_depths[ti, 0] = int32(0)
            trees_turns[ti, 0] = int8(root_turn)
            trees_leaves[ti, 0] = True
            trees_terminals[ti, 0] = False
            trees_ns[ti, 0] = int32(0)
            trees_ns_wins[ti, 0] = int32(0)            
        m, n = root_board.shape
        m_n = m * n
        bept = (m_n + tpb - 1) // tpb # board elements per thread
        e = t # board element flat index
        for _ in range(bept):
            if e < m_n:
                i = e // n
                j = e % n
                trees_boards[ti, 0, i, j] = root_board[i, j]
            e += tpb        
        extra_info_memory = root_extra_info.size
        eipt = (extra_info_memory + tpb - 1) // tpb
        e = t
        for _ in range(eipt):
            if e < extra_info_memory:
                trees_extra_infos[ti, 0, e] = root_extra_info[e] 

    @staticmethod
    @cuda.jit(void(int16, float32, int32[:, :, :], boolean[:, :], int32[:, :], int32[:, :], int32[:], int32[:, :]))        
    def _select(state_max_actions, ucb1_c, trees, trees_leaves, trees_ns, trees_ns_wins, trees_nodes_selected, trees_selected_paths):
        shared_ucb1s = cuda.shared.array(1024, dtype=float32) # 1024 - assumed limit on max actions
        shared_best_child = cuda.shared.array(1024, dtype=int32) # 1024 - assumed limit on max actions (array instead of one index due to max-argmax reduction pattern)
        shared_selected_path = cuda.shared.array(2048 + 2, dtype=int32) # 2048 - assumed equal to MAX_TREE_DEPTH + 2 
        ti = cuda.blockIdx.x # tree index 
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        node = int32(0)
        depth = int16(0)
        if t == 0:
            shared_selected_path[0] = int32(0) # path always starting from root
        while not trees_leaves[ti, node]:
            if t < state_max_actions:
                child = trees[ti, node, 1 + t]
                shared_best_child[t] = child                
                if child == int32(-1):
                    shared_ucb1s[t] = -float32(inf)
                else:
                    child_n = trees_ns[ti, child]             
                    if child_n == int32(0):
                        shared_ucb1s[t] = float32(inf)
                    else:                        
                        shared_ucb1s[t] = trees_ns_wins[ti, child] / float32(child_n) + ucb1_c * math.sqrt(math.log(trees_ns[ti, node]) / child_n)
            else:
                shared_ucb1s[t] = -float32(inf)
            cuda.syncthreads()
            stride = tpb >> 1 # half of tpb
            while stride > 0: # max-argmax reduction pattern
                if t < stride:
                    t_stride = t + stride
                    if shared_ucb1s[t] < shared_ucb1s[t_stride]:
                        shared_ucb1s[t] = shared_ucb1s[t_stride]
                        shared_best_child[t] = shared_best_child[t_stride]    
                cuda.syncthreads()
                stride >>= 1
            node = shared_best_child[0]
            depth += int16(1)
            if t == 0:
                shared_selected_path[depth] = node                                            
        path_length = depth + 1
        pept = (path_length + tpb - 1) // tpb # path elements per thread
        e = t
        for _ in range(pept):
            if e < path_length:
                trees_selected_paths[ti, e] = shared_selected_path[e]
            e += tpb
        if t == 0:
            trees_nodes_selected[ti] = node
            trees_selected_paths[ti, -1] = path_length
      
    # TODO remove _select_old when not needed
    @staticmethod
    @cuda.jit(void(int16, float32, int32[:, :, :], boolean[:, :], int32[:, :], int32[:, :], int32[:]))        
    def _select_old(state_max_actions, ucb1_c, trees, trees_leaves, trees_ns, trees_ns_wins, trees_nodes_selected):
        shared_ucb1s = cuda.shared.array(1024, dtype=float32) # 1024 - assumed limit on max actions
        shared_best_child = cuda.shared.array(1024, dtype=int32) # 1024 - assumed limit on max actions (array instead of one index due to max-argmax reduction pattern) 
        ti = cuda.blockIdx.x # tree index 
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        node = 0
        while not trees_leaves[ti, node]:            
            if t < state_max_actions:
                child = trees[ti, node, 1 + t]
                shared_best_child[t] = child                
                if child == int32(-1):
                    shared_ucb1s[t] = -float32(inf)
                else:
                    child_n = trees_ns[ti, child]             
                    if child_n == int32(0):
                        shared_ucb1s[t] = float32(inf)
                    else:                        
                        shared_ucb1s[t] = trees_ns_wins[ti, child] / float32(child_n) + ucb1_c * math.sqrt(math.log(trees_ns[ti, node]) / child_n)
            else:
                shared_ucb1s[t] = -float32(inf)
            cuda.syncthreads()
            stride = tpb >> 1 # half of tpb
            while stride > 0: # max-argmax reduction pattern
                if t < stride:
                    t_stride = t + stride
                    if shared_ucb1s[t] < shared_ucb1s[t_stride]:
                        shared_ucb1s[t] = shared_ucb1s[t_stride]
                        shared_best_child[t] = shared_best_child[t_stride]    
                cuda.syncthreads()
                stride >>= 1
            node = shared_best_child[0]
        if t == 0:
            trees_nodes_selected[ti] = node

    @staticmethod
    @cuda.jit(void(int16, int32, int32[:, :, :], int32[:], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], xoroshiro128p_type[:], int16[:, :]))
    def _expand_scpo_stage1(state_max_actions, max_tree_size, trees, trees_sizes, trees_turns, trees_leaves, trees_terminals, trees_boards, trees_extra_infos, 
                            trees_nodes_selected, random_generators_expand_stage1, trees_actions_expanded):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_legal_actions = cuda.shared.array(1024, dtype=boolean) # 1024 - assumed limit on max actions
        shared_legal_actions_child_shifts = cuda.shared.array(1024, dtype=int16) # 1024 - assumed limit on max actions
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        t_global = cuda.grid(1)
        _, _, m, n = trees_boards.shape
        m_n = m * n
        bept = (m_n + tpb - 1) // tpb # board elements per thread
        e = t # board element flat index
        selected = trees_nodes_selected[ti] # node selected
        for _ in range(bept):
            if e < m_n:
                i = e // n
                j = e % n
                shared_board[i, j] = trees_boards[ti, selected, i, j]
            e += tpb        
        _, _, extra_info_memory = trees_extra_infos.shape
        eipt = (extra_info_memory + tpb - 1) // tpb
        e = t
        for _ in range(eipt):
            if e < extra_info_memory:
                shared_extra_info[e] = trees_extra_infos[ti, selected, e]
            e += tpb
        cuda.syncthreads()
        selected_is_terminal = trees_terminals[ti, selected]
        if selected_is_terminal:
            shared_legal_actions[t] = False
        elif t < state_max_actions:            
            is_action_legal(m, n, shared_board, shared_extra_info, trees_turns[ti, selected], t, shared_legal_actions)            
        cuda.syncthreads() 
        size_so_far = trees_sizes[ti]
        child_shift = int16(-1)
        if t < state_max_actions:
            shared_legal_actions_child_shifts[t] = int16(-1)
        if t == 0:
            if not selected_is_terminal:
                for i in range(state_max_actions):
                    if shared_legal_actions[i] and size_so_far + child_shift + 1 < max_tree_size:
                        child_shift += 1
                    shared_legal_actions_child_shifts[i] = child_shift                                
                trees_actions_expanded[ti, -1] = child_shift + 1 # information for next kernel how many children expanded (as last entry in trees_actions_expanded)
                if child_shift >= 0:
                    trees_leaves[ti, selected] = False                                
                rand_child_for_playout = int16(xoroshiro128p_uniform_float32(random_generators_expand_stage1, t_global) * (child_shift + 1))
                trees_actions_expanded[ti, -2] = rand_child_for_playout
            else:
                trees_actions_expanded[ti, -1] = int16(1)
                trees_actions_expanded[ti, -2] = int16(-1) # fake child for playout indicating that selected is terminal (and playout to be done from him)
        cuda.syncthreads()        
        if t < state_max_actions: 
            child_index = int32(-1)
            if shared_legal_actions[t]:
                child_shift = shared_legal_actions_child_shifts[t]
                child_index = size_so_far + child_shift                
                trees_actions_expanded[ti, child_shift] = t
            trees[ti, selected, 1 + t] = child_index # parent gets to know where child is 
        if t == 0:
            trees_sizes[ti] += shared_legal_actions_child_shifts[state_max_actions - 1] + 1 # updating tree size            
        
    @staticmethod
    @cuda.jit(void(int16, int32, int32[:, :, :], int32[:], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :]))
    def _expand_acpo_stage1(state_max_actions, max_tree_size, trees, trees_sizes, trees_turns, trees_leaves, trees_terminals, trees_boards, trees_extra_infos, 
                            trees_nodes_selected, trees_actions_expanded):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_legal_actions = cuda.shared.array(1024, dtype=boolean) # 1024 - assumed limit on max actions
        shared_legal_actions_child_shifts = cuda.shared.array(1024, dtype=int16) # 1024 - assumed limit on max actions
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        _, _, m, n = trees_boards.shape
        m_n = m * n
        bept = (m_n + tpb - 1) // tpb # board elements per thread
        e = t # board element flat index
        selected = trees_nodes_selected[ti] # node selected
        for _ in range(bept):
            if e < m_n:
                i = e // n
                j = e % n
                shared_board[i, j] = trees_boards[ti, selected, i, j]
            e += tpb        
        _, _, extra_info_memory = trees_extra_infos.shape
        eipt = (extra_info_memory + tpb - 1) // tpb
        e = t
        for _ in range(eipt):
            if e < extra_info_memory:
                shared_extra_info[e] = trees_extra_infos[ti, selected, e]
            e += tpb
        cuda.syncthreads()
        selected_is_terminal = trees_terminals[ti, selected]
        if selected_is_terminal:
            shared_legal_actions[t] = False
        elif t < state_max_actions:            
            is_action_legal(m, n, shared_board, shared_extra_info, trees_turns[ti, selected], t, shared_legal_actions)            
        cuda.syncthreads() 
        size_so_far = trees_sizes[ti]
        child_shift = int16(-1)
        if t < state_max_actions:
            shared_legal_actions_child_shifts[t] = int16(-1)
        if t == 0:
            if not selected_is_terminal:
                for i in range(state_max_actions):
                    if shared_legal_actions[i] and size_so_far + child_shift + 1 < max_tree_size:
                        child_shift += 1
                    shared_legal_actions_child_shifts[i] = child_shift
                if child_shift >= 0:
                    trees_leaves[ti, selected] = False
                trees_actions_expanded[ti, -1] = child_shift + 1 # information for next kernel how many children expanded (as last entry in trees_actions_expanded)                
                trees_actions_expanded[ti, -2] = int16(-2) # indicates all children for playout (acpo)                                
            else:
                trees_actions_expanded[ti, -1] = int16(1)
                trees_actions_expanded[ti, -2] = int16(-1) # fake child for playout indicating that selected is terminal (and playout to be done from him)                
        cuda.syncthreads()        
        if t < state_max_actions: 
            child_index = int32(-1)
            if shared_legal_actions[t]:
                child_shift = shared_legal_actions_child_shifts[t]
                child_index = size_so_far + child_shift                
                trees_actions_expanded[ti, child_shift] = t
            trees[ti, selected, 1 + t] = child_index # parent gets to know where child is 
        if t == 0:
            trees_sizes[ti] += shared_legal_actions_child_shifts[state_max_actions - 1] + 1 # updating tree size        
        
    @staticmethod
    @cuda.jit(void(int32[:, :, :], int16[:, :], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :], int32[:, :], int32[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :]))
    def _expand_stage2(trees, trees_depths, trees_turns, trees_leaves, trees_terminals, trees_outcomes, trees_ns, trees_ns_wins, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded_flat):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        tai = cuda.blockIdx.x # tree-action pair index
        ti = trees_actions_expanded_flat[tai, 0]
        action = trees_actions_expanded_flat[tai, 1]        
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        _, _, m, n = trees_boards.shape
        m_n = m * n
        bept = (m_n + tpb - 1) // tpb # board elements per thread
        e = t # board element flat index
        selected = trees_nodes_selected[ti]
        if trees_terminals[ti, selected]:
            return 
        for _ in range(bept):
            if e < m_n:
                i = e // n
                j = e % n
                shared_board[i, j] = trees_boards[ti, selected, i, j]
            e += tpb        
        _, _, extra_info_memory = trees_extra_infos.shape
        eipt = (extra_info_memory + tpb - 1) // tpb
        e = t
        for _ in range(eipt):
            if e < extra_info_memory:
                shared_extra_info[e] = trees_extra_infos[ti, selected, e]
            e += tpb
        cuda.syncthreads()
        turn = 0
        if t == 0:
            turn = trees_turns[ti, selected]
            take_action(m, n, shared_board, shared_extra_info, turn, action)
        cuda.syncthreads()        
        child = trees[ti, selected, 1 + action]
        e = t
        for _ in range(bept):
            if e < m_n:
                i = e // n
                j = e % n
                trees_boards[ti, child, i, j] = shared_board[i, j] 
            e += tpb        
        e = t
        for _ in range(eipt):
            if e < extra_info_memory:
                trees_extra_infos[ti, child, e] = shared_extra_info[e] 
            e += tpb
        if t == 0:
            trees[ti, child, 0] = selected             
            trees_turns[ti, child] = -turn
            trees_leaves[ti, child] = True
            terminal_flag = False
            outcome = compute_outcome(m, n, shared_board, shared_extra_info, -turn, action)            
            if outcome <= int8(1):
                terminal_flag = True
            trees_terminals[ti, child] = terminal_flag
            trees_outcomes[ti, child] = outcome
            trees_ns[ti, child] = int32(0)
            trees_ns_wins[ti, child] = int32(0)
            trees_depths[ti, child] = trees_depths[ti, selected] + 1                                    
                            
    @staticmethod
    @cuda.jit(void(int32[:, :, :], int8[:, :], boolean[:, :], int8[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :], xoroshiro128p_type[:], int32[:, :]))
    def _playout_scpo(trees, trees_turns, trees_terminals, trees_outcomes, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded, random_generators_playout, trees_playout_outcomes):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_playout_outcomes = cuda.shared.array((1024, 2), dtype=int16) # 1024 - assumed max tpb for playouts, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout 
        local_board = cuda.local.array((32, 32), dtype=int8)
        local_extra_info = cuda.local.array(4096, dtype=int8)
        local_legal_actions_with_count = cuda.local.array(1024 + 1, dtype=int16) # 1024 - assumed limit on max actions
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        selected = trees_nodes_selected[ti]
        rand_child_for_playout = trees_actions_expanded[ti, -2]
        last_action = int8(-1) # none yet
        if rand_child_for_playout != int16(-1): # check if some child picked on random for playout
            last_action = trees_actions_expanded[ti, rand_child_for_playout]
            selected = trees[ti, selected, 1 + int32(last_action)]
        if trees_terminals[ti, selected]: # root for playout has been discovered terminal before (by game rules) -> taking stored outcome (multiplied by tpb)
            if t == 0:            
                outcome = trees_outcomes[ti, selected]
                trees_playout_outcomes[ti, 0] = tpb if outcome == int8(-1) else int32(0) # wins of -1
                trees_playout_outcomes[ti, 1] = tpb if outcome == int8(1) else int32(0) # wins of +1
        else:
            t = cuda.threadIdx.x
            t_global = cuda.grid(1)
            shared_playout_outcomes[t, 0] = np.int16(0)
            shared_playout_outcomes[t, 1] = np.int16(0)
            _, _, m, n = trees_boards.shape
            m_n = m * n
            bept = (m_n + tpb - 1) // tpb # board elements per thread
            e = t # board element flat index
            for _ in range(bept):
                if e < m_n:
                    i = e // n
                    j = e % n
                    shared_board[i, j] = trees_boards[ti, selected, i, j]
                e += tpb        
            _, _, extra_info_memory = trees_extra_infos.shape
            eipt = (extra_info_memory + tpb - 1) // tpb
            e = t
            for _ in range(eipt):
                if e < extra_info_memory:
                    shared_extra_info[e] = trees_extra_infos[ti, selected, e]
                e += tpb
            cuda.syncthreads()
            for i in range(m):
                for j in range(n):
                    local_board[i, j] = shared_board[i, j]
            for i in range(extra_info_memory):
                local_extra_info[i] = shared_extra_info[i]                
            local_legal_actions_with_count[-1] = 0
            playout_depth = int16(0)
            turn = trees_turns[ti, selected]
            while True: # playout loop
                outcome = compute_outcome(m, n, local_board, local_extra_info, turn, last_action)
                if outcome > int8(1): # indecisive, game ongoing
                    legal_actions_playout(m, n, local_board, local_extra_info, turn, local_legal_actions_with_count)
                    count = local_legal_actions_with_count[-1]
                    action_ord = int16(xoroshiro128p_uniform_float32(random_generators_playout, t_global) * count)
                    last_action = local_legal_actions_with_count[action_ord]
                    take_action_playout(m, n, local_board, local_extra_info, turn, last_action, action_ord, local_legal_actions_with_count)                    
                    turn = -turn
                else:
                    if playout_depth == int16(0):
                        if t == 0:
                            trees_terminals[ti, selected] = True
                            trees_outcomes[ti, selected] = outcome
                    if outcome != int8(0):
                        shared_playout_outcomes[t, (outcome + 1) // 2] = int8(1)
                    break
                playout_depth += 1
            cuda.syncthreads()
            stride = tpb >> 1 # half of tpb
            while stride > 0: # max-argmax reduction pattern
                if t < stride:
                    t_stride = t + stride
                    shared_playout_outcomes[t, 0] += shared_playout_outcomes[t_stride, 0]
                    shared_playout_outcomes[t, 1] += shared_playout_outcomes[t_stride, 1]
                cuda.syncthreads()
                stride >>= 1
            if t == 0:
                trees_playout_outcomes[ti, 0] = shared_playout_outcomes[0, 0]
                trees_playout_outcomes[ti, 1] = shared_playout_outcomes[0, 1]
    
    @staticmethod
    @cuda.jit(void(int32[:, :, :], int8[:, :], boolean[:, :], int8[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :], int16[:, :], xoroshiro128p_type[:], int32[:, :], int32[:, :, :]))
    def _playout_acpo(trees, trees_turns, trees_terminals, trees_outcomes, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded, trees_actions_expanded_flat, random_generators_playout, trees_playout_outcomes, 
                      trees_playout_outcomes_children):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_playout_outcomes = cuda.shared.array((1024, 2), dtype=int16) # 1024 - assumed max tpb for playouts, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout 
        local_board = cuda.local.array((32, 32), dtype=int8)
        local_extra_info = cuda.local.array(4096, dtype=int8)
        local_legal_actions_with_count = cuda.local.array(1024 + 1, dtype=int16) # 1024 - assumed limit on max actions        
        tai = cuda.blockIdx.x # tree-action pair index
        ti = trees_actions_expanded_flat[tai, 0]
        action = trees_actions_expanded_flat[tai, 1]  
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        selected = trees_nodes_selected[ti]
        fake_child_for_playout = trees_actions_expanded[ti, -2]
        last_action = int8(-1) # none yet
        if fake_child_for_playout != int16(-1): # check if true playouts are to be made (on all children of selected)
            last_action = action
            selected = trees[ti, selected, 1 + int32(last_action)]
        if trees_terminals[ti, selected]: # root for playout has been discovered terminal before (by game rules) -> taking stored outcome (multiplied by tpb)
            if t == 0:
                outcome = trees_outcomes[ti, selected]
                if fake_child_for_playout != int16(-1):                
                    trees_playout_outcomes_children[ti, action, 0] = tpb if outcome == int8(-1) else int32(0) # wins of -1
                    trees_playout_outcomes_children[ti, action, 1] = tpb if outcome == int8(1) else int32(0) # wins of +1
                else:
                    trees_playout_outcomes[ti, 0] = tpb if outcome == int8(-1) else int32(0) # wins of -1
                    trees_playout_outcomes[ti, 1] = tpb if outcome == int8(1) else int32(0) # wins of +1
        else:
            t = cuda.threadIdx.x
            t_global = cuda.grid(1)
            shared_playout_outcomes[t, 0] = np.int16(0)
            shared_playout_outcomes[t, 1] = np.int16(0)
            _, _, m, n = trees_boards.shape
            m_n = m * n
            bept = (m_n + tpb - 1) // tpb # board elements per thread
            e = t # board element flat index
            for _ in range(bept):
                if e < m_n:
                    i = e // n
                    j = e % n
                    shared_board[i, j] = trees_boards[ti, selected, i, j]
                e += tpb        
            _, _, extra_info_memory = trees_extra_infos.shape
            eipt = (extra_info_memory + tpb - 1) // tpb
            e = t
            for _ in range(eipt):
                if e < extra_info_memory:
                    shared_extra_info[e] = trees_extra_infos[ti, selected, e]
                e += tpb
            cuda.syncthreads()
            for i in range(m):
                for j in range(n):
                    local_board[i, j] = shared_board[i, j]
            for i in range(extra_info_memory):
                local_extra_info[i] = shared_extra_info[i]
            local_legal_actions_with_count[-1] = 0
            playout_depth = int16(0)            
            turn = trees_turns[ti, selected]
            while True: # playout loop
                outcome = compute_outcome(m, n, local_board, local_extra_info, turn, last_action)
                if outcome > int8(1): # indecisive, game ongoing
                    legal_actions_playout(m, n, local_board, local_extra_info, turn, local_legal_actions_with_count)
                    count = local_legal_actions_with_count[-1]
                    action_ord = int16(xoroshiro128p_uniform_float32(random_generators_playout, t_global) * count)
                    last_action = local_legal_actions_with_count[action_ord]
                    take_action_playout(m, n, local_board, local_extra_info, turn, last_action, action_ord, local_legal_actions_with_count)
                    turn = -turn
                else:
                    if playout_depth == int16(0):
                        if t == 0:
                            trees_terminals[ti, selected] = True
                            trees_outcomes[ti, selected] = outcome
                    if outcome != int8(0):
                        shared_playout_outcomes[t, (outcome + 1) // 2] = int8(1)
                    break
                playout_depth += 1
            cuda.syncthreads()
            stride = tpb >> 1 # half of tpb
            while stride > 0: # max-argmax reduction pattern
                if t < stride:
                    t_stride = t + stride
                    shared_playout_outcomes[t, 0] += shared_playout_outcomes[t_stride, 0]
                    shared_playout_outcomes[t, 1] += shared_playout_outcomes[t_stride, 1]
                cuda.syncthreads()
                stride >>= 1
            if t == 0:
                trees_playout_outcomes_children[ti, action, 0] = shared_playout_outcomes[0, 0]
                trees_playout_outcomes_children[ti, action, 1] = shared_playout_outcomes[0, 1]    
    
    @staticmethod
    @cuda.jit(void(int16, int32[:, :, :], int8[:, :], int32[:, :], int32[:, :], int32[:], int16[:, :], int32[:, :]))
    def _backup_scpo(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_actions_expanded, trees_playout_outcomes):
        ti = cuda.grid(1)
        n_trees = trees.shape[0]
        if ti < n_trees:
            node = trees_nodes_selected[ti]
            rand_child_for_playout = trees_actions_expanded[ti, -2]
            if rand_child_for_playout != int16(-1): # check if some child picked on random for playout
                last_action = trees_actions_expanded[ti, rand_child_for_playout]
                node = trees[ti, node, 1 + int32(last_action)]
            n_negative_wins = trees_playout_outcomes[ti, 0]
            n_positive_wins = trees_playout_outcomes[ti, 1]
            while True:
                trees_ns[ti, node] += n_playouts
                if trees_turns[ti, node] == int8(1):
                    trees_ns_wins[ti, node] += n_negative_wins 
                else:
                    trees_ns_wins[ti, node] += n_positive_wins
                node = trees[ti, node, 0]
                if node == int32(-1):
                    break

    @staticmethod
    @cuda.jit(void(int16, int32[:, :, :], int8[:, :], int32[:, :], int32[:, :], int32[:], int16[:, :], int32[:, :], int32[:, :, :]))
    def _backup_acpo_stage1(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_actions_expanded, trees_playout_outcomes, trees_playout_outcomes_children):
        shared_playout_outcomes_children = cuda.shared.array((1024, 2), dtype=int32) # 1024 - assumed max tpb for playouts, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout 
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x        
        fake_child_for_playout = trees_actions_expanded[ti, -2]
        if fake_child_for_playout != int16(-1): # check if selected is not terminal
            selected = trees_nodes_selected[ti]
            n_expanded_actions = trees_actions_expanded[ti, -1]
            if t < n_expanded_actions:
                a = trees_actions_expanded[ti, t]
                n_negative_wins = trees_playout_outcomes_children[ti, a, 0]
                n_positive_wins = trees_playout_outcomes_children[ti, a, 1]
                child_node = trees[ti, selected, 1 + a]
                trees_ns[ti, child_node] += n_playouts
                if trees_turns[ti, child_node] == int8(1):
                    trees_ns_wins[ti, child_node] += n_negative_wins 
                else:
                    trees_ns_wins[ti, child_node] += n_positive_wins
                shared_playout_outcomes_children[t, 0] = n_negative_wins
                shared_playout_outcomes_children[t, 1] = n_positive_wins
            else:
                shared_playout_outcomes_children[t, 0] = np.int32(0)
                shared_playout_outcomes_children[t, 1] = np.int32(0)                
            cuda.syncthreads()
            stride = tpb >> 1
            while stride > 0:
                if t < stride:
                    t_stride = t + stride
                    shared_playout_outcomes_children[t, 0] += shared_playout_outcomes_children[t_stride, 0]
                    shared_playout_outcomes_children[t, 1] += shared_playout_outcomes_children[t_stride, 1]
                cuda.syncthreads()                    
                stride >>= 1                
            if t == 0:
                trees_playout_outcomes[ti, 0] = shared_playout_outcomes_children[0, 0]
                trees_playout_outcomes[ti, 1] = shared_playout_outcomes_children[0, 1]

    @staticmethod
    @cuda.jit(void(int16, int8[:, :], int32[:, :], int32[:, :], int32[:], int32[:, :], int16[:, :], int32[:, :]))
    def _backup_acpo_stage2(n_playouts, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_selected_paths, trees_actions_expanded, trees_playout_outcomes):
        ti = cuda.blockIdx.x
        t = cuda.threadIdx.x
        tpb = cuda.blockDim.x
        n_negative_wins = trees_playout_outcomes[ti, 0]
        n_positive_wins = trees_playout_outcomes[ti, 1]
        n_expanded_actions = trees_actions_expanded[ti, -1]
        if n_expanded_actions == int16(0):
            n_expanded_actions = int16(1)
        n_playouts_total = n_playouts * n_expanded_actions
        path_length = trees_selected_paths[ti, -1]
        pept = (path_length + tpb - 1) // tpb # path elements per thread
        e = t
        for _ in range(pept):
            if e < path_length:                
                node = trees_selected_paths[ti, e]
                trees_ns[ti, node] += n_playouts_total
                if trees_turns[ti, node] == int8(1):
                    trees_ns_wins[ti, node] += n_negative_wins 
                else:
                    trees_ns_wins[ti, node] += n_positive_wins                
            e += tpb
            
    @staticmethod
    @cuda.jit(void(int16, int32[:, :, :], int8[:, :], int32[:, :], int32[:, :], int32[:], int16[:, :], int32[:, :]))
    def _backup_acpo_stage2_old(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_actions_expanded, trees_playout_outcomes):
        ti = cuda.grid(1)
        n_trees = trees.shape[0]
        if ti >= n_trees:
            return
        node = trees_nodes_selected[ti]
        n_negative_wins = trees_playout_outcomes[ti, 0]
        n_positive_wins = trees_playout_outcomes[ti, 1]
        n_expanded_actions = trees_actions_expanded[ti, -1]
        if n_expanded_actions == int16(0):
            n_expanded_actions = int16(1)
        n_playouts_total = n_playouts * n_expanded_actions
        while True:
            trees_ns[ti, node] += n_playouts_total
            if trees_turns[ti, node] == int8(1):
                trees_ns_wins[ti, node] += n_negative_wins 
            else:
                trees_ns_wins[ti, node] += n_positive_wins
            node = trees[ti, node, 0]
            if node == int32(-1):
                return            
        
    @staticmethod
    @cuda.jit(void(int32[:, :, :], int32[:, :], int32[:, :], int16[:], int64[:], int64[:], int64[:]))
    def _reduce_over_trees(trees, trees_ns, trees_ns_wins, root_actions_expanded, root_ns, actions_ns, actions_ns_wins):
        shared_root_ns = cuda.shared.array(512, dtype=int32) # 512 - assumed max of n_trees
        shared_actions_ns = cuda.shared.array(512, dtype=int32)
        shared_actions_ns_wins = cuda.shared.array(512, dtype=int32)
        b = cuda.blockIdx.x
        action = root_actions_expanded[cuda.blockIdx.x] # action index
        n_trees = trees.shape[0]
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x # thread index == tree index
        if t < n_trees:
            shared_root_ns[t] = trees_ns[t, 0]
            action_node = trees[t, 0, 1 + action]
            shared_actions_ns[t] = trees_ns[t, action_node]
            shared_actions_ns_wins[t] = trees_ns_wins[t, action_node]
        else:
            shared_root_ns[t] = int32(0)
            shared_actions_ns[t] = int32(0)
            shared_actions_ns_wins[t] = int32(0)
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # max-argmax reduction pattern
            if t < stride:
                t_stride = t + stride
                shared_root_ns[t] += shared_root_ns[t_stride]
                shared_actions_ns[t] += shared_actions_ns[t_stride]
                shared_actions_ns_wins[t] += shared_actions_ns_wins[t_stride]    
            cuda.syncthreads()
            stride >>= 1
        if t == 0:
            root_ns[b] = shared_root_ns[0]
            actions_ns[b] = shared_actions_ns[0]
            actions_ns_wins[b] = shared_actions_ns_wins[0]
            
    @staticmethod
    @cuda.jit(void(int64[:], int64[:], float32[:], int16[:]))
    def _reduce_over_actions(actions_ns, actions_ns_wins, best_score, best_action):
        shared_actions_ns = cuda.shared.array(512, dtype=int32) # 512 - assumed max n_actions
        shared_actions_ns_wins = cuda.shared.array(512, dtype=int32)
        shared_best_action = cuda.shared.array(512, dtype=int16)
        tpb = cuda.blockDim.x
        n_root_actions = actions_ns.size
        a = cuda.threadIdx.x # action index
        if a < n_root_actions:
            shared_actions_ns[a] = actions_ns[a]
            shared_actions_ns_wins[a] = actions_ns_wins[a]            
        else:
            shared_actions_ns[a] = int64(-1)
            shared_actions_ns_wins[a] = int64(-1)
        shared_best_action[a] = a
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # max-argmax reduction pattern
            if a < stride:
                a_stride = a + stride
                if (shared_actions_ns[a] < shared_actions_ns[a_stride]) or ((shared_actions_ns[a] == shared_actions_ns[a_stride]) and (shared_actions_ns_wins[a] < shared_actions_ns_wins[a_stride])):
                    shared_actions_ns[a] = shared_actions_ns[a_stride]
                    shared_actions_ns_wins[a] = shared_actions_ns_wins[a_stride]
                    shared_best_action[a] = shared_best_action[a_stride]     
            cuda.syncthreads()
            stride >>= 1
        if a == 0:
            best_score[0] = shared_actions_ns[0]
            best_action[0] = shared_best_action[0]
