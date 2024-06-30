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

__version__ = "1.0.0"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl" 

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


class MCTSCuda:
    
    KINDS = ["ocp", "acp", "acp_prodigal"] # one child playouts, all children playouts, all children playouts (with prodigal cuda blocks) 
    
    DEFAULT_N_TREES = 8
    DEFAULT_N_PLAYOUTS = 256
    DEFAULT_KIND = KINDS[-1]    
    DEFAULT_SEARCH_TIME_LIMIT = 5.0 # [s], np.inf possible
    DEFAULT_SEARCH_STEPS_LIMIT = np.inf # np.inf possible
    DEFAULT_UCB1_C = 2.0
    DEFAULT_DEVICE_MEMORY = 2.0 # [GB] 
    DEFAULT_SEED = 0 
    DEFAULT_VERBOSE_DEBUG = False
    DEFAULT_VERBOSE_INFO = True

    MAX_STATE_BOARD_SHAPE = (32, 32)
    MAX_STATE_EXTRA_INFO_MEMORY = 4096
    MAX_STATE_MAX_ACTIONS = 512            
    MAX_TREE_SIZE = 2**24
    MAX_N_TREES = 512    
    MAX_N_PLAYOUTS = 512        
    MAX_TREE_DEPTH = 2048 # to memorize paths at select stage          
        
    def __init__(self, state_board_shape, state_extra_info_memory, state_max_actions, 
                 n_trees=DEFAULT_N_TREES, n_playouts=DEFAULT_N_PLAYOUTS, kind=DEFAULT_KIND, 
                 search_time_limit=DEFAULT_SEARCH_TIME_LIMIT, search_steps_limit=DEFAULT_SEARCH_STEPS_LIMIT, ucb1_c=DEFAULT_UCB1_C, 
                 device_memory=DEFAULT_DEVICE_MEMORY, seed=DEFAULT_SEED,
                 verbose_debug=DEFAULT_VERBOSE_DEBUG, verbose_info=DEFAULT_VERBOSE_INFO,
                 action_to_name_function=None):
        self._set_cuda_constants()
        if not self.cuda_available:
            sys.exit(f"[MCTSCuda.__init__(): exiting due to cuda computations not available]")        
        self.state_board_shape = state_board_shape
        if self.state_board_shape[0] > self.MAX_STATE_BOARD_SHAPE[0] or self.state_board_shape[1] > self.MAX_STATE_BOARD_SHAPE[1]:
            sys.exit(f"[MCTSCuda.__init__(): exiting due to allowed state board shape exceeded]")            
        self.state_extra_info_memory = max(state_extra_info_memory, 1)
        if self.state_extra_info_memory > self.MAX_STATE_EXTRA_INFO_MEMORY:
            sys.exit(f"[MCTSCuda.__init__(): exiting due to allowed state extra info memory exceeded]")        
        self.state_max_actions = state_max_actions
        if self.state_max_actions > self.MAX_STATE_MAX_ACTIONS:
            sys.exit(f"[MCTSCuda.__init__(): exiting due to allowed state max actions memory exceeded]")        
        if self.state_max_actions > self.cuda_tpb_default:
            sys.exit(f"[MCTSCuda.__init__(): exiting due to state max actions exceeding half of cuda default tpb]")            
        self.n_trees = n_trees
        self._validate_param("n_trees", int, False, 1, False, self.MAX_N_TREES, self.DEFAULT_N_TREES)
        self.n_playouts = min(n_playouts, self.MAX_N_PLAYOUTS)
        self._validate_param("n_playouts", int, False, 1, False, self.MAX_N_PLAYOUTS, self.DEFAULT_N_PLAYOUTS)
        self.kind = kind
        if not self.kind in self.KINDS:
            invalid_kind = kind
            kind = self.DEFAULT_KIND
            print(f"[invalid kind: '{invalid_kind}' changed to '{kind}'; possible kinds: {self.KINDS}]")
        self.search_time_limit = search_time_limit
        self._validate_param("search_time_limit", float, True, 0.0, False, np.inf, self.DEFAULT_SEARCH_TIME_LIMIT)
        self.search_steps_limit = float(search_steps_limit)        
        self._validate_param("search_steps_limit", float, True, 0.0, False, np.inf, self.DEFAULT_SEARCH_STEPS_LIMIT) # purposely float, so that np.inf possible
        self.ucb1_c = ucb1_c
        self._validate_param("ucb1_c", float, False, 0.0, False, np.inf, self.DEFAULT_UCB1_C)
        self.device_memory = device_memory * 1024**3 # to bytes
        self._validate_param("device_memory", float, True, 0.0, False, np.inf, self.DEFAULT_DEVICE_MEMORY)    
        self.seed = seed
        self.verbose_debug = verbose_debug 
        self._validate_param("verbose_debug", bool, False, False, False, True, self.DEFAULT_VERBOSE_DEBUG)
        self.verbose_info = verbose_info 
        self._validate_param("verbose_info", bool, False, False, False, True, self.DEFAULT_VERBOSE_INFO)        
        self.action_to_name_function = action_to_name_function
        eps = 0.01 + 0.5 * (self.n_trees / self.MAX_N_TREES + self.state_max_actions / self.MAX_STATE_MAX_ACTIONS) * 0.04                                                                         
        self.search_time_limit_minus_eps = self.search_time_limit * (1.0 - eps)
    
    def _set_cuda_constants(self):    
        self.cuda_available = cuda.is_available() 
        self.cuda_tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2 if self.cuda_available else None
    
    def _validate_param(self, name, ptype, leq, low, geq, high, default):
        value = getattr(self, name)
        invalid = value <= low if leq else value < low
        if not invalid:
            invalid = value >= high if geq else value > high
        if not invalid:
            invalid = not isinstance(value, ptype)
        if invalid:
            low_str = str(low)
            high_str = str(high)
            correct_range_str = ("(" if leq else "[") + f"{low_str}, {high_str}" + (")" if geq else "]")
            setattr(self, name, default)
            print(f"[invalid param {name}: {value} changed to default: {default}; correct range: {correct_range_str}, correct type: {ptype}]")
        
    def init_device_side_arrays(self):
        if self.verbose_info:
            print(f"[MCTSCuda._init_device_side_arrays()... for {self}]")
        t1_dev_arrays = time.time()
        # dtypes 
        node_index_dtype = np.int32
        node_index_bytes = node_index_dtype().itemsize # 4 B
        action_index_dtype = np.int16
        action_index_bytes = action_index_dtype().itemsize # 2 B
        board_element_dtype = np.int8
        board_element_bytes = board_element_dtype().itemsize # 1 B
        extra_info_element_dtype = np.int8
        extra_info_element_bytes = extra_info_element_dtype().itemsize # 1 B
        depth_dtype = np.int16
        depth_bytes = depth_dtype().itemsize # 2 B
        size_dtype = np.int32
        size_bytes = size_dtype().itemsize # 4 B        
        turn_dtype = np.int8
        turn_bytes = turn_dtype().itemsize # 1 B                                
        flag_dtype = bool
        flag_bytes = 1 # 1 B
        outcome_dtype = np.int8
        outcome_bytes = outcome_dtype().itemsize # 1 B
        playout_outcomes_dtype = np.int32
        playout_outcomes_bytes = playout_outcomes_dtype().itemsize # 4 B                
        ns_dtype = np.int32
        ns_bytes = ns_dtype().itemsize # 4 B
        ns_extended_dtype = np.int64        
        # memory related calculations        
        per_state_additional_memory = depth_bytes + turn_bytes + 2 * flag_bytes + outcome_bytes + 2 * ns_bytes # depth, turn, leaf, terminal, ouctome, ns, ns_wins
        per_tree_additional_memory = size_bytes + node_index_bytes + action_index_bytes * (self.state_max_actions + 2) + playout_outcomes_bytes * 2 \
                                        + node_index_bytes * (self.MAX_TREE_DEPTH + 2) # tree size, tree node selected, tree actions expanded * (self.state_max_actions + 2), playout outcomes * 2, selected path          
        if "pac" in self.kind: # playout all children
            per_tree_additional_memory += playout_outcomes_bytes * self.state_max_actions * 2  # playout children outcomes            
        per_state_memory = board_element_bytes * np.prod(self.state_board_shape) + extra_info_element_bytes * self.state_extra_info_memory \
                            + node_index_bytes * (1 + self.state_max_actions) + per_state_additional_memory # board, extra info, tree array entry (parent, children nodes), additional memory
        self.max_tree_size = (int(self.device_memory) - self.n_trees * per_tree_additional_memory) // (per_state_memory * self.n_trees)
        self.max_tree_size = min(self.max_tree_size, self.MAX_TREE_SIZE)
        # tpb 
        board_tpb = int(2**np.ceil(np.log2(np.prod(self.state_board_shape))))
        extra_info_tbp = int(2**np.ceil(np.log2(self.state_extra_info_memory))) if self.state_extra_info_memory > 0 else 1
        max_actions_tpb = int(2**np.ceil(np.log2(self.state_max_actions)))        
        self.tpb_reset = min(max(board_tpb, extra_info_tbp), self.cuda_tpb_default)
        self.tpb_select = self.cuda_tpb_default
        self.tpb_expand_stage1 = min(max(self.tpb_reset, max_actions_tpb), self.cuda_tpb_default)        
        self.tpb_expand_stage2 = self.tpb_reset                                                    
        self.tpb_backup = self.cuda_tpb_default
        self.tbp_reduce_over_actions = min(max_actions_tpb, self.cuda_tpb_default)
        # device arrays
        self.dev_trees = cuda.device_array((self.n_trees, self.max_tree_size, 1 + self.state_max_actions), dtype=node_index_dtype) # each row of a tree represents a node consisting of: parent indexes and indexes of all children (associated with actions), -1 index for none parent or child 
        self.dev_trees_sizes = cuda.device_array(self.n_trees, dtype=size_dtype)
        self.dev_trees_depths = cuda.device_array((self.n_trees, self.max_tree_size), dtype=depth_dtype)
        self.dev_trees_turns = cuda.device_array((self.n_trees, self.max_tree_size), dtype=turn_dtype)
        self.dev_trees_leaves = cuda.device_array((self.n_trees, self.max_tree_size), dtype=flag_dtype)
        self.dev_trees_terminals = cuda.device_array((self.n_trees, self.max_tree_size), dtype=flag_dtype)
        self.dev_trees_outcomes = cuda.device_array((self.n_trees, self.max_tree_size), dtype=outcome_dtype)        
        self.dev_trees_ns = cuda.device_array((self.n_trees, self.max_tree_size), dtype=ns_dtype)
        self.dev_trees_ns_wins = cuda.device_array((self.n_trees, self.max_tree_size), dtype=ns_dtype)
        self.dev_trees_boards = cuda.device_array((self.n_trees, self.max_tree_size, self.state_board_shape[0], self.state_board_shape[1]), dtype=board_element_dtype)
        self.dev_trees_extra_infos = cuda.device_array((self.n_trees, self.max_tree_size, self.state_extra_info_memory), dtype=extra_info_element_dtype)
        self.dev_trees_nodes_selected = cuda.device_array(self.n_trees, dtype=node_index_dtype)
        self.dev_trees_selected_paths = cuda.device_array((self.n_trees, self.MAX_TREE_DEPTH + 2), dtype=node_index_dtype)
        self.dev_trees_actions_expanded = cuda.device_array((self.n_trees, self.state_max_actions + 2), dtype=action_index_dtype) # +2 because 2 last entries inform about: child picked randomly for playouts, number of actions (children) expanded            
        self.dev_trees_playout_outcomes = cuda.device_array((self.n_trees, 2), dtype=playout_outcomes_dtype) # each row stores counts of: -1 wins and +1 wins, respectively (for given tree) 
        self.dev_trees_playout_outcomes_children = None
        self.dev_random_generators_expand_stage1 = None         
        self.dev_random_generators_playout = None
        if "ocp" in self.kind:
            self.dev_random_generators_expand_stage1 = create_xoroshiro128p_states(self.n_trees * self.tpb_expand_stage1, seed=self.seed)
            self.dev_random_generators_playout = create_xoroshiro128p_states(self.n_trees * self.n_playouts, seed=self.seed)
        else: # "acp"
            self.dev_random_generators_playout = create_xoroshiro128p_states(self.n_trees * self.state_max_actions * self.n_playouts, seed=self.seed)                    
            self.dev_trees_playout_outcomes_children = cuda.device_array((self.n_trees, self.state_max_actions, 2), dtype=playout_outcomes_dtype) # for each (playable) action, each row stores counts of: -1 wins and +1 wins, respectively (for given tree)
        self.dev_root_actions_expanded = cuda.device_array(self.state_max_actions + 2, dtype=action_index_dtype)                    
        self.dev_root_ns = cuda.device_array(self.state_max_actions, dtype=ns_extended_dtype)
        self.dev_actions_ns = cuda.device_array(self.state_max_actions, dtype=ns_extended_dtype)
        self.dev_actions_ns_wins = cuda.device_array(self.state_max_actions, dtype=ns_extended_dtype)                 
        t2_dev_arrays = time.time()
        if self.verbose_info:
            print(f"[MCTSCuda._init_device_side_arrays() done; time: {t2_dev_arrays - t1_dev_arrays} s, per_state_memory: {per_state_memory} B,  calculated max_tree_size: {self.max_tree_size}]")

    def __str__(self):         
        return f"MCTSCuda(n_trees={self.n_trees}, n_playouts={self.n_playouts}, kind='{self.kind}', search_time_limit={self.search_time_limit}, search_steps_limit={self.search_steps_limit}, ucb1_c={self.ucb1_c}, seed: {self.seed}, device_memory={np.round(self.device_memory / 1024**3, 2)})"
        
    def __repr__(self):
        repr_str = f"{str(self)}, "
        repr_str += f"state_board_shape={self.state_board_shape}, state_extra_info_memory={self.state_extra_info_memory}, state_max_actions={self.state_max_actions})"
        return repr_str
    
    def _actions_printout(self, root_ns, actions_ns, actions_ns_wins, qs, ucb1s):
        print("[action values:")
        for i in range(qs.size):
            if qs[i] < 0.0:
                continue                     
            action_label = f" action: {i}, "
            if self.action_to_name_function:
                action_label += f"name: {self.action_to_name_function(i)}, "
                action_label += f"root_n: {root_ns[i]}, n: {actions_ns[i]}, n_wins: {actions_ns_wins[i]}, q: {qs[i]}, ucb1: {ucb1s[i]};"
                if i == self.state_max_actions - 1:
                    action_label += "]"                
                print(action_label) 
    
    def run(self, root_board, root_extra_info, root_turn):
        print(f"MCTS_CUDA RUN... [{self}]")
        t1 = time.time()        
        run_method = getattr(self, "_run_" + self.kind)
        run_method(root_board, root_extra_info, root_turn)
        t2 = time.time()
        best_action_label = str(self.best_action)
        if self.action_to_name_function is not None:
            best_action_label += f" ({self.action_to_name_function(self.best_action)})"
        print(f"MCTS_CUDA RUN DONE. [time: {t2 - t1} s; best action: {best_action_label}, best score: {self.best_score}, best q: {self.best_q}]")
        return self.best_action
    
    def _run_ocp(self, root_board, root_extra_info, root_turn):
        # MCTS reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self.tpb_reset
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.verbose_debug:
            print(f"[MCTSCuda._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSCuda._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                  self.dev_trees, self.dev_trees_sizes, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                  self.dev_trees_boards, self.dev_trees_extra_infos)
        cuda.synchronize()    
        t2_reset = time.time()
        if self.verbose_debug:
            print(f"[MCTSCuda._reset() done; time: {t2_reset - t1_reset} s]")
            
        total_time_select = 0.0
        total_time_expand = 0.0
        total_time_playout = 0.0
        total_time_backup = 0.0    
        step = 0        
        trees_actions_expanded = np.empty((self.n_trees, self.state_max_actions + 2), dtype=np.int16) # needed at host side for non-prodigal kinds
        
        t1_loop = time.time()
        while True:
            t2_loop = time.time()            
            if step >= self.search_steps_limit or t2_loop - t1_loop >= self.search_time_limit_minus_eps:
                break
            if self.verbose_debug:
                print(f"[step: {step + 1} starting, time used so far: {t2_loop - t1_loop} s]")     
            
            # MCTS select
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self.tpb_select
            if self.verbose_debug:
                print(f"[MCTSCuda._select()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._select[bpg, tpb](self.ucb1_c, 
                                       self.dev_trees, self.dev_trees_leaves, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                       self.dev_trees_nodes_selected, self.dev_trees_selected_paths)
            cuda.synchronize()
            t2_select = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._select() done; time: {t2_select - t1_select} s]")
            total_time_select += t2_select - t1_select
            
            # MCTS expand            
            t1_expand = time.time()
            t1_expand_stage1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_expand_stage1
            if self.verbose_debug:
                print(f"[MCTSCuda._expand_stage1_ocp()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSCuda._expand_stage1_ocp[bpg, tpb](self.max_tree_size, 
                                                  self.dev_trees, self.dev_trees_sizes, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals,
                                                  self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                  self.dev_trees_nodes_selected, self.dev_random_generators_expand_stage1, self.dev_trees_actions_expanded)                                                    
            self.dev_trees_actions_expanded.copy_to_host(ary=trees_actions_expanded)
            cuda.synchronize()
            if step == 0:                
                MCTSCuda._memorize_root_actions_expanded[1, self.state_max_actions + 2](self.dev_trees_actions_expanded, self.dev_root_actions_expanded)                            
                cuda.synchronize()
            t2_expand_stage1 = time.time()            
            if self.verbose_debug:
                print(f"[MCTSCuda._expand_stage1_ocp() done; time: {t2_expand_stage1 - t1_expand_stage1} s]")
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
            tpb = self.tpb_expand_stage2
            if self.verbose_debug:
                print(f"[MCTSCuda._expand_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            dev_trees_actions_expanded_flat = cuda.to_device(trees_actions_expanded_flat)
            MCTSCuda._expand_stage2[bpg, tpb](self.dev_trees, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_outcomes, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                              self.dev_trees_boards, self.dev_trees_extra_infos,                                               
                                              self.dev_trees_nodes_selected, dev_trees_actions_expanded_flat)
            cuda.synchronize()
            t2_expand_stage2 = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._expand_stage2() done; time: {t2_expand_stage2 - t1_expand_stage2} s]")
            t2_expand = time.time()
            total_time_expand += t2_expand - t1_expand
            
            # MCTS playout
            t1_playout = time.time()
            bpg = self.n_trees
            tpb = self.n_playouts
            if self.verbose_debug:
                print(f"[MCTSCuda._playout_ocp()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._playout_ocp[bpg, tpb](self.dev_trees, self.dev_trees_turns, self.dev_trees_terminals, self.dev_trees_outcomes, 
                                            self.dev_trees_boards, self.dev_trees_extra_infos, 
                                            self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, 
                                            self.dev_random_generators_playout, self.dev_trees_playout_outcomes)
            cuda.synchronize()
            t2_playout = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._playout_ocp() done; time: {t2_playout - t1_playout} s]")
            total_time_playout += t2_playout - t1_playout
            
            # MCTS backup
            t1_backup = time.time()
            # tpb = self.tpb_backup
            # bpg = (self.n_trees + tpb - 1) // tpb         
            # if self.verbose_debug:
            #     print(f"[MCTSCuda._backup()...; bpg: {bpg}, tpb: {tpb}]")
            # MCTSCuda._backup_ocp_OLD_SINGLE_THREADED[bpg, tpb](self.n_playouts,
            #                                                    self.dev_trees, self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
            #                                                    self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes)            
            bpg = self.n_trees            
            tpb = self.tpb_backup                     
            if self.verbose_debug:
                print(f"[MCTSCuda._backup_ocp()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_ocp[bpg, tpb](self.n_playouts,
                                           self.dev_trees, self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                           self.dev_trees_nodes_selected, self.dev_trees_selected_paths, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes)                                
            cuda.synchronize()            
            t2_backup = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._backup() done; time: {t2_backup - t1_backup} s]")
            total_time_backup += t2_backup - t1_backup                                        
            step += 1
            
        # MCTS sum reduction over trees for each root action        
        t1_reduce_over_trees = time.time()
        root_actions_expanded = np.empty_like(self.dev_root_actions_expanded)
        self.dev_root_actions_expanded.copy_to_host(ary=root_actions_expanded)
        n_root_actions = int(root_actions_expanded[-1]) 
        bpg = n_root_actions
        tpb = int(2**np.ceil(np.log2(self.n_trees)))
        if self.verbose_debug:
            print(f"[MCTSCuda._reduce_over_trees()...; bpg: {bpg}, tpb: {tpb}]")
        MCTSCuda._reduce_over_trees[bpg, tpb](self.dev_trees, self.dev_trees_ns, self.dev_trees_ns_wins, self.dev_root_actions_expanded, self.dev_root_ns, self.dev_actions_ns, self.dev_actions_ns_wins)
        cuda.synchronize()
        t2_reduce_over_trees = time.time()
        if self.verbose_debug:
            print(f"[MCTSCuda._reduce_over_trees() done; time: {t2_reduce_over_trees - t1_reduce_over_trees} s]")
            
        # MCTS sum reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self.tbp_reduce_over_actions
        if self.verbose_debug:
            print(f"[MCTSCuda._reduce_over_actions()...; bpg: {bpg}, tpb: {tpb}]")        
        dev_best_score = cuda.device_array(1, dtype=np.float32)
        dev_best_q = cuda.device_array(1, dtype=np.float32)
        dev_best_action = cuda.device_array(1, dtype=np.int16)
        MCTSCuda._reduce_over_actions[bpg, tpb](n_root_actions, self.dev_actions_ns, self.dev_actions_ns_wins, dev_best_score, dev_best_action)                 
        self.best_score = dev_best_score.copy_to_host()[0]
        self.best_q = dev_best_q.copy_to_host()[0]
        self.best_action = dev_best_action.copy_to_host()[0]
        self.best_action = root_actions_expanded[self.best_action]        
        cuda.synchronize()        
        t2_reduce_over_actions = time.time()
        if self.verbose_debug:
            print(f"[MCTSCuda._reduce_over_actions() done; time: {t2_reduce_over_actions - t1_reduce_over_actions} s]")                
         
        if self.verbose_info:
            # actions printout
            root_ns_packed = np.empty(self.state_max_actions, dtype=np.int64)
            actions_ns_packed = np.empty(self.state_max_actions, dtype=np.int64)
            actions_ns_wins_packed = np.empty(self.state_max_actions, dtype=np.int64)       
            self.dev_root_ns.copy_to_host(ary=root_ns_packed)
            self.dev_actions_ns.copy_to_host(ary=actions_ns_packed)
            self.dev_actions_ns_wins.copy_to_host(ary=actions_ns_wins_packed)
            root_ns = np.zeros(self.state_max_actions, dtype=np.int64)
            actions_ns = np.zeros(self.state_max_actions, dtype=np.int64)
            actions_ns_wins = np.zeros(self.state_max_actions, dtype=np.int64)                        
            qs = -np.ones(self.state_max_actions, dtype=np.float32)
            ucb1s = -np.ones(self.state_max_actions, dtype=np.float32)
            for i in range(n_root_actions):
                a = root_actions_expanded[i]
                root_ns[a] = root_ns_packed[i]
                actions_ns[a] = actions_ns_packed[i]
                actions_ns_wins[a] = actions_ns_wins_packed[i]
                qs[a] = actions_ns_wins[a] / actions_ns[a] if actions_ns[a] > 0 else np.nan                          
                ucb1s[a] = qs[a] + self.ucb1_c * np.sqrt(np.log(root_ns[a]) / actions_ns[a]) if actions_ns[a] > 0 else np.nan            
            self._actions_printout(root_ns, actions_ns, actions_ns_wins, qs, ucb1s)
            # performance and tree info printout 
            depths = self.dev_trees_depths.copy_to_host()
            sizes = self.dev_trees_sizes.copy_to_host()
            cuda.synchronize()
            i_sizes = np.zeros(self.n_trees, dtype=np.int32)
            max_depth = -1        
            for i in range(self.n_trees):            
                i_depth = np.max(depths[i, :sizes[i]])
                i_sizes[i] = sizes[i]
                max_depth = max(i_depth, max_depth)
            print(f"[steps performed: {step}]")
            print(f"[trees info -> max depth: {max_depth}, max size: {np.max(i_sizes)}, mean size: {np.mean(i_sizes)}]")
            ms_factor = 10.0**3                    
            print(f"[loop time [ms] -> total: {ms_factor * (t2_loop - t1_loop)}, mean: {ms_factor * (t2_loop - t1_loop) / step}]")
            print(f"[mean times of stages [ms] -> selection: {ms_factor * total_time_select / step}, expansion: {ms_factor * total_time_expand / step}, playout: {ms_factor * total_time_playout / step}, backup: {ms_factor * total_time_backup / step}]")
            best_action_label = str(self.best_action)
            if self.action_to_name_function:
                best_action_label += f" ({self.action_to_name_function(self.best_action)})"             
            print(f"[best action: {best_action_label}, best score: {self.best_score}, best q: {self.best_q}]")                                        
        
        return self.best_action

    def _run_acp(self, root_board, root_extra_info, root_turn):
        # MCTS reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self.tpb_reset
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array        
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.verbose_debug:
            print(f"[MCTSCuda._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSCuda._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                  self.dev_trees, self.dev_trees_sizes, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                  self.dev_trees_boards, self.dev_trees_extra_infos)
        cuda.synchronize()    
        t2_reset = time.time()
        if self.verbose_debug:
            print(f"[MCTSCuda._reset() done; time: {t2_reset - t1_reset} s]")
            
        total_time_select = 0.0
        total_time_expand = 0.0        
        total_time_playout = 0.0
        total_time_backup = 0.0        
        step = 0        
        trees_actions_expanded = np.empty((self.n_trees, self.state_max_actions + 2), dtype=np.int16)
        
        t1_loop = time.time()
        while True:
            t2_loop = time.time()
            if step >= self.search_steps_limit or t2_loop - t1_loop >= self.search_time_limit:
                break
            if self.verbose_debug:
                print(f"[step: {step + 1} starting, time used so far: {t2_loop - t1_loop} s]")     
            
            # MCTS select
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self.tpb_select
            if self.verbose_debug:
                print(f"[MCTSCuda._select()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._select[bpg, tpb](self.ucb1_c, 
                                       self.dev_trees, self.dev_trees_leaves, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                       self.dev_trees_nodes_selected, self.dev_trees_selected_paths)
            cuda.synchronize()                     
            t2_select = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._select() done; time: {t2_select - t1_select} s]")
            total_time_select += t2_select - t1_select
                                        
            # MCTS expand                        
            t1_expand_stage1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_expand_stage1
            if self.verbose_debug:
                print(f"[MCTSCuda._expand_stage1_acp()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSCuda._expand_stage1_acp[bpg, tpb](self.max_tree_size, 
                                                  self.dev_trees, self.dev_trees_sizes, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals,
                                                  self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                  self.dev_trees_nodes_selected, self.dev_trees_actions_expanded)                                             
            self.dev_trees_actions_expanded.copy_to_host(ary=trees_actions_expanded)
            cuda.synchronize()
            if step == 0:
                root_actions_expanded = np.copy(trees_actions_expanded[0])                                                                 
            actions_expanded_cumsum = np.cumsum(trees_actions_expanded[:, -1])
            trees_actions_expanded_flat = np.empty((actions_expanded_cumsum[-1], 2), dtype=np.int16)
            shift = 0
            for ti in range(self.n_trees):
                s = slice(shift, actions_expanded_cumsum[ti])
                trees_actions_expanded_flat[s, 0] = ti
                trees_actions_expanded_flat[s, 1] = trees_actions_expanded[ti, :trees_actions_expanded[ti, -1]]
                shift = actions_expanded_cumsum[ti]
            t2_expand_stage1 = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_acpo_stage1() done; time: {t2_expand_stage1 - t1_expand_stage1} s]")
            total_time_expand += t2_expand_stage1 - t1_expand_stage1                                             
            t1_expand_stage2 = time.time()
            bpg = actions_expanded_cumsum[-1]            
            tpb = self.tpb_expand_stage2
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._expand_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            dev_trees_actions_expanded_flat = cuda.to_device(trees_actions_expanded_flat)
            MCTSCuda._expand_stage2[bpg, tpb](self.dev_trees, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_outcomes, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                              self.dev_trees_boards, self.dev_trees_extra_infos,                                               
                                              self.dev_trees_nodes_selected, dev_trees_actions_expanded_flat)
            cuda.synchronize()
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
            MCTSCuda._playout_acpo[bpg, tpb](self.dev_trees, self.dev_trees_turns, self.dev_trees_terminals, self.dev_trees_outcomes, 
                                             self.dev_trees_boards, self.dev_trees_extra_infos, 
                                             self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, dev_trees_actions_expanded_flat,
                                             self.dev_random_generators_playout, self.dev_trees_playout_outcomes, self.dev_trees_playout_outcomes_children)
            cuda.synchronize()
            t2_playout = time.time()
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._playout_acpo() done; time: {t2_playout - t1_playout} s]")
            total_time_playout += t2_playout - t1_playout
            # MCTS backup
            t1_backup = time.time()
            t1_backup_stage1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_backup                     
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage1()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_acpo_stage1[bpg, tpb](self.n_playouts, 
                                                   self.dev_trees, self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                   self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes, self.dev_trees_playout_outcomes_children)
            cuda.synchronize()            
            t2_backup_stage1 = time.time()
            total_time_backup_1 += t2_backup_stage1 - t1_backup_stage1
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage1() done; time: {t2_backup_stage1 - t1_backup_stage1} s]")            
            t1_backup_stage2 = time.time()            
            # tpb = self.tpb_backup 
            # bpg = (self.n_trees + tpb) // tpb         
            # if self.VERBOSE_DEBUG:
            #     print(f"[MCTSCuda._backup_acpo_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            # MCTSCuda._backup_acpo_stage2[bpg, tpb](self.n_playouts,
            #                                        self.dev_trees, self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
            #                                        self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes)
            tpb = self.tpb_backup
            bpg = self.n_trees         
            if self.VERBOSE_DEBUG:
                print(f"[MCTSCuda._backup_acpo_stage2()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_acpo_stage2[bpg, tpb](self.n_playouts,
                                                   self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                   self.dev_trees_selected_paths, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes)
            cuda.synchronize()                                    
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
        MCTSCuda._reduce_over_trees[bpg, tpb](self.dev_trees, self.dev_trees_ns, self.dev_trees_ns_wins, dev_root_actions_expanded, dev_root_ns, dev_actions_ns, dev_actions_ns_wins)
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
        if self.verbose_info:
            print("[action values:")
        for i in range(n_root_actions):
            q = actions_ns_wins[i] / actions_ns[i] if actions_ns[i] > 0 else np.nan
            ucb1 = q + self.ucb1_c * np.sqrt(np.log(root_ns[i]) / actions_ns[i]) if actions_ns[i] > 0 else np.nan
            qs[root_actions_expanded[i]] = q
            if self.verbose_info:
                action_label = f"action: {root_actions_expanded[i]}, "
                if self.action_to_name_function:
                    action_label += f"name: {self.action_to_name_function(root_actions_expanded[i])}, "
                action_label += f"root_n: {root_ns[i]}, n: {actions_ns[i]}, n_wins: {actions_ns_wins[i]}, q: {q}, ucb1: {ucb1}"
                print(action_label) 
        if self.verbose_info:
            print("]")                                                
        # MCTS sum reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self.tbp_reduce_over_actions
        if self.VERBOSE_DEBUG:
            print(f"[MCTSCuda._reduce_over_actions()...; bpg: {bpg}, tpb: {tpb}]")        
        dev_best_score = cuda.device_array(1, dtype=np.float32)
        dev_best_action = cuda.device_array(1, dtype=np.int16)
        MCTSCuda._reduce_over_actions[bpg, tpb](n_root_actions, dev_actions_ns, dev_actions_ns_wins, dev_best_score, dev_best_action)                 
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
        depths = self.dev_trees_depths.copy_to_host()
        sizes = self.dev_trees_sizes.copy_to_host()
        i_sizes = np.zeros(self.n_trees, dtype=np.int32)
        max_depth = -1        
        for i in range(self.n_trees):            
            i_depth = np.max(depths[i, :sizes[i]])
            i_sizes[i] = sizes[i]
            # print(f"[tree {i} -> size: {sizes[i]}, depth: {i_depth}]")
            max_depth = max(i_depth, max_depth)
        print(f"[steps performed: {step}]")
        print(f"[trees -> max depth: {max_depth}, max size: {np.max(i_sizes)}, mean size: {np.mean(i_sizes)}]")
        ms_factor = 10.0**3                    
        print(f"[loop time [ms] -> total: {ms_factor * (t2_loop - t1_loop)}, mean: {ms_factor * (t2_loop - t1_loop) / step}]")
        print(f"[mean times of stages [ms] -> selection: {ms_factor * total_time_select / step}, expansion: {ms_factor * total_time_expand / step}, playout: {ms_factor * total_time_playout / step}, backup: {ms_factor * total_time_backup / step}]") 
        print(f"[best action: {best_action}, best score: {best_score}, best q: {qs[best_action]}]")                        
        print(f"MCTS_CUDA RUN ACPO DONE. [time: {t2 - t1} s, steps: {step}]")        
        return best_action

    def _run_acp_prodigal(self, root_board, root_extra_info, root_turn):        
        # MCTS reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self.tpb_reset
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array        
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.verbose_debug:
            print(f"[MCTSCuda._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSCuda._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                  self.dev_trees, self.dev_trees_sizes, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                  self.dev_trees_boards, self.dev_trees_extra_infos)
        cuda.synchronize()    
        t2_reset = time.time()
        if self.verbose_debug:
            print(f"[MCTSCuda._reset() done; time: {t2_reset - t1_reset} s]")
        
        total_time_select = 0.0
        total_time_expand = 0.0
        total_time_playout = 0.0
        total_time_backup = 0.0
        step = 0
        
        t1_loop = time.time()
        while True:
            t2_loop = time.time()
            if step >= self.search_steps_limit or t2_loop - t1_loop >= self.search_time_limit_minus_eps:
                break
            if self.verbose_debug:
                print(f"[step: {step + 1} starting, time used so far: {t2_loop - t1_loop} s]")     
        
            # MCTS select
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self.tpb_select
            if self.verbose_debug:
                print(f"[MCTSCuda._select()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._select[bpg, tpb](self.ucb1_c, 
                                       self.dev_trees, self.dev_trees_leaves, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                       self.dev_trees_nodes_selected, self.dev_trees_selected_paths)
            cuda.synchronize()                     
            t2_select = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._select() done; time: {t2_select - t1_select} s]")
            total_time_select += t2_select - t1_select                                    
            
            # MCTS expand
            t1_expand = time.time()                        
            t1_expand_stage1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_expand_stage1
            if self.verbose_debug:
                print(f"[MCTSCuda._expand_stage1_acp_prodigal()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSCuda._expand_stage1_acp_prodigal[bpg, tpb](self.max_tree_size, 
                                                           self.dev_trees, self.dev_trees_sizes, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals,
                                                           self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                           self.dev_trees_nodes_selected, self.dev_trees_actions_expanded)                        
            cuda.synchronize()
            if step == 0:                
                MCTSCuda._memorize_root_actions_expanded[1, self.state_max_actions + 2](self.dev_trees_actions_expanded, self.dev_root_actions_expanded)                            
                cuda.synchronize()
            t2_expand_stage1 = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._expand_stage1_acp_prodigal() done; time: {t2_expand_stage1 - t1_expand_stage1} s]")                                
            t1_expand_stage2 = time.time()
            bpg = (self.n_trees, self.state_max_actions)            
            tpb = self.tpb_expand_stage2 
            if self.verbose_debug:
                print(f"[MCTSCuda._expand_stage2_acp_prodigal()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._expand_stage2_acp_prodigal[bpg, tpb](self.dev_trees, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_outcomes, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                           self.dev_trees_boards, self.dev_trees_extra_infos,                                               
                                                           self.dev_trees_nodes_selected, self.dev_trees_actions_expanded)
            cuda.synchronize()            
            t2_expand_stage2 = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._expand_stage2_acp_prodigal() done; time: {t2_expand_stage2 - t1_expand_stage2} s]")
            t2_expand = time.time()
            total_time_expand += t2_expand - t1_expand
            
            # MCTS playout
            t1_playout = time.time()
            bpg = (self.n_trees, self.state_max_actions)
            tpb = self.n_playouts
            if self.verbose_debug:
                print(f"[MCTSCuda._playout_acp_prodigal()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._playout_acp_prodigal[bpg, tpb](self.dev_trees, self.dev_trees_turns, self.dev_trees_terminals, self.dev_trees_outcomes, 
                                                     self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                     self.dev_trees_nodes_selected, self.dev_trees_actions_expanded,
                                                     self.dev_random_generators_playout, self.dev_trees_playout_outcomes, self.dev_trees_playout_outcomes_children)
            cuda.synchronize()
            t2_playout = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._playout_acp_prodigal() done; time: {t2_playout - t1_playout} s]")
            total_time_playout += t2_playout - t1_playout
            
            # MCTS backup
            t1_backup = time.time()
            t1_backup_stage1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_backup                     
            if self.verbose_debug:
                print(f"[MCTSCuda._backup_stage1_acp_prodigal()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_stage1_acp_prodigal[bpg, tpb](self.n_playouts, 
                                                           self.dev_trees, self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                           self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes, self.dev_trees_playout_outcomes_children)
            cuda.synchronize()            
            t2_backup_stage1 = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._backup_stage1_acp_prodigal() done; time: {t2_backup_stage1 - t1_backup_stage1} s]")            
            t1_backup_stage2 = time.time()
            bpg = self.n_trees            
            tpb = self.tpb_backup                     
            if self.verbose_debug:
                print(f"[MCTSCuda._backup_stage2_acp()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSCuda._backup_stage2_acp[bpg, tpb](self.n_playouts,
                                                  self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                  self.dev_trees_selected_paths, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes)
            cuda.synchronize()                                    
            t2_backup_stage2 = time.time()
            if self.verbose_debug:
                print(f"[MCTSCuda._backup_stage2_acp() done; time: {t2_backup_stage2 - t1_backup_stage2} s]")
            t2_backup = time.time()
            total_time_backup += t2_backup - t1_backup
                                                    
            step += 1
        t2_loop = time.time()
                                                        
        # MCTS sum reduction over trees
        t1_reduce_over_trees = time.time()
        bpg = self.state_max_actions
        tpb = int(2**np.ceil(np.log2(self.n_trees)))
        if self.verbose_debug:
            print(f"[MCTSCuda._reduce_over_trees_prodigal()...; bpg: {bpg}, tpb: {tpb}]")        
        MCTSCuda._reduce_over_trees_prodigal[bpg, tpb](self.dev_trees, self.dev_trees_ns, self.dev_trees_ns_wins, self.dev_root_actions_expanded, self.dev_root_ns, self.dev_actions_ns, self.dev_actions_ns_wins)
        cuda.synchronize()
        t2_reduce_over_trees = time.time()
        if self.verbose_debug:
            print(f"[MCTSCuda._reduce_over_trees_prodigal() done; time: {t2_reduce_over_trees - t1_reduce_over_trees} s]")                
                    
        # MCTS sum reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self.tbp_reduce_over_actions
        if self.verbose_debug:
            print(f"[MCTSCuda._reduce_over_actions_prodigal()...; bpg: {bpg}, tpb: {tpb}]")        
        dev_best_score = cuda.device_array(1, dtype=np.float32)
        dev_best_q = cuda.device_array(1, dtype=np.float32)
        dev_best_action = cuda.device_array(1, dtype=np.int16)
        MCTSCuda._reduce_over_actions_prodigal[bpg, tpb](self.dev_actions_ns, self.dev_actions_ns_wins, dev_best_score, dev_best_q, dev_best_action)                 
        self.best_score = dev_best_score.copy_to_host()[0]
        self.best_q = dev_best_q.copy_to_host()[0]
        self.best_action = dev_best_action.copy_to_host()[0]            
        cuda.synchronize()
        t2_reduce_over_actions = time.time()                            
        if self.verbose_debug:
            print(f"[MCTSCuda._reduce_over_actions_prodigal() done; time: {t2_reduce_over_actions - t1_reduce_over_actions} s]")                
                 
        if self.verbose_info:
            # actions printout
            root_ns = np.zeros(self.state_max_actions, dtype=np.int64)
            actions_ns = np.zeros(self.state_max_actions, dtype=np.int64)
            actions_ns_wins = np.zeros(self.state_max_actions, dtype=np.int64)                                    
            self.dev_root_ns.copy_to_host(ary=root_ns)
            self.dev_actions_ns.copy_to_host(ary=actions_ns)
            self.dev_actions_ns_wins.copy_to_host(ary=actions_ns_wins)
            qs = -np.ones(self.state_max_actions, dtype=np.float32)
            ucb1s = -np.ones(self.state_max_actions, dtype=np.float32)
            for i in range(self.state_max_actions):                
                a = i
                if root_ns[a] == 0:
                    continue
                qs[a] = actions_ns_wins[a] / actions_ns[a] if actions_ns[a] > 0 else np.nan                          
                ucb1s[a] = qs[a] + self.ucb1_c * np.sqrt(np.log(root_ns[a]) / actions_ns[a]) if actions_ns[a] > 0 else np.nan            
            self._actions_printout(root_ns, actions_ns, actions_ns_wins, qs, ucb1s)
            # performance and tree info printout                        
            depths = self.dev_trees_depths.copy_to_host()
            sizes = self.dev_trees_sizes.copy_to_host()
            cuda.synchronize()
            i_sizes = np.zeros(self.n_trees, dtype=np.int32)
            max_depth = -1        
            for i in range(self.n_trees):            
                i_depth = np.max(depths[i, :sizes[i]])
                i_sizes[i] = sizes[i]
                max_depth = max(i_depth, max_depth)
            print(f"[steps performed: {step}]")
            print(f"[trees info -> max depth: {max_depth}, max size: {np.max(i_sizes)}, mean size: {np.mean(i_sizes)}]")
            ms_factor = 10.0**3                    
            print(f"[loop time [ms] -> total: {ms_factor * (t2_loop - t1_loop)}, mean: {ms_factor * (t2_loop - t1_loop) / step}]")
            print(f"[mean times of stages [ms] -> selection: {ms_factor * total_time_select / step}, expansion: {ms_factor * total_time_expand / step}, playout: {ms_factor * total_time_playout / step}, backup: {ms_factor * total_time_backup / step}]") 
            best_action_label = str(self.best_action)
            if self.action_to_name_function:
                best_action_label += f" ({self.action_to_name_function(self.best_action)})"             
            print(f"[best action: {best_action_label}, best score: {self.best_score}, best q: {self.best_q}]")                                        
        
        return self.best_action

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
    @cuda.jit(void(float32, int32[:, :, :], boolean[:, :], int32[:, :], int32[:, :], int32[:], int32[:, :]))        
    def _select(ucb1_c, trees, trees_leaves, trees_ns, trees_ns_wins, trees_nodes_selected, trees_selected_paths):
        shared_ucb1s = cuda.shared.array(1024, dtype=float32) # 1024 - assumed limit on max actions
        shared_best_child = cuda.shared.array(1024, dtype=int32) # 1024 - assumed limit on max actions (array instead of one index due to max-argmax reduction pattern)
        shared_selected_path = cuda.shared.array(2048 + 2, dtype=int32) # 2048 - assumed equal to MAX_TREE_DEPTH + 2 
        ti = cuda.blockIdx.x # tree index 
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        state_max_actions = int16(trees.shape[2] - 1)
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
            
    @staticmethod
    @cuda.jit(void(int32, int32[:, :, :], int32[:], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], xoroshiro128p_type[:], int16[:, :]))
    def _expand_stage1_ocp(max_tree_size, trees, trees_sizes, trees_turns, trees_leaves, trees_terminals, trees_boards, trees_extra_infos, 
                           trees_nodes_selected, random_generators_expand_stage1, trees_actions_expanded):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_legal_actions = cuda.shared.array(1024, dtype=boolean) # 1024 - assumed limit on max actions
        shared_legal_actions_child_shifts = cuda.shared.array(1024, dtype=int16) # 1024 - assumed limit on max actions
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        t_global = cuda.grid(1)
        state_max_actions = int16(trees.shape[2] - 1)
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
                trees_actions_expanded[ti, -1] = child_shift + 1 # information how many children expanded (as last entry)
                if child_shift >= 0:
                    trees_leaves[ti, selected] = False                                
                rand_child_for_playout = int16(xoroshiro128p_uniform_float32(random_generators_expand_stage1, t_global) * (child_shift + 1))
                trees_actions_expanded[ti, -2] = rand_child_for_playout
            else:
                trees_actions_expanded[ti, -1] = int16(1)
                trees_actions_expanded[ti, -2] = int16(-1) # fake child for playouts indicating that selected is terminal (and playouts computed from outcome)
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
    @cuda.jit(void(int32, int32[:, :, :], int32[:], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :]))
    def _expand_stage1_acp(max_tree_size, trees, trees_sizes, trees_turns, trees_leaves, trees_terminals, trees_boards, trees_extra_infos, 
                           trees_nodes_selected, trees_actions_expanded):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_legal_actions = cuda.shared.array(1024, dtype=boolean) # 1024 - assumed limit on max actions
        shared_legal_actions_child_shifts = cuda.shared.array(1024, dtype=int16) # 1024 - assumed limit on max actions
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        state_max_actions = int16(trees.shape[2] - 1)
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
                trees_actions_expanded[ti, -1] = child_shift + 1 # information how many children expanded (as last entry)                
                trees_actions_expanded[ti, -2] = int16(-2) # indicates all children for playouts (acp)                                
            else:
                trees_actions_expanded[ti, -1] = int16(1)
                trees_actions_expanded[ti, -2] = int16(-1) # fake child for playouts indicating that selected is terminal (and playouts computed from outcome)                
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
            if selected_is_terminal:
                trees_actions_expanded[ti, 0] = int16(0) # fake legal action for terminal playout (exactly 1, any, must be legal)                    

    @staticmethod
    @cuda.jit(void(int32, int32[:, :, :], int32[:], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :]))
    def _expand_stage1_acp_prodigal(max_tree_size, trees, trees_sizes, trees_turns, trees_leaves, trees_terminals, trees_boards, trees_extra_infos, 
                                    trees_nodes_selected, trees_actions_expanded):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_legal_actions = cuda.shared.array(1024, dtype=boolean) # 1024 - assumed limit on max actions
        shared_legal_actions_child_shifts = cuda.shared.array(1024, dtype=int16) # 1024 - assumed limit on max actions
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        state_max_actions = int16(trees.shape[2] - 1)
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
                trees_actions_expanded[ti, -1] = child_shift + 1 # information how many children expanded (as last entry)                
                trees_actions_expanded[ti, -2] = int16(-2) # indicates all children for playouts (acp)                                
            else:                
                trees_actions_expanded[ti, -1] = int16(1)
                trees_actions_expanded[ti, -2] = int16(-1) # fake child for playouts indicating that selected is terminal (and playouts computed from outcome)                                 
        cuda.syncthreads()        
        if t < state_max_actions: 
            child_index = int32(-1)
            if shared_legal_actions[t]:
                child_shift = shared_legal_actions_child_shifts[t]
                child_index = size_so_far + child_shift                
                trees_actions_expanded[ti, t] = child_shift
            else: 
                trees_actions_expanded[ti, t] = int16(-1)                
            trees[ti, selected, 1 + t] = child_index # parent gets to know where child is 
        if t == 0:
            trees_sizes[ti] += shared_legal_actions_child_shifts[state_max_actions - 1] + 1 # updating tree size
            if selected_is_terminal:
                trees_actions_expanded[ti, 0] = int16(0) # fake legal action for terminal playout (exactly 1, any, must be legal)
                
    @staticmethod
    @cuda.jit(void(int16[:, :], int16[:]))
    def _memorize_root_actions_expanded(dev_trees_actions_expanded, dev_root_actions_expanded):
        t = cuda.threadIdx.x
        dev_root_actions_expanded[t] = dev_trees_actions_expanded[0, t]                
        
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
    @cuda.jit(void(int32[:, :, :], int16[:, :], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :], int32[:, :], int32[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :]))
    def _expand_stage2_acp_prodigal(trees, trees_depths, trees_turns, trees_leaves, trees_terminals, trees_outcomes, trees_ns, trees_ns_wins, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        ti = cuda.blockIdx.x
        action = cuda.blockIdx.y
        if trees_actions_expanded[ti, action] == int16(-1): # prodigality
            return
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
    def _playout_ocp(trees, trees_turns, trees_terminals, trees_outcomes, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded, random_generators_playout, trees_playout_outcomes):
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
        if rand_child_for_playout != int16(-1): # check if some child picked on random for playouts
            last_action = trees_actions_expanded[ti, rand_child_for_playout]
            selected = trees[ti, selected, 1 + int32(last_action)]
        if trees_terminals[ti, selected]: # root for playouts has been discovered terminal before (by game rules) -> taking stored outcome (multiplied by tpb)
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
        if trees_terminals[ti, selected]: # root for playouts has been discovered terminal before (by game rules) -> taking stored outcome (multiplied by tpb)
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
    @cuda.jit(void(int32[:, :, :], int8[:, :], boolean[:, :], int8[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :], xoroshiro128p_type[:], int32[:, :], int32[:, :, :]))
    def _playout_acp_prodigal(trees, trees_turns, trees_terminals, trees_outcomes, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded,  random_generators_playout, trees_playout_outcomes, 
                              trees_playout_outcomes_children):
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_playout_outcomes = cuda.shared.array((1024, 2), dtype=int16) # 1024 - assumed max tpb for playouts, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout        
        ti = cuda.blockIdx.x
        action = cuda.blockIdx.y
        if trees_actions_expanded[ti, action] == int16(-1): # prodigality
            return         
        local_board = cuda.local.array((32, 32), dtype=int8)
        local_extra_info = cuda.local.array(4096, dtype=int8)
        local_legal_actions_with_count = cuda.local.array(1024 + 1, dtype=int16) # 1024 - assumed limit on max actions          
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        selected = trees_nodes_selected[ti]
        fake_child_for_playout = trees_actions_expanded[ti, -2]
        last_action = int8(-1) # none yet
        if fake_child_for_playout != int16(-1): # check if true playouts are to be made (on all children of selected)
            last_action = action
            selected = trees[ti, selected, 1 + int32(last_action)]
        if trees_terminals[ti, selected]: # root for playouts has been discovered terminal before (by game rules) -> taking stored outcome (multiplied by tpb)
            if t == 0:
                outcome = trees_outcomes[ti, selected]
                if fake_child_for_playout != int16(-1):                
                    trees_playout_outcomes_children[ti, action, 0] = tpb if outcome == int8(-1) else int32(0) # wins of -1
                    trees_playout_outcomes_children[ti, action, 1] = tpb if outcome == int8(1) else int32(0) # wins of +1
                else:
                    trees_playout_outcomes[ti, 0] = tpb if outcome == int8(-1) else int32(0) # wins of -1
                    trees_playout_outcomes[ti, 1] = tpb if outcome == int8(1) else int32(0) # wins of +1
        else: # playouts for non-terminal
            t = cuda.threadIdx.x
            state_max_actions = trees.shape[2] - 1
            t_global = ti * state_max_actions * tpb + action * tpb + t
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
    def _backup_ocp_OLD_SINGLE_THREADED(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_actions_expanded, trees_playout_outcomes):
        ti = cuda.grid(1)
        n_trees = trees.shape[0]
        if ti >= n_trees:
            return
        node = trees_nodes_selected[ti]
        rand_child_for_playout = trees_actions_expanded[ti, -2]
        if rand_child_for_playout != int16(-1): # check if some child picked on random for playouts
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
    @cuda.jit(void(int16, int32[:, :, :], int8[:, :], int32[:, :], int32[:, :], int32[:], int32[:, :], int16[:, :], int32[:, :]))
    def _backup_ocp(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_selected_paths, trees_actions_expanded, trees_playout_outcomes):
        ti = cuda.blockIdx.x
        t = cuda.threadIdx.x
        tpb = cuda.blockDim.x
        n_negative_wins = trees_playout_outcomes[ti, 0]
        n_positive_wins = trees_playout_outcomes[ti, 1]   
        path_length = trees_selected_paths[ti, -1]
        pept = (path_length + tpb - 1) // tpb # path elements per thread
        e = t
        for _ in range(pept):
            if e < path_length:                
                node = trees_selected_paths[ti, e]
                trees_ns[ti, node] += n_playouts
                if trees_turns[ti, node] == int8(1):
                    trees_ns_wins[ti, node] += n_negative_wins 
                else:
                    trees_ns_wins[ti, node] += n_positive_wins                
            e += tpb
        if t == 0:
            node = trees_nodes_selected[ti]
            rand_child_for_playout = trees_actions_expanded[ti, -2]
            if rand_child_for_playout != int16(-1): # check if some child picked on random for playouts
                last_action = trees_actions_expanded[ti, rand_child_for_playout]
                node = trees[ti, node, 1 + int32(last_action)]
                trees_ns[ti, node] += n_playouts
                if trees_turns[ti, node] == int8(1):
                    trees_ns_wins[ti, node] += n_negative_wins 
                else:
                    trees_ns_wins[ti, node] += n_positive_wins                                                    

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
    @cuda.jit(void(int16, int32[:, :, :], int8[:, :], int32[:, :], int32[:, :], int32[:], int16[:, :], int32[:, :], int32[:, :, :]))
    def _backup_stage1_acp_prodigal(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_actions_expanded, trees_playout_outcomes, trees_playout_outcomes_children):
        shared_playout_outcomes_children = cuda.shared.array((1024, 2), dtype=int32) # 1024 - assumed max tpb for playouts, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout 
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x        
        fake_child_for_playout = trees_actions_expanded[ti, -2]
        max_actions = trees_actions_expanded.shape[1] - 2
        if fake_child_for_playout != int16(-1): # check if selected is not terminal
            selected = trees_nodes_selected[ti]
            if t < max_actions and trees_actions_expanded[ti, t] != int16(-1): # prodigality
                a = t
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
    @cuda.jit(void(int16, int8[:, :], int32[:, :], int32[:, :], int32[:, :], int16[:, :], int32[:, :]))
    def _backup_stage2_acp(n_playouts, trees_turns, trees_ns, trees_ns_wins, trees_selected_paths, trees_actions_expanded, trees_playout_outcomes):
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
    def _backup_stage2_acp_OLD_SINGLE_THREADED(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_actions_expanded, trees_playout_outcomes):
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
        shared_root_ns = cuda.shared.array(512, dtype=int64) # 512 - assumed max of n_trees
        shared_actions_ns = cuda.shared.array(512, dtype=int64)
        shared_actions_ns_wins = cuda.shared.array(512, dtype=int64)
        b = cuda.blockIdx.x
        action = root_actions_expanded[b] # action index
        n_trees = trees.shape[0]
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x # thread index == tree index
        if t < n_trees:
            shared_root_ns[t] = int64(trees_ns[t, 0])
            action_node = trees[t, 0, 1 + action]
            shared_actions_ns[t] = int64(trees_ns[t, action_node])
            shared_actions_ns_wins[t] = int64(trees_ns_wins[t, action_node])
        else:
            shared_root_ns[t] = int64(0)
            shared_actions_ns[t] = int64(0)
            shared_actions_ns_wins[t] = int64(0)
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
    @cuda.jit(void(int32[:, :, :], int32[:, :], int32[:, :], int16[:], int64[:], int64[:], int64[:]))
    def _reduce_over_trees_prodigal(trees, trees_ns, trees_ns_wins, root_actions_expanded, root_ns, actions_ns, actions_ns_wins):
        shared_root_ns = cuda.shared.array(512, dtype=int64) # 512 - assumed max of n_trees
        shared_actions_ns = cuda.shared.array(512, dtype=int64)
        shared_actions_ns_wins = cuda.shared.array(512, dtype=int64)
        b = cuda.blockIdx.x
        action = b
        t = cuda.threadIdx.x # thread index == tree index
        shared_root_ns[t] = int64(0)
        shared_actions_ns[t] = int64(0)
        shared_actions_ns_wins[t] = int64(0)        
        if root_actions_expanded[action] != int16(-1): 
            n_trees = trees.shape[0]
            tpb = cuda.blockDim.x            
            if t < n_trees:
                shared_root_ns[t] = trees_ns[t, 0]
                action_node = trees[t, 0, 1 + action]
                shared_actions_ns[t] = trees_ns[t, action_node]
                shared_actions_ns_wins[t] = trees_ns_wins[t, action_node]
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
    @cuda.jit(void(int16, int64[:], int64[:], float32[:], float32[:], int16[:]))
    def _reduce_over_actions(n_root_actions, actions_ns, actions_ns_wins, best_score, best_q, best_action):
        shared_actions_ns = cuda.shared.array(512, dtype=int32) # 512 - assumed max n_actions
        shared_actions_ns_wins = cuda.shared.array(512, dtype=int32)
        shared_best_action = cuda.shared.array(512, dtype=int16)
        tpb = cuda.blockDim.x
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
            best_q[0] = shared_actions_ns_wins[0] / shared_actions_ns[0]            
            best_action[0] = shared_best_action[0]

    @staticmethod
    @cuda.jit(void(int64[:], int64[:], float32[:], float32[:], int16[:]))
    def _reduce_over_actions_prodigal(actions_ns, actions_ns_wins, best_score, best_q, best_action):
        shared_actions_ns = cuda.shared.array(512, dtype=int32) # 512 - assumed max n_actions
        shared_actions_ns_wins = cuda.shared.array(512, dtype=int32)
        shared_best_action = cuda.shared.array(512, dtype=int16)
        tpb = cuda.blockDim.x
        a = cuda.threadIdx.x # action index
        state_max_actions = actions_ns.size 
        if a < state_max_actions:
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
            best_q[0] = shared_actions_ns_wins[0] / shared_actions_ns[0]
            best_action[0] = shared_best_action[0]