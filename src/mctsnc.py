"""
This module contains the core algorithmic functionalities of the project, embodied by the class ``MCTSNC``. 

With CUDA computational model in mind, we have proposed and implemented in class ``MCTSNC`` four, fast operating and thoroughly parallel, variants of Monte Carlo Tree Search algorithm. 
The provided implementation takes advantage of `Numba <https://numba.pydata.org>`_, a just-in-time Python compiler, and its ``numba.cuda`` package (hence the "NC" suffix in the name). 
By `thoroughly parallel` we understand an algorithmic design that applies to both: (1) the structural elements of trees - leaf-/root-/tree-level parallelization 
(all those three are combined), and (2) the stages of MCTS - each stage in itself (selection, expansion, playouts, backup) employs multiple GPU threads. 
We apply suitable `reduction` patterns to carry out summations or max / argmax operations. Cooperation of threads helps to transfer information between global and shared memory. 
The implementation uses: no atomic operations, no mutexes (lock-free), and very few device-host memory transfers. 

Example usage 1 (Connect 4)
---------------------------
Assume the mechanics of the Connect 4 game have been defined to MCTS-NC in ``mctsnc_game_mechanics.py`` (via device functions ``is_action_legal``, ``take_action``, etc.), 
and that ``c4`` - instance of ``C4(State)`` - represents a state of an ongoing Connect 4 game shown below.

.. code-block:: console

    |.|.|●|○|.|.|.|
    |.|.|●|○|.|.|○|
    |.|.|●|●|.|●|●|
    |.|●|○|●|.|○|●|
    |.|○|●|○|.|●|○|
    |○|○|○|●|●|○|○|
     0 1 2 3 4 5 6      

Then, running the following code

.. code-block:: python

    ai = MCTSNC(C4.get_board_shape(), C4.get_extra_info_memory(), C4.get_max_actions())
    ai.init_device_side_arrays()
    best_action = ai.run(c4.get_board(), c4.get_extra_info(), c4.get_turn())
    print(f"BEST ACTION: {best_action}")

results in finding the best action for black - move 4 (winning in two plies), and the following printout:

.. code-block:: console

    [MCTSNC._init_device_side_arrays()... for MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
    [MCTSNC._init_device_side_arrays() done; time: 0.5193691253662109 s, per_state_memory: 95 B,  calculated max_tree_size: 2825549]
    MCTSNC RUN... [MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
    [actions info:
    {
      0: {'name': '0', 'n_root': 7474304, 'win_flag': False, 'n': 2182400, 'n_wins': 2100454, 'q': 0.9624514296187683, 'ucb': 0.9678373740384631},
      1: {'name': '1', 'n_root': 7474304, 'win_flag': False, 'n': 185344, 'n_wins': 164757, 'q': 0.8889254575276243, 'ucb': 0.9074070665330406},
      4: {'name': '4', 'n_root': 7474304, 'win_flag': False, 'n': 4921472, 'n_wins': 4885924, 'q': 0.9927769577882389, 'ucb': 0.9963635461474457},
      5: {'name': '5', 'n_root': 7474304, 'win_flag': False, 'n': 105472, 'n_wins': 91863, 'q': 0.8709704945388349, 'ucb': 0.8954701768685893},
      6: {'name': '6', 'n_root': 7474304, 'win_flag': False, 'n': 79616, 'n_wins': 68403, 'q': 0.8591614750803859, 'ucb': 0.8873601607647162},
      best: {'index': 4, 'name': '4', 'n_root': 7474304, 'win_flag': False, 'n': 4921472, 'n_wins': 4885924, 'q': 0.9927769577882389, 'ucb': 0.9963635461474457}
    }]
    [performance info:
    {
      steps: 6373,
      steps_per_second: 1274.0076324260813,
      playouts: 7474304,
      playouts_per_second: 1494166.0666990099,
      times_[ms]: {'total': 5002.324819564819, 'loop': 5000.642776489258, 'reduce_over_trees': 0.29015541076660156, 'reduce_over_actions': 0.4520416259765625, 'mean_loop': 0.7846607212441955, 'mean_select': 0.11222893376562147, 'mean_expand': 0.2786097114284054, 'mean_playout': 0.17186361935680036, 'mean_backup': 0.2193056618645448},
      trees: {'count': 8, 'mean_depth': 5.176703163017032, 'max_depth': 12, 'mean_size': 1233.0, 'max_size': 2736}
    }]
    MCTSNC RUN DONE. [time: 5.002324819564819 s; best action: 4, best win_flag: False, best n: 4921472, best n_wins: 4885924, best q: 0.9927769577882389]
    BEST ACTION: 4

Example usage 2 (Gomoku)
------------------------
Assume the mechanics of the Gomoku game have been defined to MCTS-NC in ``mctsnc_game_mechanics.py`` (via device functions ``is_action_legal``, ``take_action``, etc.), 
and that ``g`` - instance of ``Gomoku(State)`` - represents a state of an ongoing Gomoku game shown below.

.. code-block:: console

      ABCDEFGHIJKLMNO
    15+++++++++++++++15
    14+++++++++++++++14
    13+++++++++++++++13
    12++++++++●++++++12
    11++++++++○++++++11
    10++++++++○++++++10
     9++++++○+○++++++9
     8+++++++●○++++++8
     7+++++++●●●○++++7
     6++++++++●●○++++6
     5+++++++●+++++++5
     4+++++++++++++++4
     3+++++++++++++++3
     2+++++++++++++++2
     1+++++++++++++++1
      ABCDEFGHIJKLMNO
  
Then, running the following code

.. code-block:: python

    ai = MCTSNC(Gomoku.get_board_shape(), Gomoku.get_extra_info_memory(), Gomoku.get_max_actions(), action_index_to_name_function=Gomoku.action_index_to_name)
    ai.init_device_side_arrays()
    best_action = ai.run(g.get_board(), g.get_extra_info(), g.get_turn())
    print(f"BEST ACTION: {best_action}")

results in finding the defensive action for white - move K8 (indexed as 115) that prevents black from winning in three plies, and the following printout:

.. code-block:: console

    [MCTSNC._init_device_side_arrays()... for MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
    [MCTSNC._init_device_side_arrays() done; time: 0.5558419227600098 s, per_state_memory: 1144 B,  calculated max_tree_size: 234637]
    MCTSNC RUN... [MCTSNC(search_time_limit=5.0, search_steps_limit=inf, n_trees=8, n_playouts=128, variant='acp_prodigal', device_memory=2.0, ucb_c=2.0, seed: 0)]
    [actions info:
    {
      0: {'name': 'A1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148906, 'q': 0.3478852048444976, 'ucb': 0.36098484108863044},
      1: {'name': 'B1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 149000, 'q': 0.34810481459330145, 'ucb': 0.3612044508374343},
      2: {'name': 'C1', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 144339, 'q': 0.3372154418361244, 'ucb': 0.35031507808025725},
      ...
      115: {'name': 'K8', 'n_root': 94359552, 'win_flag': False, 'n': 1093632, 'n_wins': 452284, 'q': 0.41356141736891383, 'ucb': 0.4217566587685248},
      ...
      222: {'name': 'M15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 148009, 'q': 0.34578956713516745, 'ucb': 0.3588892033793003},
      223: {'name': 'N15', 'n_root': 94359552, 'win_flag': False, 'n': 401408, 'n_wins': 148802, 'q': 0.37070013552295916, 'ucb': 0.38422722440183954},
      224: {'name': 'O15', 'n_root': 94359552, 'win_flag': False, 'n': 428032, 'n_wins': 145329, 'q': 0.3395283530203349, 'ucb': 0.35262798926446776},
      best: {'index': 115, 'name': 'K8', 'n_root': 94359552, 'win_flag': False, 'n': 1093632, 'n_wins': 452284, 'q': 0.41356141736891383, 'ucb': 0.4217566587685248}
    }]
    [performance info:
    {
      steps: 442,
      steps_per_second: 88.25552729358726,
      playouts: 94359552,
      playouts_per_second: 18841067.91164404,
      times_[ms]: {'total': 5008.184909820557, 'loop': 5006.503105163574, 'reduce_over_trees': 0.20575523376464844, 'reduce_over_actions': 0.5161762237548828, 'mean_loop': 11.326930102180032, 'mean_select': 0.10066766005295974, 'mean_expand': 0.3082833139065704, 'mean_playout': 10.688265524298897, 'mean_backup': 0.226746317488036},
      trees: {'count': 8, 'mean_depth': 2.519115779878241, 'max_depth': 3, 'mean_size': 92149.0, 'max_size': 92149}
    }]
    MCTSNC RUN DONE. [time: 5.008184909820557 s; best action: 115 (K8), best win_flag: False, best n: 1093632, best n_wins: 452284, best q: 0.41356141736891383]
    BEST ACTION: 115
     
Dependencies
------------
- ``numpy``, ``math``: required for mathematical computations.

- ``numba``: required for just-in-time compilation of CUDA kernels (decorated by ``@cuda.jit``).

- ``mctsnc_game_mechanics``: required to define the mechanics of a wanted game or search problem via a set of five device-side functions - ``is_action_legal``, ``take_action``, ``legal_actions_playout``, ``take_action_playout``, ``compute_outcome`` callable by kernel functions of ``MCTSNC`` (see :doc:`mctsnc_game_mechanics`). 

- For usage of ``MCTSNC`` class, NVIDIA CUDA drivers must be present in the operating system. 

Link to project repository
--------------------------
`https://github.com/pklesk/mcts_numba_cuda <https://github.com/pklesk/mcts_numba_cuda>`_

Notes
-----
Private functions of ``MCTSNC`` class are named with a single leading underscore (e.g.: ``_set_cuda_constants``, 
``_make_performance_info``, ``_playout_acp_prodigal``, etc.). Among them, the kernel functions are additionally 
described by ``@cuda.jit`` decorators coming from ``numba`` module. Exact specifications of types come along with the decorators.
For public methods full docstrings are provided (with arguments and returns described). For private functions short docstrings are provided.    

"""

import numpy as np
from numpy import inf
from numba import cuda
from numba import void, int8, int16, int32, int64, float32, boolean
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_type 
import time
import math
from numba.core.errors import NumbaPerformanceWarning
import warnings
from mctsnc_game_mechanics import is_action_legal, take_action, legal_actions_playout, take_action_playout, compute_outcome
from utils import dict_to_str
import json

__version__ = "1.0.1"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl" 

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# the class
class MCTSNC:
    """
    Monte Carlo Tree Search implemented via ``numba.cuda`` meant for multi-threaded executions on GPU involving multiple concurrent trees and playouts (four algorithmic variants available). 
    """    
    
    # constants
    VARIANTS = ["ocp_thrifty", "ocp_prodigal", "acp_thrifty", "acp_prodigal"] # ocp - one child playouts, acp - all children playouts; thrifty/prodigal - accurate/overhead usage of cuda blocks (pertains to expanded actions)          
    DEFAULT_SEARCH_TIME_LIMIT = 5.0 # [s], np.inf possible
    DEFAULT_SEARCH_STEPS_LIMIT = np.inf # integer, np.inf possible
    DEFAULT_N_TREES = 8
    DEFAULT_N_PLAYOUTS = 128
    DEFAULT_VARIANT = VARIANTS[-1]            
    DEFAULT_DEVICE_MEMORY = 2.0 
    DEFAULT_UCB_C = 2.0
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
                 search_time_limit=DEFAULT_SEARCH_TIME_LIMIT, search_steps_limit=DEFAULT_SEARCH_STEPS_LIMIT,
                 n_trees=DEFAULT_N_TREES, n_playouts=DEFAULT_N_PLAYOUTS, variant=DEFAULT_VARIANT, device_memory=DEFAULT_DEVICE_MEMORY,                   
                 ucb_c=DEFAULT_UCB_C, seed=DEFAULT_SEED,
                 verbose_debug=DEFAULT_VERBOSE_DEBUG, verbose_info=DEFAULT_VERBOSE_INFO,
                 action_index_to_name_function=None):
        """
        Constructor of ``MCTSNC`` instances.
         
        Args:
            state_board_shape (tuple(int, int)):
                shape of board for states in a given game, at most ``(32, 32)``.
            state_extra_info_memory (int):
                number of bytes for extra information on states, at most ``4096``.        
            state_max_actions (int): 
                maximum branching factor, at most ``512``.            
            search_time_limit (float):
                time limit in seconds (computational budget), ``np.inf`` if no limit, defaults to ``5.0``.             
            search_steps_limit (float): 
                steps limit (computational budget), ``np.inf`` if no limit, defaults to ``np.inf``.
            n_trees (int): 
                number of independent trees, defaults to ``8``.                                    
            n_playouts (int):
                number of independent playouts from an expanded child, must be a power of 2, defaults to ``128``.            
            variant (str):
                choice of algorithmic variant from {``"ocp_thrifty"``, ``"ocp_prodigal"``, ``"acp_thrifty``, ``"acp_prodigal``}, defaults to ``"acp_prodigal"``.        
            device_memory (float): 
                GPU memory in GiBs (gibibytes) to be available for this instance, defaults to ``2.0``.
            ucb_c (float):
                value of C constant, influencing exploration tendency, appearing in UCT formula (upper confidence bounds for trees), defaults to ``2.0``. 
            verbose_debug (bool):
                debug verbosity flag, if ``True`` then detailed information about each kernel invocation are printed to console (in each iteration), defaults to ``False``.
            verbose_info (bool): 
                verbosity flag, if ``True`` then standard information on actions and performance are printed to console (after a full run), defaults to ``True``.
            action_index_to_name_function (callable):
                pointer to user-provided function converting action indexes to a human-friendly names (e.g. ``"e2:e4"`` for chess), defaults to ``None``.            
        """
        self._set_cuda_constants()
        if not self.cuda_available:
            sys.exit(f"[MCTSNC.__init__(): exiting due to cuda computations not available]")        
        self.state_board_shape = state_board_shape
        if self.state_board_shape[0] > self.MAX_STATE_BOARD_SHAPE[0] or self.state_board_shape[1] > self.MAX_STATE_BOARD_SHAPE[1]:
            sys.exit(f"[MCTSNC.__init__(): exiting due to allowed state board shape exceeded]")            
        self.state_extra_info_memory = max(state_extra_info_memory, 1)
        if self.state_extra_info_memory > self.MAX_STATE_EXTRA_INFO_MEMORY:
            sys.exit(f"[MCTSNC.__init__(): exiting due to allowed state extra info memory exceeded]")        
        self.state_max_actions = state_max_actions
        if self.state_max_actions > self.MAX_STATE_MAX_ACTIONS:
            sys.exit(f"[MCTSNC.__init__(): exiting due to allowed state max actions memory exceeded]")        
        if self.state_max_actions > self.cuda_tpb_default:
            sys.exit(f"[MCTSNC.__init__(): exiting due to state max actions exceeding half of cuda default tpb]")
        self.search_time_limit = search_time_limit
        self._validate_param("search_time_limit", float, True, 0.0, False, np.inf, self.DEFAULT_SEARCH_TIME_LIMIT)
        self.search_steps_limit = float(search_steps_limit)        
        self._validate_param("search_steps_limit", float, True, 0.0, False, np.inf, self.DEFAULT_SEARCH_STEPS_LIMIT) # purposely float, so that np.inf possible                        
        self.n_trees = n_trees
        self._validate_param("n_trees", int, False, 1, False, self.MAX_N_TREES, self.DEFAULT_N_TREES)            
        if 2**np.round(np.log2(n_playouts)) != n_playouts:
            invalid_n_playouts = n_playouts
            n_playouts = self.DEFAULT_N_PLAYOUTS
            print(f"[n_playouts: {invalid_n_playouts} is not a power of 2; changed to default: {n_playouts}]")
        self.n_playouts = n_playouts            
        self._validate_param("n_playouts", int, False, 1, False, self.MAX_N_PLAYOUTS, self.DEFAULT_N_PLAYOUTS)        
        if not variant in self.VARIANTS:
            invalid_variant = variant
            variant = self.DEFAULT_VARIANT
            print(f"[invalid variant: '{invalid_variant}' changed to default: '{variant}'; possible variants: {self.VARIANTS}]")
        self.variant = variant
        self.ucb_c = ucb_c
        self._validate_param("ucb_c", float, False, 0.0, False, np.inf, self.DEFAULT_UCB_C)
        self.device_memory = device_memory * 1024**3 # gibibytes (GiB) to bytes (B)
        self._validate_param("device_memory", float, True, 0.0, False, np.inf, self.DEFAULT_DEVICE_MEMORY)    
        self.seed = seed
        self.verbose_debug = verbose_debug 
        self._validate_param("verbose_debug", bool, False, False, False, True, self.DEFAULT_VERBOSE_DEBUG)
        self.verbose_info = verbose_info 
        self._validate_param("verbose_info", bool, False, False, False, True, self.DEFAULT_VERBOSE_INFO)        
        self.action_index_to_name_function = action_index_to_name_function                                                                  
    
    def _set_cuda_constants(self):
        """Investigates (via ``numba`` module) if CUDA-based computations are available and, if so, sets suitable constants."""
        self.cuda_available = cuda.is_available() 
        self.cuda_tpb_default = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2 if self.cuda_available else None
    
    def _validate_param(self, name, ptype, leq, low, geq, high, default):
        """Validates a parameter - is it of correct type and within given range (either end of the range can be open or closed)."""
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
            
    def __str__(self):
        """
        Returns a string representation of this ``MCTSNC`` instance.
        
        Returns:
            str: string representation of this ``MCTSNC`` instance.
        """   
        return f"MCTSNC(search_time_limit={self.search_time_limit}, search_steps_limit={self.search_steps_limit}, n_trees={self.n_trees}, n_playouts={self.n_playouts}, variant='{self.variant}', device_memory={np.round(self.device_memory / 1024**3, 2)}, ucb_c={self.ucb_c}, seed: {self.seed})"
        
    def __repr__(self):
        """
        Returns a detailed string representation of this ``MCTSNC`` instance.
        
        Returns:
            str: detailed string representation of this ``MCTSNC`` instance.
        """        
        repr_str = f"{str(self)}, "
        repr_str += f"state_board_shape={self.state_board_shape}, state_extra_info_memory={self.state_extra_info_memory}, state_max_actions={self.state_max_actions})"
        return repr_str            
        
    def init_device_side_arrays(self):
        """
        Allocates all the necessary device arrays based on relevant constants and available memory.
        """
        if self.verbose_info:
            print(f"[MCTSNC._init_device_side_arrays()... for {self}]")
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
        if "acp" in self.variant: # playout all children
            per_tree_additional_memory += playout_outcomes_bytes * self.state_max_actions * 2  # playout children outcomes            
        per_state_memory = board_element_bytes * np.prod(self.state_board_shape) + extra_info_element_bytes * self.state_extra_info_memory \
                            + node_index_bytes * (1 + self.state_max_actions) + per_state_additional_memory # board, extra info, tree array entry (parent, children nodes), additional memory
        self.max_tree_size = (int(self.device_memory) - self.n_trees * per_tree_additional_memory) // (per_state_memory * self.n_trees)
        self.max_tree_size = min(self.max_tree_size, self.MAX_TREE_SIZE)
        # tpb 
        tpb_board = int(2**np.ceil(np.log2(np.prod(self.state_board_shape))))
        tpb_extra_info = int(2**np.ceil(np.log2(self.state_extra_info_memory))) if self.state_extra_info_memory > 0 else 1
        tpb_max_actions = int(2**np.ceil(np.log2(self.state_max_actions)))        
        self.tpb_r = min(max(tpb_board, tpb_extra_info), self.cuda_tpb_default)
        self.tpb_s = self.cuda_tpb_default
        self.tpb_e1 = min(max(self.tpb_r, tpb_max_actions), self.cuda_tpb_default)        
        self.tpb_e2 = self.tpb_r
        self.tpb_b1 = tpb_max_actions                                                    
        self.tpb_b2 = self.cuda_tpb_default
        self.tpb_rot = int(2**np.ceil(np.log2(self.n_trees))) # rot - reduce over trees 
        self.tpb_roa = tpb_max_actions # roa - reduce over actions
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
        self.dev_random_generators_expand_1 = None         
        self.dev_random_generators_playout = None
        if "ocp" in self.variant:
            self.dev_random_generators_expand_1 = create_xoroshiro128p_states(self.n_trees * self.tpb_e1, seed=self.seed)
            self.dev_random_generators_playout = create_xoroshiro128p_states(self.n_trees * self.n_playouts, seed=self.seed)
        else: # "acp"
            self.dev_random_generators_playout = create_xoroshiro128p_states(self.n_trees * self.state_max_actions * self.n_playouts, seed=self.seed)                    
            self.dev_trees_playout_outcomes_children = cuda.device_array((self.n_trees, self.state_max_actions, 2), dtype=playout_outcomes_dtype) # for each (playable) action, each row stores counts of: -1 wins and +1 wins, respectively (for given tree)
        self.dev_root_actions_expanded = cuda.device_array(self.state_max_actions + 2, dtype=action_index_dtype)                    
        self.dev_root_ns = cuda.device_array(self.state_max_actions, dtype=ns_extended_dtype) # all entries the same regardless of root action (overhead for convenience)
        self.dev_actions_win_flags = cuda.device_array(self.state_max_actions, dtype=flag_dtype)
        self.dev_actions_ns = cuda.device_array(self.state_max_actions, dtype=ns_extended_dtype)
        self.dev_actions_ns_wins = cuda.device_array(self.state_max_actions, dtype=ns_extended_dtype)        
        self.dev_best_action = cuda.device_array(1, dtype=action_index_dtype) 
        self.dev_best_win_flag = cuda.device_array(1, dtype=flag_dtype)                
        self.dev_best_n = cuda.device_array(1, dtype=ns_extended_dtype)
        self.dev_best_n_wins = cuda.device_array(1, dtype=ns_extended_dtype)                 
        t2_dev_arrays = time.time()
        if self.verbose_info:
            print(f"[MCTSNC._init_device_side_arrays() done; time: {t2_dev_arrays - t1_dev_arrays} s, per_state_memory: {per_state_memory} B,  calculated max_tree_size: {self.max_tree_size}]")
        
    def run(self, root_board, root_extra_info, root_turn, forced_search_steps_limit=np.inf):
        """
        Runs the Monte Carlo Tree Search on GPU involving multiple concurrent trees and playouts.                 
        Computations are carried out according to the formerly chosen algorithmic variant, i.e. one of {``"ocp_thrifty"``, ``"ocp_prodigal"``, ``"acp_thrifty``, ``"acp_prodigal``}, defaults to ``"acp_prodigal"``}.
        
        Args:
            root_board (ndarray): 
                two-dimensional array with board (or other representation) of root state from which the search starts.
            root_extra_info (ndarray): 
                any additional information of root state not implied by the contents of the board itself (e.g. possibilities of castling or en-passant captures in chess, the contract in double dummy bridge, etc.), or technical information useful to generate legal actions faster.
            root_turn {-1, 1}:
                indicator of the player, minimizing or maximizing, to act first at root state.
            forced_search_steps_limit (int):
                steps limit used only when reproducing results of a previous experiment; if less than``np.inf`` then has a priority over the standard computational budget given by ``search_time_limit`` and ``search_steps_limit``.
        Returns:
            self.best_action (int):
                best action resulting from search.
        """
        print(f"MCTSNC RUN... [{self}]")        
        run_method = getattr(self, "_run_" + self.variant)
        run_method(root_board, root_extra_info, root_turn, forced_search_steps_limit)
        best_action_label = str(self.best_action)
        if self.action_index_to_name_function is not None:
            best_action_label += f" ({self.action_index_to_name_function(self.best_action)})"
        print(f"MCTSNC RUN DONE. [time: {self.time_total} s; best action: {best_action_label}, best win_flag: {self.best_win_flag}, best n: {self.best_n}, best n_wins: {self.best_n_wins}, best q: {self.best_q}]")
        return self.best_action
    
    def _flatten_trees_actions_expanded_thrifty(self, trees_actions_expanded):
        """Uses information from array ``trees_actions_expanded`` of shape ``(self.n_trees, self.state_max_actions + 2)`` and converts it to another array where the number of rows corresponds to the total of expanded legal actions in all trees. Each row contains a pair of indexes for: action and tree. The approach allows to allocate exact number of needed CUDA blocks for further operations."""            
        actions_expanded_cumsum = np.cumsum(trees_actions_expanded[:, -1])
        trees_actions_expanded_flat = -np.ones((actions_expanded_cumsum[-1], 2), dtype=np.int16)
        shift = 0
        for ti in range(self.n_trees):
            s = slice(shift, actions_expanded_cumsum[ti])
            trees_actions_expanded_flat[s, 0] = ti
            trees_actions_expanded_flat[s, 1] = trees_actions_expanded[ti, :trees_actions_expanded[ti, -1]]
            shift = actions_expanded_cumsum[ti]                                        
        return trees_actions_expanded_flat
    
    def _make_performance_info(self):
        """
        Prepares and returns a dictionary with information on performance during the last run. 
        After the call, available via ``performance_info`` attribute.
        """
        performance_info = {}
        performance_info["steps"] = int(self.steps)
        performance_info["steps_per_second"] = self.steps / self.time_total                
        root_ns = self.dev_root_ns.copy_to_host()
        playouts = root_ns[root_ns > 0][0]
        performance_info["playouts"] = int(playouts) 
        performance_info["playouts_per_second"] = performance_info["playouts"] / self.time_total           
        ms_factor = 10.0**3
        times_info = {}
        times_info["total"] = ms_factor * self.time_total
        times_info["loop"] = ms_factor * self.time_loop
        times_info["reduce_over_trees"] = ms_factor * self.time_reduce_over_trees
        times_info["reduce_over_actions"] = ms_factor * self.time_reduce_over_actions
        times_info["mean_loop"] = times_info["loop"] / self.steps
        times_info["mean_select"] = ms_factor * self.time_select / self.steps
        times_info["mean_expand"] = ms_factor * self.time_expand / self.steps
        times_info["mean_playout"] = ms_factor * self.time_playout / self.steps
        times_info["mean_backup"] = ms_factor * self.time_backup / self.steps
        performance_info["times_[ms]"] = times_info                                                              
        trees_depths = np.empty_like(self.dev_trees_depths)
        trees_sizes = np.empty_like(self.dev_trees_sizes)
        self.dev_trees_depths.copy_to_host(ary=trees_depths)
        self.dev_trees_sizes.copy_to_host(ary=trees_sizes)        
        mean_depth = 0
        max_depth = -1        
        for i in range(self.n_trees):
            depths_up_to_size = trees_depths[i, :trees_sizes[i]]
            mean_depth += np.sum(depths_up_to_size)                         
            max_depth = max(max_depth, np.max(depths_up_to_size))
        total_size = np.sum(trees_sizes)
        max_size = np.max(trees_sizes)
        mean_size = total_size / self.n_trees
        mean_depth /= total_size
        trees_info = {}
        trees_info["count"] = int(self.n_trees)
        trees_info["mean_depth"] = mean_depth
        trees_info["max_depth"] = int(max_depth)
        trees_info["mean_size"] = mean_size
        trees_info["max_size"] = int(max_size)
        performance_info["trees"] = trees_info
        self.performance_info = performance_info
        return performance_info
    
    def _make_actions_info_thrifty(self):
        """
        Prepares and returns a dictionary with information on root actions (using thrifty indexing) implied by the last run, in particular: estimates of action values, their UCBs, counts of times actions were taken, etc.
        After the call, available via ``actions_info`` attribute.
        """
        root_actions_expanded = np.empty_like(self.dev_root_actions_expanded)        
        root_ns_thrifty = np.empty_like(self.dev_root_ns)                
        actions_win_flags_thrifty = np.empty_like(self.dev_actions_win_flags)
        actions_ns_thrifty = np.empty_like(self.dev_actions_ns)
        actions_ns_wins_thrifty = np.empty_like(self.dev_actions_ns_wins)
        self.dev_root_actions_expanded.copy_to_host(ary=root_actions_expanded)
        self.dev_root_ns.copy_to_host(ary=root_ns_thrifty)               
        self.dev_actions_win_flags.copy_to_host(ary=actions_win_flags_thrifty)
        self.dev_actions_ns.copy_to_host(ary=actions_ns_thrifty)
        self.dev_actions_ns_wins.copy_to_host(ary=actions_ns_wins_thrifty)
        actions_info = {}
        best_entry = None 
        n_root_actions = root_actions_expanded[-1]
        for i in range(n_root_actions):
            entry = {}                        
            a = root_actions_expanded[i] # for thrifty variants
            entry["name"] = self.action_index_to_name_function(a) if self.action_index_to_name_function else str(a)
            entry["n_root"] = int(root_ns_thrifty[i])
            entry["win_flag"] = bool(actions_win_flags_thrifty[i])
            entry["n"] = int(actions_ns_thrifty[i])
            entry["n_wins"] = int(actions_ns_wins_thrifty[i])
            entry["q"] = entry["n_wins"] / entry["n"] if entry["n"] > 0 else np.nan                          
            entry["ucb"] = entry["q"] + self.ucb_c * np.sqrt(np.log(entry["n_root"]) / entry["n"]) if entry["n"] > 0 else np.inf
            actions_info[a] = entry
            if a == self.best_action:
                best_entry = {"index": int(a), **entry}
        actions_info["best"] = best_entry
        self.actions_info = actions_info
        return actions_info
    
    def _make_actions_info_prodigal(self):
        """
        Prepares and returns a dictionary with information on root actions (using prodigal indexing) implied by the last run, in particular: estimates of action values, their UCBs, counts of times actions were taken, etc.
        After the call, available via ``actions_info`` attribute.
        """
        root_ns_prodigal = np.empty_like(self.dev_root_ns)            
        actions_win_flags_prodigal = np.empty_like(self.dev_actions_win_flags)
        actions_ns_prodigal = np.empty_like(self.dev_actions_ns)
        actions_ns_wins_prodigal = np.empty_like(self.dev_actions_ns_wins)
        self.dev_root_ns.copy_to_host(ary=root_ns_prodigal)               
        self.dev_actions_win_flags.copy_to_host(ary=actions_win_flags_prodigal)
        self.dev_actions_ns.copy_to_host(ary=actions_ns_prodigal)
        self.dev_actions_ns_wins.copy_to_host(ary=actions_ns_wins_prodigal)
        actions_info = {}
        best_entry = None 
        for i in range(self.state_max_actions):
            if root_ns_prodigal[i] == 0:
                continue            
            a = i # for prodigal variants
            entry = {}
            entry["name"] = self.action_index_to_name_function(a) if self.action_index_to_name_function else str(a)
            entry["n_root"] = int(root_ns_prodigal[i])
            entry["win_flag"] = bool(actions_win_flags_prodigal[i])
            entry["n"] = int(actions_ns_prodigal[i])
            entry["n_wins"] = int(actions_ns_wins_prodigal[i])
            entry["q"] = entry["n_wins"] / entry["n"] if entry["n"] > 0 else np.nan                          
            entry["ucb"] = entry["q"] + self.ucb_c * np.sqrt(np.log(entry["n_root"]) / entry["n"]) if entry["n"] > 0 else np.inf
            actions_info[a] = entry
            if a == self.best_action:
                best_entry = {"index": int(a), **entry}
        actions_info["best"] = best_entry
        self.actions_info = actions_info
        return actions_info
                                                   
    def _run_ocp_thrifty(self, root_board, root_extra_info, root_turn, forced_search_steps_limit=np.inf):
        """Runs computations for algorithmic variant: ``"ocp_thrifty"``."""
        t1 = time.time()
        
        # reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self.tpb_r
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.verbose_debug:
            print(f"[MCTSNC._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSNC._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                self.dev_trees, self.dev_trees_sizes, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                self.dev_trees_boards, self.dev_trees_extra_infos)
        cuda.synchronize()    
        t2_reset = time.time()
        if self.verbose_debug:
            print(f"[MCTSNC._reset() done; time: {t2_reset - t1_reset} s]")
            
        self.time_select = 0.0
        self.time_expand = 0.0        
        self.time_playout = 0.0
        self.time_backup = 0.0    
        self.steps = 0
        trees_actions_expanded = np.empty((self.n_trees, self.state_max_actions + 2), dtype=np.int16) # needed at host side for thrifty variants
        
        t1_loop = time.time()
        while True:
            t2_loop = time.time()            
            if forced_search_steps_limit < np.inf: 
                if self.steps >= forced_search_steps_limit:
                    break
            elif self.steps >= self.search_steps_limit or t2_loop - t1_loop >= self.search_time_limit:
                break
            if self.verbose_debug:
                print(f"[step: {self.steps + 1} starting, time used so far: {t2_loop - t1_loop} s]")     
            
            # selections
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self.tpb_s
            if self.verbose_debug:
                print(f"[MCTSNC._select()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._select[bpg, tpb](self.ucb_c, 
                                     self.dev_trees, self.dev_trees_leaves, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                     self.dev_trees_nodes_selected, self.dev_trees_selected_paths)
            cuda.synchronize()
            t2_select = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._select() done; time: {t2_select - t1_select} s]")
            self.time_select += t2_select - t1_select
            
            # expansions            
            t1_expand = time.time()
            t1_expand_1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_e1
            if self.verbose_debug:
                print(f"[MCTSNC._expand_1_ocp_thrifty()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSNC._expand_1_ocp_thrifty[bpg, tpb](self.max_tree_size, 
                                                   self.dev_trees, self.dev_trees_sizes, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals,
                                                   self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                   self.dev_trees_nodes_selected, self.dev_random_generators_expand_1, self.dev_trees_actions_expanded)                                                    
            self.dev_trees_actions_expanded.copy_to_host(ary=trees_actions_expanded)
            cuda.synchronize()
            if self.steps == 0:                
                MCTSNC._memorize_root_actions_expanded[1, self.state_max_actions + 2](self.dev_trees_actions_expanded, self.dev_root_actions_expanded)                            
                cuda.synchronize()
            t2_expand_1 = time.time()            
            if self.verbose_debug:
                print(f"[MCTSNC._expand_1_ocp_thrifty() done; time: {t2_expand_1 - t1_expand_1} s]")
            t1_expand_2 = time.time()            
            trees_actions_expanded_flat = self._flatten_trees_actions_expanded_thrifty(trees_actions_expanded)
            bpg = trees_actions_expanded_flat.shape[0] # thrifty number of blocks
            tpb = self.tpb_e2
            if self.verbose_debug:
                print(f"[MCTSNC._expand_2_thrifty()...; bpg: {bpg}, tpb: {tpb}]")
            dev_trees_actions_expanded_flat = cuda.to_device(trees_actions_expanded_flat)
            MCTSNC._expand_2_thrifty[bpg, tpb](self.dev_trees, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_outcomes, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                               self.dev_trees_boards, self.dev_trees_extra_infos,                                               
                                               self.dev_trees_nodes_selected, dev_trees_actions_expanded_flat)
            cuda.synchronize()
            t2_expand_2 = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._expand_2_thrifty() done; time: {t2_expand_2 - t1_expand_2} s]")
            t2_expand = time.time()
            self.time_expand += t2_expand - t1_expand
            
            # playouts
            t1_playout = time.time()
            bpg = self.n_trees
            tpb = self.n_playouts
            if self.verbose_debug:
                print(f"[MCTSNC._playout_ocp()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._playout_ocp[bpg, tpb](self.dev_trees, self.dev_trees_turns, self.dev_trees_terminals, self.dev_trees_outcomes, 
                                          self.dev_trees_boards, self.dev_trees_extra_infos, 
                                          self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, 
                                          self.dev_random_generators_playout, self.dev_trees_playout_outcomes)
            cuda.synchronize()
            t2_playout = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._playout_ocp() done; time: {t2_playout - t1_playout} s]")
            self.time_playout += t2_playout - t1_playout
            
            # backups
            t1_backup = time.time()  
            bpg = self.n_trees            
            tpb = self.tpb_b2                     
            if self.verbose_debug:
                print(f"[MCTSNC._backup_ocp()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._backup_ocp[bpg, tpb](self.n_playouts,
                                         self.dev_trees, self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                         self.dev_trees_nodes_selected, self.dev_trees_selected_paths, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes)                                
            cuda.synchronize()            
            t2_backup = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._backup() done; time: {t2_backup - t1_backup} s]")
            self.time_backup += t2_backup - t1_backup                                        
            self.steps += 1
        self.time_loop = time.time() - t1_loop
            
        # sum reduction over trees for each root action        
        t1_reduce_over_trees = time.time()
        root_actions_expanded = np.empty_like(self.dev_root_actions_expanded)
        self.dev_root_actions_expanded.copy_to_host(ary=root_actions_expanded)
        n_root_actions = int(root_actions_expanded[-1]) 
        bpg = n_root_actions
        tpb = self.tpb_rot
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_trees_thrifty()...; bpg: {bpg}, tpb: {tpb}]")
        MCTSNC._reduce_over_trees_thrifty[bpg, tpb](self.dev_trees, self.dev_trees_terminals, self.dev_trees_outcomes,
                                                    self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                    self.dev_root_actions_expanded, root_turn,
                                                    self.dev_root_ns, self.dev_actions_win_flags, self.dev_actions_ns, self.dev_actions_ns_wins)
        cuda.synchronize()
        t2_reduce_over_trees = time.time()
        self.time_reduce_over_trees = t2_reduce_over_trees - t1_reduce_over_trees
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_trees_thrifty() done; time: {self.time_reduce_over_trees} s]")
            
        # max-argmax reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self.tpb_roa
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_actions_thrifty()...; bpg: {bpg}, tpb: {tpb}]")                                                
        MCTSNC._reduce_over_actions_thrifty[bpg, tpb](n_root_actions, 
                                                      self.dev_actions_win_flags, self.dev_actions_ns, self.dev_actions_ns_wins, 
                                                      self.dev_best_action, self.dev_best_win_flag, self.dev_best_n, self.dev_best_n_wins)
        self.best_action = self.dev_best_action.copy_to_host()[0]
        self.best_win_flag = self.dev_best_win_flag.copy_to_host()[0]                
        self.best_n = self.dev_best_n.copy_to_host()[0]
        self.best_n_wins = self.dev_best_n_wins.copy_to_host()[0]
        self.best_q = self.best_n_wins / self.best_n if self.best_n > 0 else np.nan        
        cuda.synchronize()
        self.best_action = root_actions_expanded[self.best_action]
        t2_reduce_over_actions = time.time()
        self.time_reduce_over_actions = t2_reduce_over_actions - t1_reduce_over_actions 
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_actions_thrifty() done; time: {self.time_reduce_over_actions} s]")                
        t2 = time.time()
        self.time_total = t2 - t1
        
        if self.verbose_info:
            print(f"[actions info:\n{dict_to_str(self._make_actions_info_thrifty())}]")
            print(f"[performance info:\n{dict_to_str(self._make_performance_info())}]")
                         
    def _run_ocp_prodigal(self, root_board, root_extra_info, root_turn, forced_search_steps_limit=np.inf):
        """Runs computations for algorithmic variant: ``"ocp_prodigal"``."""
        t1 = time.time()
        
        # reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self.tpb_r
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.verbose_debug:
            print(f"[MCTSNC._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSNC._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                self.dev_trees, self.dev_trees_sizes, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                self.dev_trees_boards, self.dev_trees_extra_infos)
        cuda.synchronize()    
        t2_reset = time.time()
        if self.verbose_debug:
            print(f"[MCTSNC._reset() done; time: {t2_reset - t1_reset} s]")
            
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
            if self.verbose_debug:
                print(f"[step: {self.steps + 1} starting, time used so far: {t2_loop - t1_loop} s]")     
            
            # selections
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self.tpb_s
            if self.verbose_debug:
                print(f"[MCTSNC._select()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._select[bpg, tpb](self.ucb_c, 
                                     self.dev_trees, self.dev_trees_leaves, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                     self.dev_trees_nodes_selected, self.dev_trees_selected_paths)
            cuda.synchronize()
            t2_select = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._select() done; time: {t2_select - t1_select} s]")
            self.time_select += t2_select - t1_select
            
            # expansions             
            t1_expand = time.time()
            t1_expand_1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_e1
            if self.verbose_debug:
                print(f"[MCTSNC._expand_1_ocp_prodigal()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSNC._expand_1_ocp_prodigal[bpg, tpb](self.max_tree_size, 
                                                    self.dev_trees, self.dev_trees_sizes, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals,
                                                    self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                    self.dev_trees_nodes_selected, self.dev_random_generators_expand_1, self.dev_trees_actions_expanded)                                                    
            cuda.synchronize()
            if self.steps == 0:                
                MCTSNC._memorize_root_actions_expanded[1, self.state_max_actions + 2](self.dev_trees_actions_expanded, self.dev_root_actions_expanded)
                cuda.synchronize()
            t2_expand_1 = time.time()            
            if self.verbose_debug:
                print(f"[MCTSNC._expand_1_ocp_progial() done; time: {t2_expand_1 - t1_expand_1} s]")
            t1_expand_2 = time.time()            
            bpg = (self.n_trees, self.state_max_actions) # prodigal number of blocks
            tpb = self.tpb_e2
            if self.verbose_debug:
                print(f"[MCTSNC._expand_2_prodigal()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._expand_2_prodigal[bpg, tpb](self.dev_trees, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_outcomes, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                self.dev_trees_boards, self.dev_trees_extra_infos,                                               
                                                self.dev_trees_nodes_selected, self.dev_trees_actions_expanded)
            cuda.synchronize()
            t2_expand_2 = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._expand_2_prodigal() done; time: {t2_expand_2 - t1_expand_2} s]")
            t2_expand = time.time()
            self.time_expand += t2_expand - t1_expand
            
            # playouts
            t1_playout = time.time()
            bpg = self.n_trees 
            tpb = self.n_playouts
            if self.verbose_debug:
                print(f"[MCTSNC._playout_ocp()...; bpg: {bpg}, tpb: {tpb}]")            
            MCTSNC._playout_ocp[bpg, tpb](self.dev_trees, self.dev_trees_turns, self.dev_trees_terminals, self.dev_trees_outcomes, 
                                            self.dev_trees_boards, self.dev_trees_extra_infos, 
                                            self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, 
                                            self.dev_random_generators_playout, self.dev_trees_playout_outcomes)
            cuda.synchronize()
            t2_playout = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._playout_ocp() done; time: {t2_playout - t1_playout} s]")
            self.time_playout += t2_playout - t1_playout
            
            # backups
            t1_backup = time.time()
            bpg = self.n_trees            
            tpb = self.tpb_b2                     
            if self.verbose_debug:
                print(f"[MCTSNC._backup_ocp()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._backup_ocp[bpg, tpb](self.n_playouts,
                                         self.dev_trees, self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                         self.dev_trees_nodes_selected, self.dev_trees_selected_paths, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes)                                
            cuda.synchronize()            
            t2_backup = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._backup() done; time: {t2_backup - t1_backup} s]")
            self.time_backup += t2_backup - t1_backup                                        
            self.steps += 1
        self.time_loop = time.time() - t1_loop
            
        # sum reduction over trees for each root action        
        t1_reduce_over_trees = time.time() 
        bpg = self.state_max_actions
        tpb = self.tpb_rot
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_trees_prodigal()...; bpg: {bpg}, tpb: {tpb}]")
        MCTSNC._reduce_over_trees_prodigal[bpg, tpb](self.dev_trees, self.dev_trees_terminals, self.dev_trees_outcomes,
                                                     self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                     self.dev_root_actions_expanded, root_turn,
                                                     self.dev_root_ns, self.dev_actions_win_flags, self.dev_actions_ns, self.dev_actions_ns_wins)
        cuda.synchronize()
        t2_reduce_over_trees = time.time()
        self.time_reduce_over_trees = t2_reduce_over_trees - t1_reduce_over_trees
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_trees_prodigal() done; time: {self.time_reduce_over_trees} s]")
            
        # max-argmax reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self.tpb_roa
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_actions_prodigal()...; bpg: {bpg}, tpb: {tpb}]")                                                
        MCTSNC._reduce_over_actions_prodigal[bpg, tpb](self.dev_actions_win_flags, self.dev_actions_ns, self.dev_actions_ns_wins, 
                                                       self.dev_best_action, self.dev_best_win_flag, self.dev_best_n, self.dev_best_n_wins)        
        self.best_action = self.dev_best_action.copy_to_host()[0]
        self.best_win_flag = self.dev_best_win_flag.copy_to_host()[0]                
        self.best_n = self.dev_best_n.copy_to_host()[0]
        self.best_n_wins = self.dev_best_n_wins.copy_to_host()[0]
        self.best_q = self.best_n_wins / self.best_n if self.best_n > 0 else np.nan        
        cuda.synchronize()
        t2_reduce_over_actions = time.time()
        self.time_reduce_over_actions = t2_reduce_over_actions - t1_reduce_over_actions 
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_actions_prodigal() done; time: {self.time_reduce_over_actions} s]")                
        t2 = time.time()
        self.time_total = t2 - t1
        
        if self.verbose_info:
            print(f"[actions info:\n{dict_to_str(self._make_actions_info_prodigal())}]")
            print(f"[performance info:\n{dict_to_str(self._make_performance_info())}]")
                                                  
    def _run_acp_thrifty(self, root_board, root_extra_info, root_turn, forced_search_steps_limit=np.inf):
        """Runs computations for algorithmic variant: ``"acp_thrifty"``."""
        t1 = time.time()
        
        # reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self.tpb_r
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array        
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.verbose_debug:
            print(f"[MCTSNC._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSNC._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                self.dev_trees, self.dev_trees_sizes, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                self.dev_trees_boards, self.dev_trees_extra_infos)
        cuda.synchronize()    
        t2_reset = time.time()
        if self.verbose_debug:
            print(f"[MCTSNC._reset() done; time: {t2_reset - t1_reset} s]")
            
        self.time_select = 0.0
        self.time_expand = 0.0        
        self.time_playout = 0.0
        self.time_backup = 0.0        
        self.steps = 0        
        trees_actions_expanded = np.empty((self.n_trees, self.state_max_actions + 2), dtype=np.int16)
        
        t1_loop = time.time()
        while True:
            t2_loop = time.time()
            if forced_search_steps_limit < np.inf: 
                if self.steps >= forced_search_steps_limit:
                    break            
            elif self.steps >= self.search_steps_limit or t2_loop - t1_loop >= self.search_time_limit:
                break
            if self.verbose_debug:
                print(f"[step: {self.steps + 1} starting, time used so far: {t2_loop - t1_loop} s]")     
            
            # selections
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self.tpb_s
            if self.verbose_debug:
                print(f"[MCTSNC._select()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._select[bpg, tpb](self.ucb_c, 
                                     self.dev_trees, self.dev_trees_leaves, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                     self.dev_trees_nodes_selected, self.dev_trees_selected_paths)
            cuda.synchronize()
            t2_select = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._select() done; time: {t2_select - t1_select} s]")
            self.time_select += t2_select - t1_select
                                        
            # expansions
            t1_expand = time.time()           
            t1_expand_1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_e1
            if self.verbose_debug:
                print(f"[MCTSNC._expand_1_acp_thrifty()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSNC._expand_1_acp_thrifty[bpg, tpb](self.max_tree_size, 
                                                   self.dev_trees, self.dev_trees_sizes, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals,
                                                   self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                   self.dev_trees_nodes_selected, self.dev_trees_actions_expanded)                                             
            self.dev_trees_actions_expanded.copy_to_host(ary=trees_actions_expanded)
            cuda.synchronize()            
            if self.steps == 0:            
                MCTSNC._memorize_root_actions_expanded[1, self.state_max_actions + 2](self.dev_trees_actions_expanded, self.dev_root_actions_expanded)                            
                cuda.synchronize()
            t2_expand_1 = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._expand_1_acp_thrifty() done; time: {t2_expand_1 - t1_expand_1} s]")
            t1_expand_2 = time.time()            
            trees_actions_expanded_flat = self._flatten_trees_actions_expanded_thrifty(trees_actions_expanded)
            bpg = trees_actions_expanded_flat.shape[0] # thrifty number of blocks                                
            tpb = self.tpb_e2
            if self.verbose_debug:
                print(f"[MCTSNC._expand_2_thrifty()...; bpg: {bpg}, tpb: {tpb}]")
            dev_trees_actions_expanded_flat = cuda.to_device(trees_actions_expanded_flat)
            MCTSNC._expand_2_thrifty[bpg, tpb](self.dev_trees, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_outcomes, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                               self.dev_trees_boards, self.dev_trees_extra_infos,                                               
                                               self.dev_trees_nodes_selected, dev_trees_actions_expanded_flat)
            cuda.synchronize()
            t2_expand_2 = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._expand_2_thrifty() done; time: {t2_expand_2 - t1_expand_2} s]")
            t2_expand = time.time()
            self.time_expand += t2_expand - t1_expand
            
            # playouts
            t1_playout = time.time()
            bpg = trees_actions_expanded_flat.shape[0] # thrifty number of blocks
            tpb = self.n_playouts
            if self.verbose_debug:
                print(f"[MCTSNC._playout_acp_thrifty()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._playout_acp_thrifty[bpg, tpb](self.dev_trees, self.dev_trees_turns, self.dev_trees_terminals, self.dev_trees_outcomes, 
                                                  self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                  self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, dev_trees_actions_expanded_flat,
                                                  self.dev_random_generators_playout, self.dev_trees_playout_outcomes, self.dev_trees_playout_outcomes_children)
            cuda.synchronize()
            t2_playout = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._playout_acp_thrifty() done; time: {t2_playout - t1_playout} s]")
            self.time_playout += t2_playout - t1_playout
            
            # backups
            t1_backup = time.time()
            t1_backup_1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_b1                     
            if self.verbose_debug:
                print(f"[MCTSNC._backup_acp_thrifty()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._backup_1_acp_thrifty[bpg, tpb](self.n_playouts, 
                                                   self.dev_trees, self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                   self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, self.dev_trees_playout_outcomes, self.dev_trees_playout_outcomes_children)
            cuda.synchronize()            
            t2_backup_1 = time.time()            
            if self.verbose_debug:
                print(f"[MCTSNC._backup_1_acp_thrifty() done; time: {t2_backup_1 - t1_backup_1} s]")            
            t1_backup_2 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_b2            
            if self.verbose_debug:
                print(f"[MCTSNC._backup_2_acp()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._backup_2_acp[bpg, tpb](self.n_playouts,
                                           self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                           self.dev_trees_selected_paths, self.dev_trees_actions_expanded, 
                                           self.dev_trees_playout_outcomes)
            cuda.synchronize()                                    
            t2_backup_2 = time.time()        
            if self.verbose_debug:
                print(f"[MCTSNC._backup_2_acp() done; time: {t2_backup_2 - t1_backup_2} s]")
            t2_backup = time.time()
            self.time_backup += t2_backup - t1_backup
            self.steps += 1
        self.time_loop = time.time() - t1_loop
                    
        # sum reduction over trees for each root action        
        t1_reduce_over_trees = time.time()
        root_actions_expanded = np.empty_like(self.dev_root_actions_expanded)
        self.dev_root_actions_expanded.copy_to_host(ary=root_actions_expanded)
        n_root_actions = int(root_actions_expanded[-1])  
        bpg = n_root_actions
        tpb = self.tpb_rot
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_trees_thrifty()...; bpg: {bpg}, tpb: {tpb}]")
        MCTSNC._reduce_over_trees_thrifty[bpg, tpb](self.dev_trees, self.dev_trees_terminals, self.dev_trees_outcomes,
                                                    self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                    self.dev_root_actions_expanded, root_turn,
                                                    self.dev_root_ns, self.dev_actions_win_flags, self.dev_actions_ns, self.dev_actions_ns_wins)
        cuda.synchronize()
        t2_reduce_over_trees = time.time()
        self.time_reduce_over_trees = t2_reduce_over_trees - t1_reduce_over_trees
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_trees_thrifty() done; time: {self.time_reduce_over_trees} s]")
            
        # max-argmax reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self.tpb_roa
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_actions_thrifty()...; bpg: {bpg}, tpb: {tpb}]")                                                
        MCTSNC._reduce_over_actions_thrifty[bpg, tpb](n_root_actions, 
                                                      self.dev_actions_win_flags, self.dev_actions_ns, self.dev_actions_ns_wins, 
                                                      self.dev_best_action, self.dev_best_win_flag, self.dev_best_n, self.dev_best_n_wins)        
        self.best_action = self.dev_best_action.copy_to_host()[0]
        self.best_win_flag = self.dev_best_win_flag.copy_to_host()[0]                
        self.best_n = self.dev_best_n.copy_to_host()[0]
        self.best_n_wins = self.dev_best_n_wins.copy_to_host()[0]
        self.best_q = self.best_n_wins / self.best_n if self.best_n > 0 else np.nan        
        cuda.synchronize()
        self.best_action = root_actions_expanded[self.best_action]
        t2_reduce_over_actions = time.time()
        self.time_reduce_over_actions = t2_reduce_over_actions - t1_reduce_over_actions 
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_actions_thrifty() done; time: {self.time_reduce_over_actions} s]")                
        t2 = time.time()
        self.time_total = t2 - t1
        
        if self.verbose_info:
            print(f"[actions info:\n{dict_to_str(self._make_actions_info_thrifty())}]")
            print(f"[performance info:\n{dict_to_str(self._make_performance_info())}]")
            
    def _run_acp_prodigal(self, root_board, root_extra_info, root_turn, forced_search_steps_limit=np.inf):
        """Runs computations for algorithmic variant: ``"acp_prodigal"``."""
        t1 = time.time()    
        
        # reset
        t1_reset = time.time()
        bpg = self.n_trees
        tpb = self.tpb_r
        dev_root_board = cuda.to_device(root_board)
        if root_extra_info is None:
            root_extra_info = np.zeros(1, dtype=np.int8) # fake extra info array        
        dev_root_extra_info = cuda.to_device(root_extra_info)
        if self.verbose_debug:
            print(f"[MCTSNC._reset()...; bpg: {bpg}, tpb: {tpb}]")                
        MCTSNC._reset[bpg, tpb](dev_root_board, dev_root_extra_info, root_turn, 
                                self.dev_trees, self.dev_trees_sizes, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                self.dev_trees_boards, self.dev_trees_extra_infos)
        cuda.synchronize()    
        t2_reset = time.time()
        if self.verbose_debug:
            print(f"[MCTSNC._reset() done; time: {t2_reset - t1_reset} s]")
        
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
            if self.verbose_debug:
                print(f"[step: {self.steps + 1} starting, time used so far: {t2_loop - t1_loop} s]")     
        
            # selections
            t1_select = time.time()
            bpg = self.n_trees
            tpb = self.tpb_s
            if self.verbose_debug:
                print(f"[MCTSNC._select()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._select[bpg, tpb](self.ucb_c, 
                                     self.dev_trees, self.dev_trees_leaves, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                     self.dev_trees_nodes_selected, self.dev_trees_selected_paths)
            cuda.synchronize()                     
            t2_select = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._select() done; time: {t2_select - t1_select} s]")
            self.time_select += t2_select - t1_select                                    
            
            # expansions
            t1_expand = time.time()                        
            t1_expand_1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_e1
            if self.verbose_debug:
                print(f"[MCTSNC._expand_1_acp_prodigal()...; bpg: {bpg}, tpb: {tpb}]")                         
            MCTSNC._expand_1_acp_prodigal[bpg, tpb](self.max_tree_size, 
                                                    self.dev_trees, self.dev_trees_sizes, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals,
                                                    self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                    self.dev_trees_nodes_selected, self.dev_trees_actions_expanded)                 
            cuda.synchronize()
            if self.steps == 0:                
                MCTSNC._memorize_root_actions_expanded[1, self.state_max_actions + 2](self.dev_trees_actions_expanded, self.dev_root_actions_expanded)                            
                cuda.synchronize()
            t2_expand_1 = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._expand_1_acp_prodigal() done; time: {t2_expand_1 - t1_expand_1} s]")                                
            t1_expand_2 = time.time()
            bpg = (self.n_trees, self.state_max_actions) # prodigal number of blocks    
            tpb = self.tpb_e2 
            if self.verbose_debug:
                print(f"[MCTSNC._expand_2_prodigal()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._expand_2_prodigal[bpg, tpb](self.dev_trees, self.dev_trees_depths, self.dev_trees_turns, self.dev_trees_leaves, self.dev_trees_terminals, self.dev_trees_outcomes, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                self.dev_trees_boards, self.dev_trees_extra_infos,                                               
                                                self.dev_trees_nodes_selected, self.dev_trees_actions_expanded)
            cuda.synchronize()            
            t2_expand_2 = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._expand_2_prodigal() done; time: {t2_expand_2 - t1_expand_2} s]")
            t2_expand = time.time()
            self.time_expand += t2_expand - t1_expand
                        
            # playouts
            t1_playout = time.time()
            bpg = (self.n_trees, self.state_max_actions) # prodigal number of blocks
            tpb = self.n_playouts
            if self.verbose_debug:
                print(f"[MCTSNC._playout_acp_prodigal()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._playout_acp_prodigal[bpg, tpb](self.dev_trees, self.dev_trees_turns, self.dev_trees_terminals, self.dev_trees_outcomes, 
                                                   self.dev_trees_boards, self.dev_trees_extra_infos, 
                                                   self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, 
                                                   self.dev_random_generators_playout, self.dev_trees_playout_outcomes, self.dev_trees_playout_outcomes_children)
            cuda.synchronize()
            t2_playout = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._playout_acp_prodigal() done; time: {t2_playout - t1_playout} s]")
            self.time_playout += t2_playout - t1_playout
            
            # backups
            t1_backup = time.time()
            t1_backup_1 = time.time()
            bpg = self.n_trees
            tpb = self.tpb_b1                    
            if self.verbose_debug:
                print(f"[MCTSNC._backup_1_acp_prodigal()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._backup_1_acp_prodigal[bpg, tpb](self.n_playouts, 
                                                    self.dev_trees, self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                    self.dev_trees_nodes_selected, self.dev_trees_actions_expanded, 
                                                    self.dev_trees_playout_outcomes, self.dev_trees_playout_outcomes_children)
            cuda.synchronize()            
            t2_backup_1 = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._backup_1_acp_prodigal() done; time: {t2_backup_1 - t1_backup_1} s]")            
            t1_backup_2 = time.time()
            bpg = self.n_trees            
            tpb = self.tpb_b2              
            if self.verbose_debug:
                print(f"[MCTSNC._backup_2_acp()...; bpg: {bpg}, tpb: {tpb}]")
            MCTSNC._backup_2_acp[bpg, tpb](self.n_playouts,
                                           self.dev_trees_turns, self.dev_trees_ns, self.dev_trees_ns_wins, 
                                           self.dev_trees_selected_paths, self.dev_trees_actions_expanded, 
                                           self.dev_trees_playout_outcomes)
            cuda.synchronize()                                    
            t2_backup_2 = time.time()
            if self.verbose_debug:
                print(f"[MCTSNC._backup_2_acp() done; time: {t2_backup_2 - t1_backup_2} s]")
            t2_backup = time.time()
            self.time_backup += t2_backup - t1_backup
                                                    
            self.steps += 1
        self.time_loop = time.time() - t1_loop
                                                        
        # sum reduction over trees
        t1_reduce_over_trees = time.time()
        bpg = self.state_max_actions
        tpb = self.tpb_rot
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_trees_prodigal()...; bpg: {bpg}, tpb: {tpb}]")            
        MCTSNC._reduce_over_trees_prodigal[bpg, tpb](self.dev_trees, self.dev_trees_terminals, self.dev_trees_outcomes, 
                                                     self.dev_trees_ns, self.dev_trees_ns_wins, 
                                                     self.dev_root_actions_expanded, root_turn, 
                                                     self.dev_root_ns, self.dev_actions_win_flags, self.dev_actions_ns, self.dev_actions_ns_wins)
        cuda.synchronize()
        t2_reduce_over_trees = time.time()
        self.time_reduce_over_trees = t2_reduce_over_trees - t1_reduce_over_trees
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_trees_prodigal() done; time: {self.time_reduce_over_trees} s]")                
                    
        # max-argmax reduction over root actions
        t1_reduce_over_actions = time.time() 
        bpg = 1
        tpb = self.tpb_roa
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_actions_prodigal()...; bpg: {bpg}, tpb: {tpb}]")                                        
        MCTSNC._reduce_over_actions_prodigal[bpg, tpb](self.dev_actions_win_flags, self.dev_actions_ns, self.dev_actions_ns_wins, 
                                                       self.dev_best_action, self.dev_best_win_flag, self.dev_best_n, self.dev_best_n_wins)        
        self.best_action = self.dev_best_action.copy_to_host()[0]
        self.best_win_flag = self.dev_best_win_flag.copy_to_host()[0]                
        self.best_n = self.dev_best_n.copy_to_host()[0]
        self.best_n_wins = self.dev_best_n_wins.copy_to_host()[0]
        self.best_q = self.best_n_wins / self.best_n if self.best_n > 0 else np.nan      
        cuda.synchronize()
        t2_reduce_over_actions = time.time() 
        self.time_reduce_over_actions = t2_reduce_over_actions - t1_reduce_over_actions                           
        if self.verbose_debug:
            print(f"[MCTSNC._reduce_over_actions_prodigal() done; time: {self.time_reduce_over_actions} s]")
        t2 = time.time()
        self.time_total = t2 - t1                            
                 
        if self.verbose_info:
            print(f"[actions info:\n{dict_to_str(self._make_actions_info_prodigal())}]")
            print(f"[performance info:\n{dict_to_str(self._make_performance_info())}]")                             

    @staticmethod
    @cuda.jit(void(int8[:, :], int8[:], int8, int32[:, :, :], int32[:], int16[:, :], int8[:, :], boolean[:, :], boolean[:, :], int32[:, :], int32[:, :], int8[:, :, :, :], int8[:, :, :]))
    def _reset(root_board, root_extra_info, root_turn, trees, trees_sizes, trees_depths, trees_turns, trees_leaves, trees_terminals, trees_ns, trees_ns_wins, trees_boards, trees_extra_infos):
        """CUDA kernel responsible for reseting root nodes of trees to new root state."""         
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
    def _select(ucb_c, trees, trees_leaves, trees_ns, trees_ns_wins, trees_nodes_selected, trees_selected_paths):
        """CUDA kernel responsible for computations of stage: selections."""
        shared_ucbs = cuda.shared.array(512, dtype=float32) # 512 - assumed limit on max actions
        shared_best_child = cuda.shared.array(512, dtype=int32) # 512 - assumed limit on max actions (array instead of one index due to max-argmax reduction pattern)
        shared_selected_path = cuda.shared.array(2048 + 2, dtype=int32) # 2048 - assumed equal to MAX_TREE_DEPTH 
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
                    shared_ucbs[t] = -float32(inf)
                else:
                    child_n = trees_ns[ti, child]             
                    if child_n == int32(0):
                        shared_ucbs[t] = float32(inf)
                    else:                        
                        shared_ucbs[t] = trees_ns_wins[ti, child] / float32(child_n) + ucb_c * math.sqrt(math.log(trees_ns[ti, node]) / child_n)
            else:
                shared_ucbs[t] = -float32(inf)
            cuda.syncthreads()
            stride = tpb >> 1 # half of tpb
            while stride > 0: # max-argmax reduction pattern
                if t < stride:
                    t_stride = t + stride
                    if shared_ucbs[t] < shared_ucbs[t_stride]:
                        shared_ucbs[t] = shared_ucbs[t_stride]
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
    def _expand_1_ocp_thrifty(max_tree_size, trees, trees_sizes, trees_turns, trees_leaves, trees_terminals, trees_boards, trees_extra_infos, 
                                   trees_nodes_selected, random_generators_expand_1, trees_actions_expanded):
        """CUDA kernel responsible for computations of stage: expansions (substage 1, variant ``"ocp_thrifty"``)."""
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_legal_actions = cuda.shared.array(512, dtype=boolean) # 512 - assumed limit on max actions
        shared_legal_actions_child_shifts = cuda.shared.array(512, dtype=int16) # 512 - assumed limit on max actions
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
        rand_child_for_playout = int16(-3) # remains like this when tree cannot grow due to memory exhausted
        if t < state_max_actions:
            shared_legal_actions_child_shifts[t] = int16(-1)
        if t == 0:
            if not selected_is_terminal:
                for i in range(state_max_actions):
                    if shared_legal_actions[i] and size_so_far + child_shift + 1 < max_tree_size:
                        child_shift += 1
                    shared_legal_actions_child_shifts[i] = child_shift                                
                if child_shift >= int16(0):
                    trees_actions_expanded[ti, -1] = child_shift + 1 # information how many children expanded (as last entry)
                    trees_leaves[ti, selected] = False                                
                    rand_child_for_playout = int16(xoroshiro128p_uniform_float32(random_generators_expand_1, t_global) * (child_shift + 1))
                else:
                    trees_actions_expanded[ti, -1] = int16(1) # tree not grown due to memory exhausted, but selected shall be played out (hence 1 needed)
                trees_actions_expanded[ti, -2] = rand_child_for_playout
            else:
                trees_actions_expanded[ti, -1] = int16(1) # terminal in fact not expanded, but shall be played out (hence 1 needed)
                trees_actions_expanded[ti, -2] = int16(-1) # fake child for playouts indicating that selected is terminal (and playouts computed from outcome)    
        cuda.syncthreads()
        if t < state_max_actions: 
            child_index = int32(-1)
            if shared_legal_actions[t]:
                child_shift = shared_legal_actions_child_shifts[t]
                if child_shift >= int16(0):
                    child_index = size_so_far + child_shift                
                    trees_actions_expanded[ti, child_shift] = t # for thrifty variants
            trees[ti, selected, 1 + t] = child_index # parent gets to know where child is 
        if t == 0:
            trees_sizes[ti] += shared_legal_actions_child_shifts[state_max_actions - 1] + 1 # updating tree size
        
    @staticmethod
    @cuda.jit(void(int32, int32[:, :, :], int32[:], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], xoroshiro128p_type[:], int16[:, :]))
    def _expand_1_ocp_prodigal(max_tree_size, trees, trees_sizes, trees_turns, trees_leaves, trees_terminals, trees_boards, trees_extra_infos, 
                                   trees_nodes_selected, random_generators_expand_1, trees_actions_expanded):
        """CUDA kernel responsible for computations of stage: expansions (substage 1, variant ``"ocp_prodigal"``)."""        
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_legal_actions = cuda.shared.array(512, dtype=boolean) # 512 - assumed limit on max actions
        shared_legal_actions_child_shifts = cuda.shared.array(512, dtype=int16) # 512 - assumed limit on max actions
        shared_map_child_shifts_to_action = cuda.shared.array(512, dtype=int16) # 512 - assumed limit on max actions
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
        rand_child_for_playout = int16(-3) # remains like this when tree cannot grow due to memory exhausted
        if t < state_max_actions:
            shared_legal_actions_child_shifts[t] = int16(-1)
        if t == 0:
            if not selected_is_terminal:
                for i in range(state_max_actions):
                    if shared_legal_actions[i] and size_so_far + child_shift + 1 < max_tree_size:
                        child_shift += 1
                        shared_map_child_shifts_to_action[child_shift] = i
                    shared_legal_actions_child_shifts[i] = child_shift                                                
                if child_shift >= int16(0):
                    trees_actions_expanded[ti, -1] = child_shift + 1 # information how many children expanded (as last entry)
                    trees_leaves[ti, selected] = False                                
                    rand_child_for_playout = int16(xoroshiro128p_uniform_float32(random_generators_expand_1, t_global) * (child_shift + 1))
                    rand_child_for_playout = shared_map_child_shifts_to_action[rand_child_for_playout]
                else:
                    trees_actions_expanded[ti, -1] = int16(1) # tree not grown due to memory exhausted, but selected shall be played out (hence 1 needed)
                trees_actions_expanded[ti, -2] = rand_child_for_playout
            else:
                trees_actions_expanded[ti, -1] = int16(1)
                trees_actions_expanded[ti, -2] = int16(-1) # fake child for playouts indicating that selected is terminal (and playouts computed from outcome)
        cuda.syncthreads()        
        if t < state_max_actions: 
            child_index = int32(-1)
            if shared_legal_actions[t]:
                child_shift = shared_legal_actions_child_shifts[t]
                if child_shift >= int16(0):
                    child_index = size_so_far + child_shift                
                    trees_actions_expanded[ti, t] = t # for prodigal variants
            else: 
                trees_actions_expanded[ti, t] = int16(-1) # for prodigal variants                 
            trees[ti, selected, 1 + t] = child_index # parent gets to know where child is 
        if t == 0:
            trees_sizes[ti] += shared_legal_actions_child_shifts[state_max_actions - 1] + 1 # updating tree size
        
    @staticmethod
    @cuda.jit(void(int32, int32[:, :, :], int32[:], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :]))
    def _expand_1_acp_thrifty(max_tree_size, trees, trees_sizes, trees_turns, trees_leaves, trees_terminals, trees_boards, trees_extra_infos, 
                           trees_nodes_selected, trees_actions_expanded):
        """CUDA kernel responsible for computations of stage: expansions (substage 1, variant ``"acp_thrifty"``)."""
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_legal_actions = cuda.shared.array(512, dtype=boolean) # 512 - assumed limit on max actions
        shared_legal_actions_child_shifts = cuda.shared.array(512, dtype=int16) # 512 - assumed limit on max actions
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
        fake_child_for_playout = int16(-3) # remains like this when tree cannot grow due to memory exhausted 
        if t < state_max_actions:
            shared_legal_actions_child_shifts[t] = int16(-1)
        if t == 0:
            if not selected_is_terminal:
                for i in range(state_max_actions):
                    if shared_legal_actions[i] and size_so_far + child_shift + 1 < max_tree_size:
                        child_shift += 1
                    shared_legal_actions_child_shifts[i] = child_shift
                if child_shift >= int16(0):
                    trees_actions_expanded[ti, -1] = child_shift + 1 # information how many children expanded (as last entry)
                    trees_leaves[ti, selected] = False
                    fake_child_for_playout = int16(-2) # indicates all children for playouts (acp)
                else:
                    trees_actions_expanded[ti, -1] = int16(1) # tree not grown due to memory exhausted, but selected shall be played out (hence 1 needed)
                trees_actions_expanded[ti, -2] = fake_child_for_playout
            else:
                trees_actions_expanded[ti, -1] = int16(1)
                trees_actions_expanded[ti, -2] = int16(-1) # fake child for playouts indicating that selected is terminal (and playouts computed from outcome)                
        cuda.syncthreads()
        if t < state_max_actions: 
            child_index = int32(-1)
            if shared_legal_actions[t]:
                child_shift = shared_legal_actions_child_shifts[t]
                if child_shift >= int16(0):
                    child_index = size_so_far + child_shift                
                    trees_actions_expanded[ti, child_shift] = t # for thrifty variants
            trees[ti, selected, 1 + t] = child_index # parent gets to know where child is 
        if t == 0:
            trees_sizes[ti] += shared_legal_actions_child_shifts[state_max_actions - 1] + 1 # updating tree size
            if selected_is_terminal or fake_child_for_playout == int16(-3):
                trees_actions_expanded[ti, 0] = int16(0) # fake legal action for playout (so that exactly one block becomes executed in full body)

    @staticmethod
    @cuda.jit(void(int32, int32[:, :, :], int32[:], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :]))
    def _expand_1_acp_prodigal(max_tree_size, trees, trees_sizes, trees_turns, trees_leaves, trees_terminals, trees_boards, trees_extra_infos, 
                                    trees_nodes_selected, trees_actions_expanded):
        """CUDA kernel responsible for computations of stage: expansions (substage 1, variant ``"acp_prodigal"``)."""
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_legal_actions = cuda.shared.array(512, dtype=boolean) # 512 - assumed limit on max actions
        shared_legal_actions_child_shifts = cuda.shared.array(512, dtype=int16) # 512 - assumed limit on max actions
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
        fake_child_for_playout = int16(-3) # remains like this when tree cannot grow due to memory exhausted
        if t < state_max_actions:
            shared_legal_actions_child_shifts[t] = int16(-1)
        if t == 0:
            if not selected_is_terminal:
                for i in range(state_max_actions):
                    if shared_legal_actions[i] and size_so_far + child_shift + 1 < max_tree_size:
                        child_shift += 1
                    shared_legal_actions_child_shifts[i] = child_shift
                if child_shift >= int16(0):
                    trees_actions_expanded[ti, -1] = child_shift + 1 # information how many children expanded (as last entry)
                    trees_leaves[ti, selected] = False
                    fake_child_for_playout = int16(-2) # indicates all children for playouts (acp)
                else:
                    trees_actions_expanded[ti, -1] = int16(1) # tree not grown due to memory exhausted, but selected shall be played out (hence 1 needed)                                
                trees_actions_expanded[ti, -2] = fake_child_for_playout                                
            else:                
                trees_actions_expanded[ti, -1] = int16(1)
                trees_actions_expanded[ti, -2] = int16(-1) # fake child for playouts indicating that selected is terminal (and playouts computed from outcome)                                 
        cuda.syncthreads()        
        if t < state_max_actions: 
            child_index = int32(-1)
            if shared_legal_actions[t]:
                child_shift = shared_legal_actions_child_shifts[t]
                if child_shift >= int16(0):
                    child_index = size_so_far + child_shift
                    trees_actions_expanded[ti, t] = t # for prodigal variants
                else:
                    trees_actions_expanded[ti, t] = int16(-1) # tree not grown case
            else: 
                trees_actions_expanded[ti, t] = int16(-1) # for prodigal variants             
            trees[ti, selected, 1 + t] = child_index # parent gets to know where child is 
        if t == 0:
            trees_sizes[ti] += shared_legal_actions_child_shifts[state_max_actions - 1] + 1 # updating tree size
            if selected_is_terminal or fake_child_for_playout == int16(-3):
                trees_actions_expanded[ti, 0] = int16(0) # fake legal action for playout (so that exactly one block becomes executed in full body)
                
    @staticmethod
    @cuda.jit(void(int16[:, :], int16[:]))
    def _memorize_root_actions_expanded(dev_trees_actions_expanded, dev_root_actions_expanded):
        """CUDA kernel responsible for memorizing actions expanded at root node(s)."""
        t = cuda.threadIdx.x
        dev_root_actions_expanded[t] = dev_trees_actions_expanded[0, t]                
        
    @staticmethod
    @cuda.jit(void(int32[:, :, :], int16[:, :], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :], int32[:, :], int32[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :]))
    def _expand_2_thrifty(trees, trees_depths, trees_turns, trees_leaves, trees_terminals, trees_outcomes, trees_ns, trees_ns_wins, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded_flat):
        """CUDA kernel responsible for computations of stage: expansions (substage 2, thrifty number of blocks - variant ``"ocp_thrifty"`` or ``"acp_thrifty"``)."""
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        tai = cuda.blockIdx.x # tree-action pair index
        ti = trees_actions_expanded_flat[tai, 0]
        action = trees_actions_expanded_flat[tai, 1]
        if action < int16(0):
            return # selected is terminal or tree not grown due to memory exhausted          
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
            if outcome == int8(-1) or outcome == int8(0) or outcome == int8(1):
                terminal_flag = True
            trees_terminals[ti, child] = terminal_flag
            trees_outcomes[ti, child] = outcome
            trees_ns[ti, child] = int32(0)
            trees_ns_wins[ti, child] = int32(0)
            trees_depths[ti, child] = trees_depths[ti, selected] + 1
            
    @staticmethod
    @cuda.jit(void(int32[:, :, :], int16[:, :], int8[:, :], boolean[:, :], boolean[:, :], int8[:, :], int32[:, :], int32[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :]))
    def _expand_2_prodigal(trees, trees_depths, trees_turns, trees_leaves, trees_terminals, trees_outcomes, trees_ns, trees_ns_wins, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded):
        """CUDA kernel responsible for computations of stage: expansions (substage 2, prodigal number of blocks - variant ``"ocp_prodigal"`` or ``"acp_prodigal"``)."""
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        ti = cuda.blockIdx.x
        action = cuda.blockIdx.y
        if trees_actions_expanded[ti, action] < int16(0): 
            return # prodigality
        if trees_actions_expanded[ti, -2] == int16(-1) or trees_actions_expanded[ti, -2] == int16(-3): 
            return # selected is terminal or tree cannot grow due to memory exhausted
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
        turn = int8(0)
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
            if outcome == int8(-1) or outcome == int8(0) or outcome == int8(1):
                terminal_flag = True
            trees_terminals[ti, child] = terminal_flag
            trees_outcomes[ti, child] = outcome
            trees_ns[ti, child] = int32(0)
            trees_ns_wins[ti, child] = int32(0)
            trees_depths[ti, child] = trees_depths[ti, selected] + 1                                                
                            
    @staticmethod
    @cuda.jit(void(int32[:, :, :], int8[:, :], boolean[:, :], int8[:, :], int8[:, :, :, :], int8[:, :, :], int32[:], int16[:, :], xoroshiro128p_type[:], int32[:, :]))
    def _playout_ocp(trees, trees_turns, trees_terminals, trees_outcomes, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded, random_generators_playout, trees_playout_outcomes):
        """CUDA kernel responsible for computations of stage: playouts (variant ``"ocp_thrifty"`` or ``"ocp_prodigal"``)."""
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_playout_outcomes = cuda.shared.array((512, 2), dtype=int16) # 512 - assumed max tpb for playouts, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout 
        local_board = cuda.local.array((32, 32), dtype=int8)
        local_extra_info = cuda.local.array(4096, dtype=int8)
        local_legal_actions_with_count = cuda.local.array(512 + 1, dtype=int16) # 512 - assumed limit on max actions
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        to_be_played_out = trees_nodes_selected[ti] # temporarily to_be_played_out equals selected
        rand_child_for_playout = trees_actions_expanded[ti, -2]
        last_action = int16(-1) # none yet
        if rand_child_for_playout >= int16(0): # check if some child picked on random for playouts
            last_action = trees_actions_expanded[ti, rand_child_for_playout]
            to_be_played_out = trees[ti, to_be_played_out, 1 + last_action]
        if trees_terminals[ti, to_be_played_out]: # root for playouts has been discovered terminal before (by game rules) -> taking stored outcome ("multiplied" by tpb)
            if t == 0:            
                outcome = trees_outcomes[ti, to_be_played_out]
                trees_playout_outcomes[ti, 0] = int32(tpb) if outcome == int8(-1) else int32(0) # wins of -1
                trees_playout_outcomes[ti, 1] = int32(tpb) if outcome == int8(1) else int32(0) # wins of +1
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
                    shared_board[i, j] = trees_boards[ti, to_be_played_out, i, j]
                e += tpb        
            _, _, extra_info_memory = trees_extra_infos.shape
            eipt = (extra_info_memory + tpb - 1) // tpb
            e = t
            for _ in range(eipt):
                if e < extra_info_memory:
                    shared_extra_info[e] = trees_extra_infos[ti, to_be_played_out, e]
                e += tpb
            cuda.syncthreads()
            for i in range(m):
                for j in range(n):
                    local_board[i, j] = shared_board[i, j]
            for i in range(extra_info_memory):
                local_extra_info[i] = shared_extra_info[i]                
            local_legal_actions_with_count[-1] = 0
            turn = trees_turns[ti, to_be_played_out]
            outcome = compute_outcome(m, n, local_board, local_extra_info, turn, last_action) if last_action != int16(-1) else int8(2) # else case only when trees not grown due to memory limit (then selected played out)
            while True: # playout loop                
                if not (outcome == int8(-1) or outcome == int8(0) or outcome == int8(1)): # indecisive, game ongoing
                    legal_actions_playout(m, n, local_board, local_extra_info, turn, local_legal_actions_with_count)
                    count = local_legal_actions_with_count[-1]
                    action_ord = int16(xoroshiro128p_uniform_float32(random_generators_playout, t_global) * count)
                    last_action = local_legal_actions_with_count[action_ord]
                    take_action_playout(m, n, local_board, local_extra_info, turn, last_action, action_ord, local_legal_actions_with_count)                    
                    turn = -turn
                else:
                    if outcome != int8(0):
                        shared_playout_outcomes[t, (outcome + 1) // 2] = int8(1)
                    break
                outcome = compute_outcome(m, n, local_board, local_extra_info, turn, last_action)
            cuda.syncthreads()
            stride = tpb >> 1 # half of tpb
            while stride > 0: # sum reduction pattern
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
    def _playout_acp_thrifty(trees, trees_turns, trees_terminals, trees_outcomes, trees_boards, trees_extra_infos, trees_nodes_selected, trees_actions_expanded, trees_actions_expanded_flat, random_generators_playout, trees_playout_outcomes, 
                             trees_playout_outcomes_children):
        """CUDA kernel responsible for computations of stage: playouts (variant ``"acp_thrifty"``)."""
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_playout_outcomes = cuda.shared.array((512, 2), dtype=int16) # 1024 - assumed max tpb for playouts, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout 
        local_board = cuda.local.array((32, 32), dtype=int8)
        local_extra_info = cuda.local.array(4096, dtype=int8)
        local_legal_actions_with_count = cuda.local.array(512 + 1, dtype=int16) # 512 - assumed limit on max actions        
        tai = cuda.blockIdx.x # tree-action pair index
        ti = trees_actions_expanded_flat[tai, 0]
        action = trees_actions_expanded_flat[tai, 1]  
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        to_be_played_out = trees_nodes_selected[ti] # temporarily to_be_played_out equals selected
        fake_child_for_playout = trees_actions_expanded[ti, -2]
        last_action = int16(-1) # none yet
        if fake_child_for_playout == int16(-2): # check if playouts are to be made on all children of selected
            last_action = action
            to_be_played_out = trees[ti, to_be_played_out, 1 + last_action]
        if trees_terminals[ti, to_be_played_out]: # root for playouts has been discovered terminal before (by game rules) -> taking stored outcome ("multiplied" by tpb)
            if t == 0:
                outcome = trees_outcomes[ti, to_be_played_out]
                if fake_child_for_playout == int16(-2): # case where terminal is one child among all children of selected node 
                    trees_playout_outcomes_children[ti, action, 0] = int32(tpb) if outcome == int8(-1) else int32(0) # wins of -1
                    trees_playout_outcomes_children[ti, action, 1] = int32(tpb) if outcome == int8(1) else int32(0) # wins of +1
                else: # case where terminal was selected
                    trees_playout_outcomes[ti, 0] = int32(tpb) if outcome == int8(-1) else int32(0) # wins of -1
                    trees_playout_outcomes[ti, 1] = int32(tpb) if outcome == int8(1) else int32(0) # wins of +1
        else:
            t = cuda.threadIdx.x
            state_max_actions = trees.shape[2] - 1
            t_global = ti * state_max_actions * tpb + action * tpb + t # purposely (instead of t_global = cuda.grid(1)) to make resutls of acp_prodigal and acp_thrifty same (for equal number of steps)
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
                    shared_board[i, j] = trees_boards[ti, to_be_played_out, i, j]
                e += tpb        
            _, _, extra_info_memory = trees_extra_infos.shape
            eipt = (extra_info_memory + tpb - 1) // tpb
            e = t
            for _ in range(eipt):
                if e < extra_info_memory:
                    shared_extra_info[e] = trees_extra_infos[ti, to_be_played_out, e]
                e += tpb
            cuda.syncthreads()
            for i in range(m):
                for j in range(n):
                    local_board[i, j] = shared_board[i, j]
            for i in range(extra_info_memory):
                local_extra_info[i] = shared_extra_info[i]
            local_legal_actions_with_count[-1] = 0            
            turn = trees_turns[ti, to_be_played_out]
            outcome = compute_outcome(m, n, local_board, local_extra_info, turn, last_action) if last_action != int16(-1) else int8(2) # else case only when trees not grown due to memory limit (then selected played out)
            while True: # playout loop                
                if not (outcome == int8(-1) or outcome == int8(0) or outcome == int8(1)): # indecisive, game ongoing
                    legal_actions_playout(m, n, local_board, local_extra_info, turn, local_legal_actions_with_count)
                    count = local_legal_actions_with_count[-1]
                    action_ord = int16(xoroshiro128p_uniform_float32(random_generators_playout, t_global) * count)
                    last_action = local_legal_actions_with_count[action_ord]
                    take_action_playout(m, n, local_board, local_extra_info, turn, last_action, action_ord, local_legal_actions_with_count)
                    turn = -turn
                else:
                    if outcome != int8(0):
                        shared_playout_outcomes[t, (outcome + 1) // 2] = int8(1)
                    break
                outcome = compute_outcome(m, n, local_board, local_extra_info, turn, last_action)
            cuda.syncthreads()
            stride = tpb >> 1 # half of tpb
            while stride > 0: # sum reduction pattern
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
        """CUDA kernel responsible for computations of stage: playouts (variant ``"acp_prodigal"``)."""
        shared_board = cuda.shared.array((32, 32), dtype=int8) # assumed max board size (for selected node in tree associated with block)
        shared_extra_info = cuda.shared.array(4096, dtype=int8) # 4096 - assumed limit on max extra info
        shared_playout_outcomes = cuda.shared.array((512, 2), dtype=int16) # 1024 - assumed max tpb for playouts, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout        
        ti = cuda.blockIdx.x
        action = cuda.blockIdx.y
        if trees_actions_expanded[ti, action] < int16(0): # prodigality
            return
        local_board = cuda.local.array((32, 32), dtype=int8)
        local_extra_info = cuda.local.array(4096, dtype=int8)
        local_legal_actions_with_count = cuda.local.array(512 + 1, dtype=int16) # 512 - assumed limit on max actions          
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x
        to_be_played_out = trees_nodes_selected[ti] # temporarily to_be_played_out equals selected  
        fake_child_for_playout = trees_actions_expanded[ti, -2]
        last_action = int16(-1) # none yet
        if fake_child_for_playout == int16(-2): # check if playouts are to be made on all children of selected
            last_action = action
            to_be_played_out = trees[ti, to_be_played_out, 1 + last_action] # moving one level down from selected
        if trees_terminals[ti, to_be_played_out]: # root for playouts has been discovered terminal before (by game rules) -> taking stored outcome ("multiplied" by tpb)
            if t == 0:
                outcome = trees_outcomes[ti, to_be_played_out]
                if fake_child_for_playout == int16(-2): # case where terminal is one child among all children of selected node             
                    trees_playout_outcomes_children[ti, action, 0] = int32(tpb) if outcome == int8(-1) else int32(0) # wins of -1
                    trees_playout_outcomes_children[ti, action, 1] = int32(tpb) if outcome == int8(1) else int32(0) # wins of +1
                else: # case where terminal was selected
                    trees_playout_outcomes[ti, 0] = int32(tpb) if outcome == int8(-1) else int32(0) # wins of -1
                    trees_playout_outcomes[ti, 1] = int32(tpb) if outcome == int8(1) else int32(0) # wins of +1                                
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
                    shared_board[i, j] = trees_boards[ti, to_be_played_out, i, j]
                e += tpb        
            _, _, extra_info_memory = trees_extra_infos.shape
            eipt = (extra_info_memory + tpb - 1) // tpb
            e = t
            for _ in range(eipt):
                if e < extra_info_memory:
                    shared_extra_info[e] = trees_extra_infos[ti, to_be_played_out, e]
                e += tpb
            cuda.syncthreads()
            for i in range(m):
                for j in range(n):
                    local_board[i, j] = shared_board[i, j]
            for i in range(extra_info_memory):
                local_extra_info[i] = shared_extra_info[i]
            local_legal_actions_with_count[-1] = 0
            turn = trees_turns[ti, to_be_played_out]
            outcome = compute_outcome(m, n, local_board, local_extra_info, turn, last_action) if last_action != int16(-1) else int8(2) # else case only when trees not grown due to memory limit (then selected played out)
            while True: # playout loop                
                if not (outcome == int8(-1) or outcome == int8(0) or outcome == int8(1)): # indecisive, game ongoing
                    legal_actions_playout(m, n, local_board, local_extra_info, turn, local_legal_actions_with_count)
                    count = local_legal_actions_with_count[-1]
                    action_ord = int16(xoroshiro128p_uniform_float32(random_generators_playout, t_global) * count)
                    last_action = local_legal_actions_with_count[action_ord]
                    take_action_playout(m, n, local_board, local_extra_info, turn, last_action, action_ord, local_legal_actions_with_count)
                    turn = -turn
                else:                                                
                    if outcome != int8(0):
                        shared_playout_outcomes[t, (outcome + 1) // 2] = int8(1)
                    break
                outcome = compute_outcome(m, n, local_board, local_extra_info, turn, last_action)
            cuda.syncthreads()
            stride = tpb >> 1 # half of tpb
            while stride > 0: # sum reduction pattern
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
    @cuda.jit(void(int16, int32[:, :, :], int8[:, :], int32[:, :], int32[:, :], int32[:], int32[:, :], int16[:, :], int32[:, :]))
    def _backup_ocp(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_selected_paths, trees_actions_expanded, trees_playout_outcomes):
        """CUDA kernel responsible for computations of stage: backups (variant ``"ocp_thrifty"`` or ``"ocp_prodigal"``)."""        
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
                node = trees[ti, node, 1 + last_action]
                trees_ns[ti, node] += n_playouts
                if trees_turns[ti, node] == int8(1):
                    trees_ns_wins[ti, node] += n_negative_wins 
                else:
                    trees_ns_wins[ti, node] += n_positive_wins
                    
    @staticmethod
    @cuda.jit(void(int16, int32[:, :, :], int8[:, :], int32[:, :], int32[:, :], int32[:], int16[:, :], int32[:, :], int32[:, :, :]))
    def _backup_1_acp_thrifty(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_actions_expanded, trees_playout_outcomes, trees_playout_outcomes_children):
        """CUDA kernel responsible for computations of stage: backups (substage 1, variant ``"acp_thrifty"``)."""
        shared_playout_outcomes_children = cuda.shared.array((512, 2), dtype=int32) # 512 - assumed limit on max actions, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout 
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x  
        fake_child_for_playout = trees_actions_expanded[ti, -2]
        if fake_child_for_playout == int16(-2): # check if actual children of selected were played out
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
            stride = tpb >> 1 # half of tpb
            while stride > 0: # sum reduction pattern                
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
    def _backup_1_acp_prodigal(n_playouts, trees, trees_turns, trees_ns, trees_ns_wins, trees_nodes_selected, trees_actions_expanded, trees_playout_outcomes, trees_playout_outcomes_children):
        """CUDA kernel responsible for computations of stage: backups (substage 1, variant ``"acp_prodigal"``)."""
        shared_playout_outcomes_children = cuda.shared.array((512, 2), dtype=int32) # 512 - assumed limit on max actions, two cells for a row (-1 win, +1 win), each flagged by 0 or 1 after playout 
        ti = cuda.blockIdx.x # tree index
        tpb = cuda.blockDim.x
        t = cuda.threadIdx.x        
        fake_child_for_playout = trees_actions_expanded[ti, -2]
        max_actions = trees_actions_expanded.shape[1] - 2
        if fake_child_for_playout == int16(-2): # check if actual children of selected were played out
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
            stride = tpb >> 1 # half of tpb
            while stride > 0: # sum reduction pattern
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
    def _backup_2_acp(n_playouts, trees_turns, trees_ns, trees_ns_wins, trees_selected_paths, trees_actions_expanded, trees_playout_outcomes):
        """CUDA kernel responsible for computations of stage: backups (substage 2, variant ``"acp_thrifty"`` or ``"acp_prodigal"``)."""
        ti = cuda.blockIdx.x
        t = cuda.threadIdx.x
        tpb = cuda.blockDim.x
        n_negative_wins = trees_playout_outcomes[ti, 0]
        n_positive_wins = trees_playout_outcomes[ti, 1]
        n_expanded_actions = trees_actions_expanded[ti, -1]
        if n_expanded_actions == int16(0): # terminal was being "played out"
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
    @cuda.jit(void(int32[:, :, :], boolean[:, :], int8[:, :], int32[:, :], int32[:, :], int16[:], int8, int64[:], boolean[:], int64[:], int64[:]))
    def _reduce_over_trees_thrifty(trees, trees_terminals, trees_outcomes, trees_ns, trees_ns_wins, root_actions_expanded, root_turn, root_ns, actions_win_flags, actions_ns, actions_ns_wins):
        """CUDA kernel responsible for sum-reduction over trees (thrifty number of blocks, variant ``ocp_thrifty`` or ``acp_thrifty``)."""
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
        while stride > 0: # sum reduction pattern
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
            action_node = trees[0, 0, 1 + action] 
            actions_win_flags[b] = action_node != int32(-1) and trees_terminals[0, action_node] and trees_outcomes[0, action_node] == root_turn            
            
    @staticmethod
    @cuda.jit(void(int32[:, :, :], boolean[:, :], int8[:, :], int32[:, :], int32[:, :], int16[:], int8, int64[:], boolean[:], int64[:], int64[:]))
    def _reduce_over_trees_prodigal(trees, trees_terminals, trees_outcomes, trees_ns, trees_ns_wins, root_actions_expanded, root_turn, root_ns, actions_win_flags, actions_ns, actions_ns_wins):
        """CUDA kernel responsible for sum-reduction over trees (prodigal number of blocks, variant ``ocp_prodigal`` or ``acp_prodigal``)."""
        shared_root_ns = cuda.shared.array(512, dtype=int64) # 512 - assumed max of n_trees
        shared_actions_ns = cuda.shared.array(512, dtype=int64)
        shared_actions_ns_wins = cuda.shared.array(512, dtype=int64)
        b = cuda.blockIdx.x
        action = b # action index
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
            while stride > 0: # sum reduction pattern
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
            action_node = trees[0, 0, 1 + b] 
            actions_win_flags[b] = action_node != int32(-1) and trees_terminals[0, action_node] and trees_outcomes[0, action_node] == root_turn            
            
    @staticmethod
    @cuda.jit(void(int16, boolean[:], int64[:], int64[:], int16[:], boolean[:], int64[:], int64[:]))
    def _reduce_over_actions_thrifty(n_root_actions, actions_win_flags, actions_ns, actions_ns_wins, best_action, best_win_flag, best_n, best_n_wins):
        """CUDA kernel responsible for max/argmax-reduction over actions (thrifty number of blocks, variant ``ocp_thrifty`` or ``acp_thrifty``)."""
        shared_actions = cuda.shared.array(512, dtype=int16) # 512 - assumed max state actions
        shared_actions_win_flags = cuda.shared.array(512, dtype=boolean) 
        shared_actions_ns = cuda.shared.array(512, dtype=int64) 
        shared_actions_ns_wins = cuda.shared.array(512, dtype=int64)
        tpb = cuda.blockDim.x
        a = cuda.threadIdx.x # action index
        shared_actions[a] = a
        if a < n_root_actions:
            shared_actions_win_flags[a] = actions_win_flags[a]
            shared_actions_ns[a] = actions_ns[a]
            shared_actions_ns_wins[a] = actions_ns_wins[a]            
        else:
            shared_actions_win_flags[a] = False
            shared_actions_ns[a] = int64(0)
            shared_actions_ns_wins[a] = int64(0)         
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # max-argmax reduction pattern
            if a < stride:
                a_stride = a + stride
                if (shared_actions_win_flags[a] < shared_actions_win_flags[a_stride]) or\
                 ((shared_actions_win_flags[a] == shared_actions_win_flags[a_stride]) and (shared_actions_ns[a] < shared_actions_ns[a_stride])) or\
                 ((shared_actions_win_flags[a] == shared_actions_win_flags[a_stride]) and (shared_actions_ns[a] == shared_actions_ns[a_stride]) and (shared_actions_ns_wins[a] < shared_actions_ns_wins[a_stride])):
                    shared_actions[a] = shared_actions[a_stride]                                
                    shared_actions_ns[a] = shared_actions_ns[a_stride]
                    shared_actions_ns_wins[a] = shared_actions_ns_wins[a_stride]                    
                    shared_actions_win_flags[a] = shared_actions_win_flags[a_stride]     
            cuda.syncthreads()
            stride >>= 1
        if a == 0:            
            best_action[0] = shared_actions[0]
            best_win_flag[0] = shared_actions_win_flags[0]
            best_n[0] = shared_actions_ns[0]
            best_n_wins[0] = shared_actions_ns_wins[0]

    @staticmethod
    @cuda.jit(void(boolean[:], int64[:], int64[:], int16[:], boolean[:], int64[:], int64[:]))
    def _reduce_over_actions_prodigal(actions_win_flags, actions_ns, actions_ns_wins, best_action, best_win_flag, best_n, best_n_wins):
        """CUDA kernel responsible for max/argmax-reduction over actions (prodigal number of blocks, variant ``ocp_prodigal`` or ``acp_prodigal``)."""
        shared_actions = cuda.shared.array(512, dtype=int16) # 512 - assumed max state actions
        shared_actions_win_flags = cuda.shared.array(512, dtype=boolean) 
        shared_actions_ns = cuda.shared.array(512, dtype=int64) 
        shared_actions_ns_wins = cuda.shared.array(512, dtype=int64)
        tpb = cuda.blockDim.x
        a = cuda.threadIdx.x # action index        
        shared_actions[a] = a 
        state_max_actions = actions_ns.size
        if a < state_max_actions:
            shared_actions_win_flags[a] = actions_win_flags[a]
            shared_actions_ns[a] = actions_ns[a]
            shared_actions_ns_wins[a] = actions_ns_wins[a]                                                
        else:
            shared_actions_win_flags[a] = False
            shared_actions_ns[a] = int64(0)
            shared_actions_ns_wins[a] = int64(0)                                  
        cuda.syncthreads()
        stride = tpb >> 1 # half of tpb
        while stride > 0: # max-argmax reduction pattern
            if a < stride:
                a_stride = a + stride
                if (shared_actions_win_flags[a] < shared_actions_win_flags[a_stride]) or\
                 ((shared_actions_win_flags[a] == shared_actions_win_flags[a_stride]) and (shared_actions_ns[a] < shared_actions_ns[a_stride])) or\
                 ((shared_actions_win_flags[a] == shared_actions_win_flags[a_stride]) and (shared_actions_ns[a] == shared_actions_ns[a_stride]) and (shared_actions_ns_wins[a] < shared_actions_ns_wins[a_stride])):
                    shared_actions[a] = shared_actions[a_stride]                                
                    shared_actions_ns[a] = shared_actions_ns[a_stride]
                    shared_actions_ns_wins[a] = shared_actions_ns_wins[a_stride]                    
                    shared_actions_win_flags[a] = shared_actions_win_flags[a_stride]     
            cuda.syncthreads()
            stride >>= 1
        if a == 0:
            best_action[0] = shared_actions[0]
            best_win_flag[0] = shared_actions_win_flags[0]
            best_n[0] = shared_actions_ns[0]
            best_n_wins[0] = shared_actions_ns_wins[0]                    
            
    def _json_dump(self, fname):
        """Dumps (saves) device-side arrays, copied to host, representing trees and MCTS elements from the last run to a text file in json format."""        
        if self.verbose_info:
            print(f"JSON DUMP... [to file: {fname}]")
        t1 = time.time()                
        d = {}
                
        d["n_trees"] = self.n_trees
        d["n_playouts"] = self.n_playouts
        d["variant"] = self.variant
        d["search_time_limit"] = self.search_time_limit if self.search_time_limit < np.inf else "inf" 
        d["search_steps_limit"] = self.search_steps_limit if self.search_steps_limit < np.inf else "inf"
        d["ucb_c"] = self.ucb_c
        d["seed"] = self.seed
        d["device_memory"] = self.device_memory
        
        trees_sizes = np.empty_like(self.dev_trees_sizes)
        self.dev_trees_sizes.copy_to_host(ary=trees_sizes)
        tree_size_max = np.max(trees_sizes)                        
        
        trees = np.empty_like(self.dev_trees)        
        self.dev_trees.copy_to_host(ary=trees)
        trees = trees[:, :tree_size_max, :]        
        
        trees_depths = np.empty_like(self.dev_trees_depths)
        self.dev_trees_depths.copy_to_host(ary=trees_depths)
        trees_depths = trees_depths[:, :tree_size_max]
        depth_max = -np.inf
        for i in range(self.n_trees):
            depth_max = max(depth_max, np.max(trees_depths[i, :trees_sizes[i]]))

        trees_turns = np.empty_like(self.dev_trees_turns)        
        self.dev_trees_turns.copy_to_host(ary=trees_turns)
        trees_turns = trees_turns[:, :tree_size_max]

        trees_ns = np.empty_like(self.dev_trees_ns)        
        self.dev_trees_ns.copy_to_host(ary=trees_ns)
        trees_ns = trees_ns[:, :tree_size_max]

        trees_ns_wins = np.empty_like(self.dev_trees_ns_wins)        
        self.dev_trees_ns_wins.copy_to_host(ary=trees_ns_wins)
        trees_ns_wins = trees_ns_wins[:, :tree_size_max]
        
        trees_nodes_selected = np.empty_like(self.dev_trees_nodes_selected)
        self.dev_trees_nodes_selected.copy_to_host(ary=trees_nodes_selected)    

        trees_selected_paths = np.empty_like(self.dev_trees_selected_paths)
        self.dev_trees_selected_paths.copy_to_host(ary=trees_selected_paths)
        tmp_trees_selected_paths = trees_selected_paths[:, :depth_max + 2];
        tmp_trees_selected_paths[:, -1] = trees_selected_paths[:, -1]
        trees_selected_paths = tmp_trees_selected_paths
        
        trees_actions_expanded = np.empty_like(self.dev_trees_actions_expanded)
        self.dev_trees_actions_expanded.copy_to_host(ary=trees_actions_expanded)
        
        trees_playout_outcomes = np.empty_like(self.dev_trees_playout_outcomes)
        self.dev_trees_playout_outcomes.copy_to_host(ary=trees_playout_outcomes)
        
        trees_playout_outcomes_children = None
        if self.dev_trees_playout_outcomes_children is not None:
            trees_playout_outcomes_children = np.empty_like(self.dev_trees_playout_outcomes_children)
            self.dev_trees_playout_outcomes_children.copy_to_host(ary=trees_playout_outcomes_children)
        
        d["trees"] = trees.tolist()
        d["trees"] = trees.tolist()
        d["trees_sizes"] = trees_sizes.tolist()
        d["trees_depths"] = trees_depths.tolist()
        d["trees_turns"] = trees_turns.tolist()
        d["trees_ns"] = trees_ns.tolist()
        d["trees_ns_wins"] = trees_ns_wins.tolist()
        d["trees_nodes_selected"] = trees_nodes_selected.tolist()
        d["trees_selected_paths"] = trees_selected_paths.tolist()
        d["trees_actions_expanded"] = trees_actions_expanded.tolist()
        d["trees_playout_outcomes"] = trees_playout_outcomes.tolist()
        if trees_playout_outcomes_children is not None:
            d["trees_playout_outcomes_children"] = trees_playout_outcomes_children.tolist()    
        
        try:
            f = open(fname, "w+")
            json.dump(d, f, indent=2)
            f.close()
        except IOError:
            sys.exit(f"[error occurred when trying to dump MCTSNC as json to file: {fname}]")
        t2 = time.time()
        if self.verbose_info:
            print(f"JSON DUMP DONE. [time: {t2 - t1} s]")