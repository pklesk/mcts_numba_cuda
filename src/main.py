"""
Main script to carry out experiments with MCTS-NC project, i.e., matches of multiple games played by AIs (or human vs AI), using Monte Carlo Tree Search algorithm.
AIs can be instances of class ``MCTSNC`` from :doc:`mctsnc` representing the CUDA-based MCTS implementation, 
or instances of class ``MCTS`` from :doc:`mcts` representing the standard CPU-based (single-threaded) implementation serving as reference;
or ``None``s for human players.  

The following variables allow to define the settings of an experiment:

.. code-block:: python
    
    # main settings
    STATE_CLASS = C4 # C4 or Gomoku
    N_GAMES = 10
    AI_A_SHORTNAME = None # human
    AI_B_SHORTNAME = "mctsnc_5_inf_4_256_acp_prodigal" 
    REPRODUCE_EXPERIMENT = False
 
String names of predefined AI instances can be found in dictionary named ``AIS``.

Link to project repository
--------------------------
`https://github.com/pklesk/mcts_numba_cuda <https://github.com/pklesk/mcts_numba_cuda>`_
"""

import numpy as np
from mcts import MCTS
from mctsnc import MCTSNC
from c4 import C4
from gomoku import Gomoku
from game_runner import GameRunner
import time
from utils import cpu_and_system_props, gpu_props, dict_to_str, Logger, experiment_hash_str, save_and_zip_experiment, unzip_and_load_experiment
import sys

__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

# main settings
STATE_CLASS = C4 # C4 or Gomoku
N_GAMES = 10
AI_A_SHORTNAME = None # human
AI_B_SHORTNAME = "mctsnc_5_inf_4_256_acp_prodigal" 
REPRODUCE_EXPERIMENT = False

# folders
FOLDER_EXPERIMENTS = "../experiments/"
FOLDER_EXTRAS = "../extras/"

# automatic settings
_BOARD_SHAPE = STATE_CLASS.get_board_shape()
_EXTRA_INFO_MEMORY = STATE_CLASS.get_extra_info_memory()
_MAX_ACTIONS = STATE_CLASS.get_max_actions()
_ACTION_INDEX_TO_NAME_FUNCTION = STATE_CLASS.action_index_to_name
_HUMAN_PARTICIPANT = AI_A_SHORTNAME is None or AI_B_SHORTNAME is None
if _HUMAN_PARTICIPANT:
    REPRODUCE_EXPERIMENT = False
    if AI_A_SHORTNAME is None:
        AI_A_SHORTNAME = "human"
    if AI_B_SHORTNAME is None:
        AI_B_SHORTNAME = "human"         

# dictionary of AIs
AIS = {
    "mcts_1_inf_vanilla": MCTS(search_time_limit=1.0, search_steps_limit=np.inf, vanilla=True),
    "mcts_5_inf_vanilla": MCTS(search_time_limit=5.0, search_steps_limit=np.inf, vanilla=True),
    "mcts_30_inf_vanilla": MCTS(search_time_limit=30.0, search_steps_limit=np.inf, vanilla=True),        
    "mctsnc_1_inf_1_32_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=32, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_64_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=64, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_128_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=128, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_256_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=256, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),    
    "mctsnc_1_inf_2_32_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=32, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_64_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=64, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_128_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=128, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_256_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=256, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_32_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=32, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_64_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=64, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_128_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_256_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_32_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=32, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_64_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=64, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_128_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=128, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_256_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=256, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),    
    "mctsnc_1_inf_1_32_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=32, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_64_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=64, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_128_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=128, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_256_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=256, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),    
    "mctsnc_1_inf_2_32_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=32, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_64_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=64, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_128_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=128, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_256_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=256, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_32_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=32, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_64_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=64, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_128_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_256_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_32_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=32, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_64_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=64, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_128_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=128, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_256_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=256, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_32_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=32, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_64_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=64, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_128_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=128, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_256_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=256, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),    
    "mctsnc_1_inf_2_32_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=32, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_64_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=64, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_128_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=128, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_256_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=256, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_32_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=32, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_64_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=64, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_128_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_256_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_32_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=32, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_64_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=64, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_128_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=128, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_256_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=256, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),    
    "mctsnc_1_inf_1_32_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=32, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_64_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=64, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_128_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=128, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_1_256_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=256, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),    
    "mctsnc_1_inf_2_32_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=32, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_64_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=64, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_128_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=128, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_2_256_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=2, n_playouts=256, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_32_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=32, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_64_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=64, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_128_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_4_256_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_32_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=32, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_64_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=64, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_128_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=128, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_1_inf_8_256_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=256, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_5_inf_4_128_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="ocp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_5_inf_4_256_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="ocp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),    
    "mctsnc_5_inf_4_256_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_5_inf_4_256_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="acp_prodigal", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),                                                                    
    "mctsnc_30_inf_4_128_ocp_thrifty_16g": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=30.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="ocp_thrifty", device_memory=16.0, action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_30_inf_4_256_ocp_prodigal_16g": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=30.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="ocp_prodigal", device_memory=16.0, action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),        
    "mctsnc_30_inf_4_256_acp_thrifty_16g": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=30.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="acp_thrifty", device_memory=16.0, action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION),
    "mctsnc_30_inf_4_256_acp_prodigal_16g": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=30.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="acp_prodigal", device_memory=16.0, action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION)                            
    }

LINE_SEPARATOR = 208 * "="

if __name__ == "__main__":    
    ai_a = AIS[AI_A_SHORTNAME] if AI_A_SHORTNAME in AIS else None 
    ai_b = AIS[AI_B_SHORTNAME] if AI_B_SHORTNAME in AIS else None   
    matchup_info = {
        "ai_a_shortname": AI_A_SHORTNAME, "ai_a_instance": str(ai_a), 
        "ai_b_shortname": AI_B_SHORTNAME, "ai_b_instance": str(ai_b),
        "game_name": STATE_CLASS.class_repr(),
        "n_games": N_GAMES} 
    outcomes = np.zeros(N_GAMES, dtype=np.int8)
    c_props = cpu_and_system_props()
    g_props = gpu_props()
    experiment_hs = experiment_hash_str(matchup_info, c_props, g_props)
    experiment_info = {"matchup_info":  matchup_info, "cpu_and_system_props": c_props, "gpu_props": g_props, "games_infos": {}, "stats": {}}
    if not (REPRODUCE_EXPERIMENT or _HUMAN_PARTICIPANT):
        logger = Logger(f"{FOLDER_EXPERIMENTS}{experiment_hs}.log")    
        sys.stdout = logger
    
    print("MCTS-NC EXPERIMENT..." + f"{' [to be reproduced]' if REPRODUCE_EXPERIMENT else ''}", flush=True)
    t1 = time.time()    

    experiment_info_old = None
    if REPRODUCE_EXPERIMENT:  
        experiment_info_old = unzip_and_load_experiment(experiment_hs, FOLDER_EXPERIMENTS)
    
    print(f"HASH STRING: {experiment_hs}")    
    print(LINE_SEPARATOR)
    print(f"MATCH-UP:\n{dict_to_str(matchup_info)}")
    print(LINE_SEPARATOR)
    cpu_gpu_info = f"[CPU: {c_props['cpu_name']}, gpu: {g_props['name']}]".upper()
    print(f"CPU AND SYSTEM PROPS:\n{dict_to_str(c_props)}")
    print(f"GPU PROPS:\n{dict_to_str(g_props)}")
    print(LINE_SEPARATOR)        

    if isinstance(ai_a, MCTSNC):        
        ai_a.init_device_side_arrays()
        print(LINE_SEPARATOR)
    if isinstance(ai_b, MCTSNC):        
        ai_b.init_device_side_arrays()
        print(LINE_SEPARATOR)        
    
    score_a = 0.0
    score_b = 0.0
    black_player_ai = None
    white_player_ai = None
    
    for i in range(N_GAMES):
        print(f"\n\n\nGAME {i + 1}/{N_GAMES}:")                
        ai_a_starts = i % 2 == 0
        black_player_ai = ai_a if ai_a_starts else ai_b 
        white_player_ai = ai_b if ai_a_starts else ai_a 
        print(f"BLACK: {black_player_ai if black_player_ai else 'human'}")
        print(f"WHITE: {white_player_ai if white_player_ai else 'human'}")
        game_runner = GameRunner(STATE_CLASS, black_player_ai, white_player_ai, i + 1, N_GAMES, experiment_info_old)
        outcome, game_info = game_runner.run()
        experiment_info["games_infos"][str(i + 1)] = game_info
        outcomes[i] = outcome
        outcome_normed = 0.5 * (outcome + 1.0) # to: 0.0 - loss, 0.5 - draw, 1.0 - win
        score_a += outcome_normed if ai_a_starts else 1.0 - outcome_normed
        score_b += 1.0 - outcome_normed if ai_a_starts else outcome_normed
        print(f"[score so far for A -> total: {score_a}, mean: {score_a / (i + 1)} ({ai_a if ai_a else 'human'})]")
        print(f"[score so far for B -> total: {score_b}, mean: {score_b / (i + 1)} ({ai_b if ai_b else 'human'})]")
        print(LINE_SEPARATOR)
    
    print(f"OUTCOMES: {outcomes}")
    outcomes = np.array(outcomes, dtype=np.int8)    
    n_wins_white = np.sum(outcomes == -1)
    n_draws = np.sum(outcomes == 0)
    n_wins_black = np.sum(outcomes == 1)
    print(f"COUNTS -> WHITE WINS (-1): {n_wins_white}, DRAWS (0): {n_draws}, BLACK WINS (+1): {n_wins_black}")
    print(f"FREQUENCIES -> WHITE WINS (-1): {n_wins_white / N_GAMES}, DRAWS (0): {n_draws / N_GAMES}, BLACK WINS (+1): {n_wins_black / N_GAMES}")
    print(LINE_SEPARATOR)
    
    experiment_info["stats"]["score_a_total"] = score_a
    experiment_info["stats"]["score_a_mean"] = score_a / N_GAMES
    experiment_info["stats"]["score_b_total"] = score_b
    experiment_info["stats"]["score_b_mean"] = score_b / N_GAMES
    experiment_info["stats"]["white_wins_count"] = int(n_wins_white) # needed for serialization to json
    experiment_info["stats"]["white_wins_freq"] = n_wins_white / N_GAMES
    experiment_info["stats"]["black_wins_count"] = int(n_wins_black) # needed for serialization to json
    experiment_info["stats"]["black_wins_freq"] = n_wins_black / N_GAMES                
    
    t2 = time.time()
    print(f"MCTS-NC EXPERIMENT DONE. [time: {t2 - t1} s]")
    
    if not (REPRODUCE_EXPERIMENT or _HUMAN_PARTICIPANT):
        sys.stdout = sys.__stdout__
        logger.logfile.close()
        save_and_zip_experiment(experiment_hs, experiment_info, FOLDER_EXPERIMENTS)