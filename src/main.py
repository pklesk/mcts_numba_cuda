import numpy as np
from mcts import MCTS
from mctsnc import MCTSNC
from c4 import C4
from gomoku import Gomoku
from game_runner import GameRunner
import time

STATE_CLASS = Gomoku
_BOARD_SHAPE = STATE_CLASS.get_board_shape()
_EXTRA_INFO_MEMORY = STATE_CLASS.get_extra_info_memory()
_MAX_ACTIONS = STATE_CLASS.get_max_actions()
_ACTION_TO_NAME_FUNCTION = STATE_CLASS.move_index_to_name

AIS = {
    "mcts_10_inf": MCTS(search_time_limit=10.0, search_steps_limit=np.inf),
    "mcts_15_inf": MCTS(search_time_limit=15.0, search_steps_limit=np.inf),
    "mctsnc_5_inf_1_512_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=1, n_playouts=512, variant="ocp_thrifty", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_5_inf_4_128_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="ocp_thrifty", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_inf_1000_4_128_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=np.inf, search_steps_limit=1000, n_trees=4, n_playouts=128, variant="ocp_thrifty", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_inf_1000_4_128_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=np.inf, search_steps_limit=1000, n_trees=4, n_playouts=128, variant="acp_thrifty", action_to_name_function=_ACTION_TO_NAME_FUNCTION),                
    "mctsnc_5_inf_4_128_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="ocp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_5_inf_1_512_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=1, n_playouts=512, variant="acp_thrifty", action_to_name_function=_ACTION_TO_NAME_FUNCTION),    
    "mctsnc_5_inf_4_128_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="acp_thrifty", action_to_name_function=_ACTION_TO_NAME_FUNCTION),            
    "mctsnc_5_inf_1_32_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=1, n_playouts=32, variant="acp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_5_inf_1_64_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=1, n_playouts=64, variant="acp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),        
    "mctsnc_5_inf_1_256_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=1, n_playouts=256, variant="acp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),    
    "mctsnc_5_inf_4_32_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=32, variant="acp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),    
    "mctsnc_5_inf_4_128_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="acp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_2_inf_4_128_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=2.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="acp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_3_inf_4_128_ocp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=3.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="ocp_thrifty", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_3_inf_4_128_ocp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=3.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="ocp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_3_inf_4_128_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=3.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="acp_thrifty", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_3_inf_4_128_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=3.0, search_steps_limit=np.inf, n_trees=4, n_playouts=128, variant="acp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_10_inf_1_32_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=10.0, search_steps_limit=np.inf, n_trees=1, n_playouts=32, variant="acp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION),
    "mctsnc_20_inf_8_256_acp_prodigal": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=20.0, search_steps_limit=np.inf, n_trees=8, n_playouts=256, variant="acp_prodigal", action_to_name_function=_ACTION_TO_NAME_FUNCTION)                    
    } 

LINE_SEPARATOR = 208 * "="

if __name__ == "__main__":
    print("MAIN (MCTS EXPERIMENTS)...", flush=True)
    t1 = time.time()
    n_games = 1
    outcomes = np.zeros(n_games, dtype=np.int8)
    ai_A = AIS["mctsnc_5_inf_1_32_acp_prodigal"]
    ai_B = None
    
    print(LINE_SEPARATOR)
    print("MATCH-UP:")
    print(f"A: {ai_A if ai_A else 'human'}")
    print("VS")
    print(f"B: {ai_B if ai_B else 'human'}")    
    if isinstance(ai_A, MCTSNC):
        print(LINE_SEPARATOR)
        ai_A.init_device_side_arrays()
    if isinstance(ai_B, MCTSNC):
        print(LINE_SEPARATOR)
        ai_B.init_device_side_arrays()        
    print(LINE_SEPARATOR)
    
    score_A = 0.0
    score_B = 0.0
    c4_runner = None
    black_player_ai = None
    white_player_ai = None
    for i in range(n_games):
        print(f"GAME {i + 1}/{n_games}:")                
        ai_A_starts = i % 2 == 0
        black_player_ai = ai_A if ai_A_starts else ai_B 
        white_player_ai = ai_B if ai_A_starts else ai_A
        print(f"BLACK: {black_player_ai}")
        print(f"WHITE: {white_player_ai}")
        game_runner = GameRunner(STATE_CLASS, black_player_ai, white_player_ai)
        outcome = game_runner.run()         
        outcomes[i] = outcome
        outcome_normed = 0.5 * (outcome + 1.0) # to: 0.0 - loss, 0.5 - draw, 1.0 - win
        score_A += outcome_normed if ai_A_starts else 1.0 - outcome_normed
        score_B += 1.0 - outcome_normed if ai_A_starts else outcome_normed
        print(f"[score so far for A -> total: {score_A}, mean: {score_A / (i + 1)} ({ai_A})]")
        print(f"[score so far for B -> total: {score_B}, mean: {score_B / (i + 1)} ({ai_B})]")
        print(LINE_SEPARATOR)    
        
    print(f"OUTCOMES: {outcomes}")
    outcomes = np.array(outcomes, dtype=np.int8)    
    n_wins_white = np.sum(outcomes == -1)
    n_draws = np.sum(outcomes == 0)
    n_wins_black = np.sum(outcomes == 1)
    print(f"COUNTS -> WHITE WINS (-1): {n_wins_white}, DRAWS (0): {n_draws}, BLACK WINS (+1): {n_wins_black}")
    print(f"FREQUENCIES -> WHITE WINS (-1): {n_wins_white / n_games}, DRAWS (0): {n_draws / n_games}, BLACK WINS (+1): {n_wins_black / n_games}")
    t2 = time.time()
    print(f"MAIN (MCTS EXPERIMENTS) DONE. [time: {t2 - t1} s]")