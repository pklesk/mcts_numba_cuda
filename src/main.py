import numpy as np
from mcts import MCTS
from mctsnc import MCTSNC
from c4 import C4
from gomoku import Gomoku
from game_runner import GameRunner
import time
from utils import cpu_and_system_props, gpu_props, hash_str, dict_to_str, Logger
import json
import sys
import zipfile as zf
import os

__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

# folders
FOLDER_EXPERIMENTS = "../experiments/"
FOLDER_EXTRAS = "../extras/"

# main settings
STATE_CLASS = C4 # C4 or Gomoku
N_GAMES = 100
AI_A_SHORTNAME = "mcts_4_inf_vanilla"
AI_B_SHORTNAME = "mctsnc_1_inf_1_64_ocp_thrifty"
REPRODUCE_EXPERIMENT = False
_BOARD_SHAPE = STATE_CLASS.get_board_shape()
_EXTRA_INFO_MEMORY = STATE_CLASS.get_extra_info_memory()
_MAX_ACTIONS = STATE_CLASS.get_max_actions()
_ACTION_INDEX_TO_NAME_FUNCTION = STATE_CLASS.action_index_to_name

# dictionary of AIs
AIS = {
    "mcts_4_inf_vanilla": MCTS(search_time_limit=4.0, search_steps_limit=np.inf, vanilla=True),
    "mcts_8_inf_vanilla": MCTS(search_time_limit=4.0, search_steps_limit=np.inf, vanilla=True),
    "mcts_16_inf_vanilla": MCTS(search_time_limit=4.0, search_steps_limit=np.inf, vanilla=True),        
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
    "mctsnc_1_inf_8_256_acp_thrifty": MCTSNC(_BOARD_SHAPE, _EXTRA_INFO_MEMORY, _MAX_ACTIONS, search_time_limit=1.0, search_steps_limit=np.inf, n_trees=8, n_playouts=256, variant="acp_thrifty", action_index_to_name_function=_ACTION_INDEX_TO_NAME_FUNCTION)    
    }

LINE_SEPARATOR = 208 * "="

def experiment_hash_str(matchup_info, c_props, g_props, main_hs_digits=10, matchup_hs_digits=5, env_hs_digits=3):
    """Returns a hash string for an experiment, based on its settings and properties."""
    matchup_hs = hash_str(matchup_info, digits=matchup_hs_digits)
    env_props = {**c_props, **g_props}    
    env_hs =  hash_str(env_props, digits=env_hs_digits)
    all_info = {**matchup_info, **env_props}
    all_hs = hash_str(all_info, digits=main_hs_digits)
    hs = f"{all_hs}_{matchup_hs}_{env_hs}_[{matchup_info['ai_a_shortname']};{matchup_info['ai_b_shortname']};{matchup_info['game_name']};{matchup_info['n_games']}]"
    return hs

def save_and_zip_experiment(experiment_hs, experiment_info, folder):
    print(f"SAVE AND ZIP EXPERIMENT... [hash string: {experiment_hs}]")
    t1 = time.time()
    fpath = folder + experiment_hs    
    try:        
        f = open(fpath + ".json", "w+")
        json.dump(experiment_info, f, indent=2)
        f.close()
        with zf.ZipFile(fpath + ".zip", mode="w", compression=zf.ZIP_DEFLATED) as archive:
                archive.write(fpath + ".json", arcname=experiment_hs + ".json")
                archive.write(fpath + ".log", arcname=experiment_hs + ".log")
        os.remove(fpath + ".json")
        os.remove(fpath + ".log") 
    except IOError:
        sys.exit(f"[error occurred when trying to save and zip experiment info: {fname}]")            
    t2 = time.time()
    print(f"SAVE AND ZIP EXPERIMENT DONE. [time: {t2 - t1} s]")

def unzip_and_load_experiment(experiment_hs, folder):
    print(f"UNZIP AND LOAD EXPERIMENT... [hash string: {experiment_hs}]")
    t1 = time.time()
    fpath = folder + experiment_hs    
    try:        
        with zf.ZipFile(fpath + ".zip", "r") as zip_ref:
            zip_ref.extract(experiment_hs + ".json", path=os.path.dirname(fpath + ".json"))            
        with open(fpath + ".json", 'r', encoding="utf-8") as json_file:
            experiment_info = json.load(json_file) 
        os.remove(fpath + ".json") # TODO uncomment this back, to have extracted file removed once used
    except IOError:
        sys.exit(f"[error occurred when trying to unzip and load experiment info: {experiment_hs}]")            
    t2 = time.time()
    print(f"UNZIP AND LOAD EXPERIMENT DONE. [time: {t2 - t1} s]")
    return experiment_info


if __name__ == "__main__":    
    ai_a = AIS[AI_A_SHORTNAME]
    ai_b = AIS[AI_B_SHORTNAME]    
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
    if not REPRODUCE_EXPERIMENT:
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
    c4_runner = None
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
    
    if not REPRODUCE_EXPERIMENT:
        sys.stdout = sys.__stdout__
        logger.logfile.close()
        save_and_zip_experiment(experiment_hs, experiment_info, FOLDER_EXPERIMENTS)