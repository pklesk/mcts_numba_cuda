import numpy as np
from mctsnc import MCTSNC

__version__ = "1.0.1"
__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl" 

class GameRunner:
    """
    Class responsible for carrying out a single game (game class specified via parameter) within a match between two AIs or human vs AI. 
    
    Parameters:
        game_class (class): 
            class object allowing to instantiate the initial state of some two-person game (e.g., Connect4, Gomoku, chess, etc.) and to carry out a game on it.            
        black_ai (object): 
            reference to AI instance responsible for the black player (instance of class ``MCTS`` or ``MCTSNC``) or ``None`` for human.
        white_ai (object): 
            reference to AI instance responsible for the white player (instance of class ``MCTS`` or ``MCTSNC``) or ``None`` for human.            
        game_index (int):
            index of game with a match (for informative purposes).
        n_games (int):
            total of games in a match (for informative purposes).
        experiment_info_old (dict):
            dictionary allowing to reproduce a former experiment (allows to force the limit of steps rather than time on an AI instance) or ``None`` for a new experiment, , defaults to ``None``.
            
    Attributes:
        OUTCOME_MESSAGES (list):
            list of strings with messages describing outcomes of games.
    """
    
    OUTCOME_MESSAGES = ["WHITE WINS", "DRAW", "BLACK WINS"]
    
    def __init__(self, game_class, black_ai, white_ai, game_index, n_games, experiment_info_old=None):
        """
        Constructor ``GameRunner`` instances.
         
        Args:
            game_class (class): 
                class object allowing to instantiate the initial state of some two-person game (e.g., Connect4, Gomoku, chess, etc.) and to carry out a game on it.            
            black_ai (object): 
                reference to AI instance responsible for the black player (instance of class ``MCTS`` or ``MCTSNC``) or ``None`` for human.
            white_ai (object): 
                reference to AI instance responsible for the white player (instance of class ``MCTS`` or ``MCTSNC``) or ``None`` for human.            
            game_index (int):
                index of game with a match (for informative purposes).
            n_games (int):
                total of games in a match (for informative purposes).
            experiment_info_old (dict):
                dictionary allowing to reproduce a former experiment (allows to force the limit of steps rather than time on an AI instance) or ``None`` for a new experiment, , defaults to ``None``.
        """        
        self.game_class = game_class
        self.black_ai = black_ai
        self.white_ai = white_ai
        self.game_index = game_index
        self.n_games = n_games
        self.experiment_info_old = experiment_info_old
        
    def run(self):
        """Carries out a game."""
        game = self.game_class()   
        print(game)
        outcome = 0
        game_info = {"black": str(self.black_ai), "white": str(self.white_ai), "initial_state": str(game), "moves_rounds": {}, "outcome": None, "outcome_message": None}                
        move_count = 0                       
        while True:
            print(f"\nMOVES ROUND: {move_count + 1} [game: {self.game_index}/{self.n_games}]")
            forced_search_steps_limit = np.inf
            moves_round_info = {}                     
            if not self.black_ai:
                move_valid = False
                escaped = False
                while not (move_valid or escaped):
                    try:
                        move_name = input("BLACK PLAYER, PICK YOUR MOVE: ")
                        move_index = self.game_class.action_name_to_index(move_name)
                        game_moved = game.take_action(move_index)
                        if game_moved is not None:
                            game = game_moved
                            move_valid = True                                                        
                    except:
                        print("INVALID MOVE. GAME STOPPED.")
                        escaped = True
                        break
                if escaped:
                    break
            else:
                if self.experiment_info_old is not None:
                    forced_search_steps_limit = self.experiment_info_old["games_infos"][str(self.game_index)]["moves_rounds"][str(move_count + 1)]["black_performance_info"]["steps"] 
                if isinstance(self.black_ai, MCTSNC):
                    move_index = self.black_ai.run(game.get_board(), game.get_extra_info(), game.get_turn(), forced_search_steps_limit)
                else:
                    move_index = self.black_ai.run(game, forced_search_steps_limit)
                move_name = self.game_class.action_index_to_name(move_index)
                print(f"MOVE PLAYED: {move_name}")
                game = game.take_action(move_index)
                moves_round_info["black_best_action_info"] = self.black_ai.actions_info["best"]
                moves_round_info["black_performance_info"] = self.black_ai.performance_info                
            print(str(game), flush=True)                                                
            outcome = game.compute_outcome()
            if outcome is not None:
                outcome_message = GameRunner.OUTCOME_MESSAGES[outcome + 1]           
                print(f"GAME OUTCOME: {outcome_message}")
                game_info["moves_rounds"][str(move_count + 1)] = moves_round_info
                game_info["outcome"] = outcome
                game_info["outcome_message"] = outcome_message
                break                
            if not self.white_ai:
                move_valid = False
                escaped = False
                while not (move_valid or escaped):
                    try:
                        move_name = input("WHITE PLAYER, PICK YOUR MOVE: ")
                        move_index = self.game_class.action_name_to_index(move_name)
                        game_moved = game.take_action(move_index)
                        if game_moved is not None:
                            game = game_moved
                            move_valid = True                            
                    except:
                        print("INVALID MOVE. GAME STOPPED.")
                        escaped = True
                        break
                if escaped:
                    break                
            else:
                if self.experiment_info_old is not None:
                    forced_search_steps_limit = self.experiment_info_old["games_infos"][str(self.game_index)]["moves_rounds"][str(move_count + 1)]["white_performance_info"]["steps"]                
                if isinstance(self.white_ai, MCTSNC):
                    move_index = self.white_ai.run(game.get_board(), game.get_extra_info(), game.get_turn(), forced_search_steps_limit)
                else:
                    move_index = self.white_ai.run(game, forced_search_steps_limit)
                move_name = self.game_class.action_index_to_name(move_index)
                print(f"MOVE PLAYED: {move_name}")
                game = game.take_action(move_index)
                moves_round_info["white_best_action_info"] = self.white_ai.actions_info["best"]            
                moves_round_info["white_performance_info"] = self.white_ai.performance_info                
            print(str(game), flush=True)                                        
            game_info["moves_rounds"][str(move_count + 1)] = moves_round_info  
            outcome = game.compute_outcome()
            if outcome is not None:
                outcome_message = GameRunner.OUTCOME_MESSAGES[outcome + 1]
                print(f"GAME OUTCOME: {outcome_message}")
                game_info["moves_rounds"][str(move_count + 1)] = moves_round_info
                game_info["outcome"] = outcome
                game_info["outcome_message"] = outcome_message                                
                break
            move_count += 1
        return outcome, game_info