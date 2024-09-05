from mctsnc import MCTSNC

class GameRunner:
    
    OUTCOME_MESSAGES = ["WHITE WINS", "DRAW", "BLACK WINS"]
    
    def __init__(self, game_class, black_ai, white_ai, game_index_or_name):
        self.game_class = game_class
        self.black_ai = black_ai
        self.white_ai = white_ai
        self.game_index_or_name = game_index_or_name
        
    def run(self):
        game = self.game_class()   
        print(game)
        outcome = 0
        move_count = 0                       
        while True:
            print(f"\nMOVES ROUND: {move_count + 1} [game: {self.game_index_or_name}]")                     
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
                if isinstance(self.black_ai, MCTSNC):
                    move_index = self.black_ai.run(game.get_board(), game.get_extra_info(), game.turn)
                else:
                    move_index = self.black_ai.run(game)
                move_name = self.game_class.action_index_to_name(move_index)
                print(f"MOVE PLAYED: {move_name}")
                game = game.take_action(move_index)
            print(str(game), flush=True)
            outcome = game.compute_outcome()
            if outcome is not None:                
                print(f"GAME OUTCOME: {GameRunner.OUTCOME_MESSAGES[outcome + 1]}")
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
                if isinstance(self.white_ai, MCTSNC):
                    move_index = self.white_ai.run(game.get_board(), game.get_extra_info(), game.turn)
                else:
                    move_index = self.white_ai.run(game)
                move_name = self.game_class.action_index_to_name(move_index)
                print(f"MOVE PLAYED: {move_name}")
                game = game.take_action(move_index)
            print(str(game), flush=True)
            outcome = game.compute_outcome()
            if outcome is not None:
                print(f"GAME OUTCOME: {GameRunner.OUTCOME_MESSAGES[outcome + 1]}")
                break
            move_count += 1
        return outcome