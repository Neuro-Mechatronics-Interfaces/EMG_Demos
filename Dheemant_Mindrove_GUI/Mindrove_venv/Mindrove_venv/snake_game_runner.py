from Snake_Game import Game
from The_Space_Shooter_Game_master.Game import *

def run_pygame_game_process(cell_size, cell_num, font_path, direction_queue):
    game = Game(cell_size=cell_size, cell_num=cell_num, font_path=font_path, direction_queue=direction_queue)
    game.run()

def run_SSGgame_game_process(direction_queue):
    game = Game(direction_queue = direction_queue)
    game.run()