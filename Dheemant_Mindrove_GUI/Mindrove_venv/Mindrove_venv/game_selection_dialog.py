# game_selection_dialog.py
import sys
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QApplication
from PyQt5.QtCore import Qt
import multiprocessing
from multiprocessing import Process, Queue
import os

from snake_game_runner import run_pygame_game_process, run_SSGgame_game_process





class GameSelectionDialog(QDialog):
    def __init__(self, parent=None, direction_queue=None):
        super().__init__(parent)
        self.setWindowTitle('Select a Game')
        self.setGeometry(200, 200, 300, 150)
        self.direction_queue = direction_queue  # << Store externally passed queue
        self.pygame_process = None

        self.initUI()

    def initUI(self):
        # Use a QVBoxLayout as the main layout for the dialog
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter) # Center content vertically

        self.label = QLabel("Choose a game to play:", self)
        self.label.setAlignment(Qt.AlignCenter) # Center text within the label
        main_layout.addWidget(self.label)

        # Create an QHBoxLayout to hold the game buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setAlignment(Qt.AlignCenter) # Center buttons horizontally within their layout

        # Button for Snake Game
        self.snake_button = QPushButton('Snake Game', self)
        self.snake_button.clicked.connect(self.launch_snake_game)
        buttons_layout.addWidget(self.snake_button)

        # Button for Space SHooter Game
        self.SSG_button = QPushButton('Space Shooter Game', self)
        self.SSG_button.clicked.connect(self.launch_SSG_game)
        buttons_layout.addWidget(self.SSG_button)


        # Add the buttons layout to the main vertical layout
        main_layout.addLayout(buttons_layout)

        # Set the main layout for the dialog
        self.setLayout(main_layout)

    def launch_snake_game(self):
        if self.pygame_process:
            if self.pygame_process.is_alive():
                QMessageBox.information(self, "Game Running", "The Snake game is already active.")
                return
            else:
                self.pygame_process.join()
                self.pygame_process = None

        print("Launching Snake Game...")
        try:
            self.direction_queue = Queue()


            if self.parent() is not None:
                self.parent().snake_game_dialog = self  # Attach this dialog (with the queue) to the main GUI

            game_args = {
                'cell_size': 40,
                'cell_num': 20,
                'font_path': "E:\\Mindrove_venv\\game_assests\\gabardina\\Gabardina-regular.ttf",
                'direction_queue': self.direction_queue
            }
            from snake_game_runner import run_pygame_game_process
            self.pygame_process = Process(target=run_pygame_game_process, kwargs=game_args)
            self.pygame_process.start()
            
            print("Snake game process started.")

        except Exception as e:
            print(f"Failed to launch Snake Game: {e}")

    def launch_SSG_game(self):
        if self.pygame_process:
            if self.pygame_process.is_alive():
                QMessageBox.information(self, "Game Running", "The Space Shooter game is already active.")
                return
            else:
                self.pygame_process.join()
                self.pygame_process = None

        print("Launching Space Shooter Game...")
        try:
            self.direction_queue = Queue()


            if self.parent() is not None:
                self.parent().SSG_game_dialog = self  # Attach this dialog (with the queue) to the main GUI

            game_args = {

                'direction_queue': self.direction_queue
            }

            self.pygame_process = Process(target=run_SSGgame_game_process, kwargs=game_args)
            self.pygame_process.start()
            
            print("Space Shooter process started.")

        except Exception as e:
            print(f"Failed to launch Space Shooter: {e}")

    def reject(self):
        self.cleanup_process()
        super().reject()

    def closeEvent(self, event):
        self.cleanup_process()
        event.accept()

    def cleanup_process(self):
        if self.pygame_process and self.pygame_process.is_alive():
            print("Terminating game process from GameSelectionDialog...")
            self.pygame_process.terminate()
            self.pygame_process.join()
            print("Game process terminated.")

    

 


# if __name__ == '__main__':
#     # multiprocessing.freeze_support()

#     app = QApplication(sys.argv)
#     dialog = GameSelectionDialog()
#     dialog.exec_()
#     sys.exit(0)