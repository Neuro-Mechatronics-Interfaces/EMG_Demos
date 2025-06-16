import sys
import random
import pygame
from pygame.math import Vector2
import os


class Snake:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.body = [Vector2(5,10), Vector2(4,10), Vector2(3,10)]
        self.direction = Vector2(1,0)
        self.new_block = False

    def Draw_snake(self, screen):
        for block in self.body:
            block_rect = pygame.Rect(int(block.x * self.cell_size), int(block.y * self.cell_size), self.cell_size, self.cell_size)
            pygame.draw.rect(screen, (123,111,182), block_rect)

    def move_snake(self):
        if self.new_block == False:
            body_copy = self.body[:-1]
        else:
            body_copy = self.body[:]
        body_copy.insert(0,body_copy[0] + self.direction)
        self.body = body_copy[:]
        self.new_block = False

    def add_body_block(self):
        self.new_block = True


class Fruit:
    def __init__(self, cell_size, cell_num):
        self.cell_size = cell_size
        self.cell_num = cell_num
        self.Randomize_fruit()

    def Draw_fruit(self, screen):
        fruit_rect = pygame.Rect(int(self.pos.x * self.cell_size), int(self.pos.y * self.cell_size), self.cell_size, self.cell_size)
        pygame.draw.ellipse(screen, (182,10,10), fruit_rect)

    def Randomize_fruit(self):
        self.x = random.randint(0, self.cell_num - 1)
        self.y = random.randint(0, self.cell_num - 1)
        self.pos = Vector2(self.x, self.y)


class Game:
    def __init__(self, cell_size=40, cell_num=20, font_path="E:\Mindrove_venv\game_assests\gabardina\Gabardina-regular.ttf", direction_queue=None):
        pygame.init()
        pygame.font.init()

        self.cell_size = cell_size
        self.cell_num = cell_num
        self.screen = pygame.display.set_mode((self.cell_size * self.cell_num, self.cell_num * self.cell_size))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.direction_queue = direction_queue

        try:
            self.game_font = pygame.font.Font(font_path, 40)
            self._font_path = font_path # Store font path for resetting
        except FileNotFoundError:
            print(f"Error: Font file not found at {font_path}! Using default Pygame font.")
            self.game_font = pygame.font.Font(None, 40)
            self._font_path = None
        except Exception as e:
            print(f"An error occurred loading the font: {e}")
            self.game_font = pygame.font.Font(None, 40)
            self._font_path = None

        self.snake = Snake(self.cell_size)
        self.fruit = Fruit(self.cell_size, self.cell_num)
        self.score_count = 0
        self.game_active = True

        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 250)

    def reset_game(self):
        self.snake = Snake(self.cell_size)
        self.fruit = Fruit(self.cell_size, self.cell_num)
        self.score_count = 0
        self.game_active = True
        pygame.time.set_timer(self.SCREEN_UPDATE, 250)

    def update(self):
        if self.game_active:
            self.snake.move_snake()
            self.Check_collision()
            self.Check_fail()

    def Draw_elements(self):
        self.Draw_grass()
        self.snake.Draw_snake(self.screen)
        self.fruit.Draw_fruit(self.screen)
        self.draw_score()

    def Check_collision(self):
        if self.fruit.pos == self.snake.body[0]:
            self.fruit.Randomize_fruit()
            self.snake.add_body_block()
            self.score_count += 1
            print(f'Score: {self.score_count}')

            for block in self.snake.body:
                if self.fruit.pos == block:
                    self.fruit.Randomize_fruit()

    def Check_fail(self):
        if not 0 <= self.snake.body[0].x < self.cell_num or not 0 <= self.snake.body[0].y < self.cell_num:
            self.Game_over()

        # for block in self.snake.body[1:]:
        #     if block == self.snake.body[0]:
        #         self.Game_over()

    def Draw_grass(self):
        grass_color = (207,255,158)
        for row in range(self.cell_num):
            for col in range(self.cell_num):
                if (row + col) % 2 == 0:
                    grass_rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, grass_color, grass_rect)

    def draw_score(self):
        score_text = f"Score: {self.score_count}"
        score_surface = self.game_font.render(score_text, True, (0,0,0))
        score_x = int(self.screen.get_width() - 100)
        score_y = int(40)
        score_rect = score_surface.get_rect(center = (score_x, score_y))
        self.screen.blit(score_surface, score_rect)

    def Game_over(self):
        print("Game Over")
        self.game_active = False

    def Game_over_disp(self):
        game_over_text = "Game Over!"
        game_over_surface = self.game_font.render(game_over_text, True, (0, 0, 0))
        game_over_rect = game_over_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        self.screen.blit(game_over_surface, game_over_rect)

    def update_direction_from_prediction(self, predicted_label):
        if not self.game_active:
            return

        if predicted_label.lower() == 'wradial' and self.snake.direction.y != 1:
            self.snake.direction = Vector2(0, -1)  # UP
        elif predicted_label.lower() == 'wulnar' and self.snake.direction.y != -1:
            self.snake.direction = Vector2(0, 1)   # DOWN
        elif predicted_label.lower() == 'wexten' and self.snake.direction.x != -1:
            self.snake.direction = Vector2(1, 0)   # RIGHT
        elif predicted_label.lower() == 'wflex' and self.snake.direction.x != 1:
            self.snake.direction = Vector2(-1, 0)  # LEFT


    def run(self):
        running = True
        while running:
            if self.direction_queue and not self.direction_queue.empty():
                try:
                    predicted_label = self.direction_queue.get_nowait()
                    print(f"[Game] Received prediction: {predicted_label}")
                    self.update_direction_from_prediction(predicted_label)
                except Exception as e:
                    print(f"Queue read error: {e}")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == self.SCREEN_UPDATE:
                    self.update()

                if event.type == pygame.KEYDOWN:
                    if self.game_active:
                        if event.key == pygame.K_UP:
                            if self.snake.direction.y != 1:
                                self.snake.direction = Vector2(0,-1)
                        if event.key == pygame.K_DOWN:
                            if self.snake.direction.y != -1:
                                self.snake.direction = Vector2(0,1)
                        if event.key == pygame.K_LEFT:
                            if self.snake.direction.x != 1:
                                self.snake.direction = Vector2(-1,0)
                        if event.key == pygame.K_RIGHT:
                            if self.snake.direction.x != -1:
                                self.snake.direction = Vector2(1,0)
                    else:
                        if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                            self.reset_game()

            self.screen.fill([175,250,70])
       

            if self.game_active:
                self.Draw_elements()
            else:
                self.Draw_grass()
                self.Game_over_disp()
                self.draw_score()

            pygame.display.update()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()



# if __name__ == '__main__':
#     game_instance = Game()
#     game_instance.run()