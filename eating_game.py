import pygame
import random
from math import sqrt
import numpy as np

# constants
SCREEN_SIZE = (800, 800)
DOT_SIZE = 40
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Dot:  # used to represent characters
    def __init__(self, color, pos):
        self.color = color
        self.pos = pos

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, pygame.Rect(
            self.pos[0]*DOT_SIZE, self.pos[1]*DOT_SIZE, DOT_SIZE, DOT_SIZE))


class Game:  # contains objects used for game
    def __init__(self):
        pygame.init()
        # game_state and game_scores are the state of the board and the score at each state
        # TODO modify this to just score and not euclidean distance at each time point
        self.game_states = []
        self.game_scores = []
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()
        self.player = Dot(WHITE, [10, 10])
        self.player.pos[0] = 10
        self.player.pos[1] = 10
        self.red_dots = []
        self.green_dots = []
        self.score = 0
        self.font = pygame.font.SysFont(None, 50)
        self.spawn_green_dot()
        self.spawn_red_dot()

    def spawn_green_dot(self):  # spawn food around the map
        pos = self.get_random_empty_position()
        self.green_dots.append(Dot(GREEN, pos))

    def spawn_red_dot(self):  # spawn enemies around the map
        pos = self.get_random_empty_position()
        self.red_dots.append(Dot(RED, pos))

    def get_random_empty_position(self):
        while True:
            pos = [random.randint(0, 19), random.randint(0, 19)]
            if pos not in [dot.pos for dot in self.green_dots + self.red_dots] and pos != self.player.pos:
                return pos

    def remove(self, dot):
        if dot in self.green_dots:
            self.green_dots.remove(dot)
        elif dot in self.red_dots:
            self.red_dots.remove(dot)

    def get_game_state(self):  # return 2D representation of game state for NN
        # Initialize an empty 2D array
        # 0 for empty, 1 for snake, 2 for food, 3 for enemies
        game_state = [[0 for _ in range(20)] for _ in range(20)]

        game_state[self.player.pos[1]][self.player.pos[0]] = 1

        for green_dot in self.green_dots:
            game_state[green_dot.pos[1]][green_dot.pos[0]] = 2

        for red_dot in self.red_dots:
            game_state[red_dot.pos[1]][red_dot.pos[0]] = 3

        return game_state

    def get_score(self):  # score is distance to green / distance to red
        green_score = 0
        for green_dot in self.green_dots:
            # calculate distance from green dot to player
            green_score += sqrt(
                (self.player.pos[0] - green_dot.pos[0])**2 + (self.player.pos[1] - green_dot.pos[1])**2)
        red_score = 0
        for red_dot in self.red_dots:
            # calculate distance from red dot to player
            red_score += sqrt(
                (self.player.pos[0] - red_dot.pos[0])**2 + (self.player.pos[1] - red_dot.pos[1])**2)
        return green_score / red_score

    def run(self):
        running = True
        while running:
            self.screen.fill((0, 0, 0))  # clear screen
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # move snake around with arrow keys
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.player.pos[1] -= 1
                    elif event.key == pygame.K_DOWN:
                        self.player.pos[1] += 1
                    elif event.key == pygame.K_LEFT:
                        self.player.pos[0] -= 1
                    elif event.key == pygame.K_RIGHT:
                        self.player.pos[0] += 1

            # restrict player movement to 20x20 grid
            self.player.pos = [max(0, min(19, self.player.pos[0])), max(
                0, min(19, self.player.pos[1]))]

            self.player.draw(self.screen)  # draw player

            # check for red dot collison, update score + spawn new dot
            for red_dot in self.red_dots:
                red_dot.draw(self.screen)
                if self.player.pos == red_dot.pos:
                    self.remove(red_dot)
                    self.score -= 1
                    self.spawn_red_dot()

            # check for green dot collison, update score + spawn new dot
            for green_dot in self.green_dots:
                green_dot.draw(self.screen)
                if self.player.pos == green_dot.pos:
                    self.remove(green_dot)
                    self.score += 1
                    self.spawn_green_dot()

            # display score
            score_text = self.font.render(
                f'Score: {self.score}', True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))

            # get game state and write to array only if game_state different from previous state - don't insert duplicate data
            current_game_state = self.get_game_state()
            if (len(self.game_states) == 0 or np.array_equal(current_game_state, self.game_states[-1]) == False):
                print("len(self.game_states): ", len(self.game_states))
                # print("b: ", current_game_state)
                self.game_states.append(current_game_state)
                self.game_scores.append(self.get_score())

            pygame.display.flip()  # update display


if __name__ == "__main__":
    game = Game()
    game.run()

# input is board state, output is left, right, up, down commands.
# heuristic/loss can be distance from green dots and red dots = score. Maximize score.
# 1 calulcate score at one point given board ✅
# 2 build method to store board state as well as score ✅
# 3 step through NN and update hidden neurons based on score - teaches to avoid red and go to green
