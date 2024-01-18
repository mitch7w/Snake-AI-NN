# Save game state and score data to .npy files for use in pytorch training

import pygame
import random
from math import sqrt
import numpy as np

# game setup

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

    def get_game_state(self):
        # Initialize an empty 2D array
        # One-hot encoding: 0 for empty, 1 for snake, 2 for food, 3 for enemies
        game_state = [[0 for _ in range(20)] for _ in range(20)]

        # Set player position
        game_state[self.player.pos[0]][self.player.pos[1]] = 1

        # Set green dot positions
        for green_dot in self.green_dots:
            game_state[green_dot.pos[0]][green_dot.pos[1]] = 2

        # Set red dot positions
        for red_dot in self.red_dots:
            game_state[red_dot.pos[0]][red_dot.pos[1]] = 3

        return game_state

    # score is distance to green / distance to red for up,right,down,left positions of player position

    def get_score_for_movement(self):
        current_x = self.player.pos[0]
        current_y = self.player.pos[1]

        # calculate the score for each position next to the player
        def calculate_score_for_pos(pos_x, pos_y):
            green_score = 0
            for green_dot in self.green_dots:
                # calculate distance from green dot to player
                green_score += sqrt(
                    (pos_x - green_dot.pos[0])**2 + (pos_y - green_dot.pos[1])**2)
            red_score = 0
            for red_dot in self.red_dots:
                # calculate distance from red dot to player
                red_score += sqrt(
                    (pos_x - red_dot.pos[0])**2 + (pos_y - red_dot.pos[1])**2)
            # now normalize to beween 0-1. Min green_score could be 1, max red score could be 26
            if (red_score == 0):
                return 0
            if (green_score == 0):
                return 1
            unnormalized_score = red_score/(2*green_score+1e-8)
            return (unnormalized_score/26)
        # (0,0) is top left
        if (current_y+1 == 20):  # reached bottom
            score_down = 0
        else:
            score_down = calculate_score_for_pos(current_x, current_y+1)
        if (current_y-1 == -1):  # reached top
            score_up = 0
        else:
            score_up = calculate_score_for_pos(current_x, current_y-1)
        if (current_x+1 == 20):
            score_right = 0
        else:
            score_right = calculate_score_for_pos(current_x+1, current_y)
        if (current_x-1 == -1):
            score_left = 0
        else:
            score_left = calculate_score_for_pos(current_x-1, current_y)
        # print out directions for user movement
        chosen = np.argmax([score_up, score_right, score_down, score_left])
        if (self.score > 1000):
            newevent = pygame.event.Event(pygame.K_q, unicode="q",
                                          key=pygame.K_q, mod=pygame.KMOD_NONE)  # create the event
            pygame.event.post(newevent)  # add the event to the queue
        elif (chosen == 0):
            print("Up")
            newevent = pygame.event.Event(pygame.KEYDOWN, unicode="↑",
                                          key=pygame.K_UP, mod=pygame.KMOD_NONE)  # create the event
            pygame.event.post(newevent)  # add the event to the queue
        elif (chosen == 1):
            print("Right")
            newevent = pygame.event.Event(pygame.KEYDOWN, unicode="›",
                                          key=pygame.K_RIGHT, mod=pygame.KMOD_NONE)  # create the event
            pygame.event.post(newevent)  # add the event to the queue
        elif (chosen == 2):
            print("Down")
            newevent = pygame.event.Event(pygame.KEYDOWN, unicode="↓",
                                          key=pygame.K_DOWN, mod=pygame.KMOD_NONE)  # create the event
            pygame.event.post(newevent)  # add the event to the queue
        elif (chosen == 3):
            print("Left")
            newevent = pygame.event.Event(pygame.KEYDOWN, unicode="‹",
                                          key=pygame.K_LEFT, mod=pygame.KMOD_NONE)  # create the event
            pygame.event.post(newevent)  # add the event to the queue
        print(score_up, score_right, score_down, score_left)
        return [score_up, score_right, score_down, score_left]

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
                    elif event.key == pygame.K_q:
                        np.save("saved_game_states.npy", self.game_states)
                        np.save("saved_game_scores.npy", self.game_scores)
                        running = False

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
                self.game_states.append(current_game_state)
                #  score for positions up, down, left right of current position
                self.game_scores.append(self.get_score_for_movement())

            pygame.display.flip()  # update display


if __name__ == "__main__":
    game = Game()
    game.run()

# input is board state, output is left, right, up, down commands.
# heuristic/loss can be distance from green dots and red dots = score. Maximize score.
# 1 calulcate score at one point given board ✅
# 2 build method to store board state as well as score ✅
# 3 step through NN and update hidden neurons based on score - teaches to avoid red and go to green

# check values of output and input - then check values in each layer of NN in pytorch program TODO


# Video topics
    # heuristic function important - garbage in garbage out
    # one hot encoding and normalizing
    # basic nn - basic behaviour - try policy gradient
