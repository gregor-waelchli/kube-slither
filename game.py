# game.py
import pygame
import random
from collections import namedtuple
import numpy as np

Point = namedtuple('Point', 'x y')

class SnakeGameAI:
    def __init__(self, w=640, h=480, block_size=20):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.reset()

    def reset(self):
        self.direction = pygame.K_RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head,
                      Point(self.head.x - self.block_size, self.head.y),
                      Point(self.head.x - 2*self.block_size, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.h - self.block_size) // self.block_size) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def step(self, action):  # action: 0 = straight, 1 = right, 2 = left
        self.frame_iteration += 1

        # Update direction based on action
        clock_wise = [pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT, pygame.K_UP]
        idx = clock_wise.index(self.direction)
        if action == 1:   # right turn
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2: # left turn
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = self.direction
        self.direction = new_dir

        # Move
        x = self.head.x
        y = self.head.y
        if self.direction == pygame.K_RIGHT: x += self.block_size
        elif self.direction == pygame.K_LEFT: x -= self.block_size
        elif self.direction == pygame.K_DOWN: y += self.block_size
        elif self.direction == pygame.K_UP: y -= self.block_size
        self.head = Point(x, y)

        self.snake.insert(0, self.head)


        # Timeout
        #if self.frame_iteration > 100 * len(self.snake):
        #    print("Snake took too long to find food...")
        #    game_over = True
        #    reward = -5
        #    return reward, game_over, self.score

        # Check collision or timeout
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Eat food
        if self.head == self.food:
            self.score += 1
            reward = 15
            self._place_food()
        else:
            self.snake.pop()

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        # Hits self
        if pt in self.snake[1:]:
            return True
        return False

    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - self.block_size, head.y)
        point_r = Point(head.x + self.block_size, head.y)
        point_u = Point(head.x, head.y - self.block_size)
        point_d = Point(head.x, head.y + self.block_size)

        dir_l = self.direction == pygame.K_LEFT
        dir_r = self.direction == pygame.K_RIGHT
        dir_u = self.direction == pygame.K_UP
        dir_d = self.direction == pygame.K_DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_r and self.is_collision(point_d)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)),

            # Danger left
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_d and self.is_collision(point_r)),

            # Direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y  # food down
        ]
        return np.array(state, dtype=int)

    def get_json_state(self):
        """Send minimal data to browser"""
        return {
            "snake": [[p.x, p.y] for p in self.snake],
            "food": [self.food.x, self.food.y],
            "score": self.score,
            "game_over": False  # updated in loop
        }