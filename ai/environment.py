# game/environment.py
"""
Pac-Man game environment for reinforcement learning, following OpenAI Gym conventions.
Initializes the maze, manages state transitions, computes rewards, and provides observations.
Supports training with Dueling DQN agents.
"""

import os
import sys
import numpy as np
import pygame
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adjust sys.path for imports
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from game.entities.pacman import PacMan
from game.entities.entity_initializer import initialize_entities
from game.entities.pellets import PowerPellet, ScorePellet
from game.maze_generator import Map
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, CELL_SIZE, FPS, EDIBLE_DURATION, GHOST_SCORES
from game.renderer import Renderer
from game.game import Game 

class PacManEnv:
    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED, visualize=False):
        """
        Initialize the Pac-Man game environment.

        Args:
            width (int): Maze width, default from config.
            height (int): Maze height, default from config.
            seed (int): Random seed, default from config.
        """
        self.visualize = visualize
        self.game = Game(player_name="RL_Agent")
        self.maze = self.game.get_maze()
        self.pacman = self.game.get_pacman()
        self.ghosts = self.game.get_ghosts()
        self.power_pellets = self.game.get_power_pellets()
        self.score_pellets = self.game.get_score_pellets()
        
        self.width = width
        self.height = height
        self.cell_size = CELL_SIZE
        self.seed = seed
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        self.eaten_pellets = 0
        self.game_over = False
        self.current_score = 0
        self.lives = 1
        if self.visualize:
            pygame.init()
            self.screen_width = self.width * self.cell_size
            self.screen_height = self.height * self.cell_size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Pac-Man RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
            self.renderer = Renderer(self.screen, self.font, self.screen_width, self.screen_height)
            logger.info("Pygame initialized for visualization")
        else:
            logger.info("Visualization disabled, skipping Pygame initialization")

        self.state_channels = 6  # Pac-Man, PowerPellet, ScorePellet, EdibleGhost, NormalGhost, Wall
        self.state_shape = (self.state_channels, self.height, self.width)
        self.frame_count = 0

    def _get_state(self):
        """
        Generate the current game state.

        Returns:
            np.ndarray: State array of shape (C, H, W).
        """
        state = np.zeros((self.state_channels, self.height, self.width), dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                if self.maze.get_tile(x, y) in ['#', 'X']:
                    state[5, y, x] = 1.0

        state[0, self.pacman.y, self.pacman.x] = 1.0

        for pellet in self.power_pellets:
            state[1, pellet.y, pellet.x] = 1.0

        for pellet in self.score_pellets:
            state[2, pellet.y, pellet.x] = 1.0

        for ghost in self.ghosts:
            if ghost.edible and ghost.edible_timer > 0 and not ghost.returning_to_spawn:
                state[3, ghost.y, ghost.x] = 1.0
            else:
                state[4, ghost.y, ghost.x] = 1.0

        return state

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            np.ndarray: Initial state observation.
        """
        try:
            self.game = Game(player_name="Player")
            self.maze = self.game.get_maze()
            self.pacman = self.game.get_pacman()
            self.ghosts = self.game.get_ghosts()
            self.power_pellets = self.game.get_power_pellets()
            self.score_pellets = self.game.get_score_pellets()
            self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
            self.eaten_pellets = 0
            self.game_over = False
            self.current_score = 0
            self.frame_count = 0
            self.lives = 1
            for ghost in self.ghosts:
                ghost.reset(self.maze)
            logger.info("Environment reset successfully")
            return np.array(self._get_state(), dtype=np.float32)
        except Exception as e:
            logger.error(f"Reset failed: {str(e)}")
            raise RuntimeError(f"Failed to reset environment: {str(e)}")

    def step(self, action):
        """
        Execute an action and advance the game state.

        Args:
            action (int): Action (0: up, 1: down, 2: left, 3: right).

        Returns:
            Tuple[np.ndarray, float, bool, dict]: (next_state, reward, done, info)
        """
        if not 0 <= action < 4:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, 2, or 3.")

        old_score = self.current_score
        old_pellets_count = len(self.power_pellets) + len(self.score_pellets)

        def move_pacman():
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = directions[action]
            self.pacman.set_new_target(dx, dy, self.maze)

        try:
            self.game.update(FPS, move_pacman)
        except Exception as e:
            logger.error(f"Game update failed: {str(e)}")
            raise RuntimeError(f"Game update failed: {str(e)}")

        self.pacman = self.game.get_pacman()
        self.ghosts = self.game.get_ghosts()
        self.power_pellets = self.game.get_power_pellets()
        self.score_pellets = self.game.get_score_pellets()
        self.current_score = self.pacman.score
        new_pellets_count = len(self.power_pellets) + len(self.score_pellets)

        reward = 0.0
        pellets_eaten = old_pellets_count - new_pellets_count
        if pellets_eaten > 0:
            score_diff = self.current_score - old_score
            reward += score_diff

        for ghost in self.ghosts:
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if distance < CELL_SIZE / 2:
                if ghost.edible and ghost.edible_timer > 0:
                    reward += GHOST_SCORES[self.game.ghost_score_index]
                elif not ghost.edible and not ghost.returning_to_spawn:
                    self.pacman.alive = False
                    self.lives = 0

        reward += -0.01  # Time penalty

        if not self.pacman.alive:
            reward += -100.0
            self.game_over = True
            logger.info("Game over: Pac-Man lost all lives")
        elif not self.power_pellets and not self.score_pellets:
            reward += 200.0
            self.game_over = True
            logger.info("Game over: All pellets eaten")

        next_state = np.array(self._get_state(), dtype=np.float32)
        info = {
            'score': self.current_score,
            'game_over': self.game_over,
            'eaten_pellets': self.total_pellets - new_pellets_count,
            'total_pellets': self.total_pellets,
            'lives': self.lives
        }

        self.frame_count += 1
        return next_state, reward, self.game_over, info

    def render(self):
        """
        Render the game screen.
        """
        self.renderer.render(self.game, control_mode="DQN_AI", frame_count=self.frame_count)
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        """
        Close the Pygame window.
        """
        pygame.quit()