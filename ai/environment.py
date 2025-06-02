import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import logging
from gym.spaces import Discrete, Box
from game.game import Game
from game.maze_generator import Map
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED, CELL_SIZE, FPS, EDIBLE_DURATION, GHOST_SCORES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

class PacManEnv:
    metadata = {"render_modes": [], "render_fps": None}

    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED):
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
        self.lives = 3
        self.state_channels = 6
        self.state_shape = (self.state_channels, self.height, self.width)
        self.frame_count = 0

        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=self.state_shape, dtype=np.float32)

        np.random.seed(seed)

    def _get_state(self):
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

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        self.pacman = self.game.get_pacman()
        self.ghosts = self.game.get_ghosts()
        self.power_pellets = self.game.get_power_pellets()
        self.score_pellets = self.game.get_score_pellets()
        self.total_pellets = len(self.score_pellets) + len(self.power_pellets)
        self.eaten_pellets = 0
        self.game_over = False
        self.current_score = 0
        self.lives = 3
        self.frame_count = 0
        for ghost in self.ghosts:
            ghost.reset(self.maze)
        self.pacman.score = 0
        self.pacman.alive = True
        self.game.running = True  # Ensure game is running
        state = self._get_state()
        logger.debug(f"Reset: Pac-Man at ({self.pacman.x}, {self.pacman.y}), Ghosts at {[f'({g.x}, {g.y})' for g in self.ghosts]}")
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        if not 0 <= action < 4:
            raise ValueError(f"Invalid action: {action}")

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
            reward += 10.0 if pellets_eaten == len(self.power_pellets) else 2.0

        collision_detected = False
        for ghost in self.ghosts:
            distance = ((self.pacman.current_x - ghost.current_x) ** 2 + 
                        (self.pacman.current_y - ghost.current_y) ** 2) ** 0.5
            if distance < self.cell_size / 2:
                if ghost.edible and ghost.edible_timer > 0:
                    reward += 50.0
                elif not ghost.edible and not ghost.returning_to_spawn:
                    self.lives -= 1
                    logger.debug(f"Collision with ghost at ({ghost.x}, {ghost.y}), Lives left: {self.lives}")
                    collision_detected = True
                    if self.lives > 0:
                        # Reset Pac-Man and ghosts but continue episode
                        self.pacman.x, self.pacman.y = self.pacman.initial_x, self.pacman.initial_y
                        self.pacman.current_x, self.pacman.current_y = self.pacman.x, self.pacman.y
                        for g in self.ghosts:
                            g.reset(self.maze)
                        reward -= 50.0  # Penalty for losing a life
                    else:
                        self.pacman.alive = False
                        self.game_over = True
                        reward -= 100.0  # Penalty for game over

        if not self.power_pellets and not self.score_pellets:
            reward += 200.0
            self.game_over = True
            logger.debug("All pellets eaten, game won")

        reward -= 0.01  # Time penalty

        next_state = np.array(self._get_state(), dtype=np.float32)
        terminated = self.game_over
        truncated = False
        info = {
            'score': self.current_score,
            'game_over': self.game_over,
            'eaten_pellets': self.total_pellets - new_pellets_count,
            'total_pellets': self.total_pellets,
            'lives': self.lives,
            'collision': collision_detected
        }
        self.frame_count += 1
        logger.debug(f"Step: Action={action}, Reward={reward:.2f}, Lives={self.lives}, Terminated={terminated}")
        return next_state, reward, terminated, truncated, info

    def close(self):
        pass