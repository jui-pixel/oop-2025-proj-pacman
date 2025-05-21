# game/environment.py
import numpy as np
from game.entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities
from game.maze_generator import Map
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED

class PacManEnv:
    def __init__(self, width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED):
        self.maze = Map(w=width, h=height, seed=seed)
        self.maze.generate_connected_maze(path_density=0.7)
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        self.done = False
        self.action_space = [0, 1, 2, 3]  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = (height, width, 5)  # (height, width, channels)

    def reset(self):
        self.maze.generate_connected_maze(path_density=0.7)
        self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
        self.done = False
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.maze.h, self.maze.w, 5), dtype=np.float32)
        state[self.pacman.x, self.pacman.y, 0] = 1.0
        for pellet in self.power_pellets:
            state[pellet.x, pellet.y, 1] = 1.0
        for pellet in self.score_pellets:
            state[pellet.x, pellet.y, 2] = 1.0
        for ghost in self.ghosts:
            if ghost.edible and ghost.respawn_timer > 0:
                state[ghost.x, ghost.y, 3] = 1.0
            else:
                state[ghost.x, ghost.y, 4] = 1.0
        return state

    def step(self, action):
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        moving = self.pacman.set_new_target(dx, dy, self.maze)
        if moving:
            self.pacman.move_towards_target(self.maze)

        reward = -1
        if self.pacman.eat_pellet(self.power_pellets) > 0:
            reward = 10
        if self.pacman.eat_score_pellet(self.score_pellets) > 0:
            reward = 2

        for ghost in self.ghosts:
            if self.pacman.x == ghost.x and self.pacman.y == ghost.y:
                if ghost.edible and ghost.respawn_timer > 0:
                    reward = [50, 100, 150, 200][min(ghost.death_count, 3)]
                    ghost.set_returning_to_spawn(30)
                elif not ghost.returning_to_spawn and not ghost.waiting:
                    reward = -100
                    self.done = True

        for ghost in self.ghosts:
            if ghost.move_towards_target(self.maze):
                if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == 'S':
                    ghost.set_waiting(30)
                else:
                    ghost.move(self.pacman, self.maze, 30)

        return self._get_state(), reward, self.done, {}

    def render(self):
        pass  # Handled by main.py