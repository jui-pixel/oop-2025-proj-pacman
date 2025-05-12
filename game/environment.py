# # game/environment.py
# import numpy as np
# from game.entities import PacMan, Ghost, PowerPellet, ScorePellet, initialize_entities
# from game.maze_generator import Map

# class PacManEnv:
#     def __init__(self, width=29, height=31, seed=1):
#         self.maze = Map(w=width, h=height, seed=seed)
#         self.maze.generate_connected_maze(path_density=0.7)
#         self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
#         self.done = False
#         self.action_space = [0, 1, 2, 3]  # 0: up, 1: down, 2: left, 3: right
#         self.observation_space = (width, height, 5)  # (x, y, pellet, score_pellet, ghost)

#     def reset(self):
#         self.maze.generate_connected_maze(path_density=0.7)
#         self.pacman, self.ghosts, self.power_pellets, self.score_pellets = initialize_entities(self.maze)
#         self.done = False
#         return self._get_state()

#     def _get_state(self):
#         # 簡單狀態表示：Pac-Man 位置、最近目標（球或鬼魂）、鬼魂位置
#         state = np.zeros((self.maze.w, self.maze.h, 5))  # 5 通道：Pac-Man, PowerPellet, ScorePellet, EdibleGhost, NonEdibleGhost
#         state[self.pacman.x, self.pacman.y, 0] = 1  # Pac-Man 位置
#         for pellet in self.power_pellets:
#             state[pellet.x, pellet.y, 1] = 1  # 能量球
#         for pellet in self.score_pellets:
#             state[pellet.x, pellet.y, 2] = 1  # 分數球
#         for ghost in self.ghosts:
#             if ghost.edible and ghost.respawn_timer > 0:
#                 state[ghost.x, ghost.y, 3] = 1  # 可吃鬼魂
#             else:
#                 state[ghost.x, ghost.y, 4] = 1  # 不可吃鬼魂
#         return state

#     def step(self, action):
#         # 執行行動
#         dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
#         moving = self.pacman.set_new_target(dx, dy, self.maze)
#         if moving:
#             if self.pacman.move_towards_target(self.maze):
#                 pass  # 移動完成，狀態更新

#         # 獎勵計算
#         reward = -1  # 默認每步懲罰 -1
#         if self.pacman.eat_pellet(self.power_pellets) > 0:
#             reward = 10
#         if self.pacman.eat_score_pellet(self.score_pellets) > 0:
#             reward = 2

#         # 檢查鬼魂碰撞
#         for ghost in self.ghosts:
#             if self.pacman.x == ghost.x and self.pacman.y == ghost.y:
#                 if ghost.edible and ghost.respawn_timer > 0:
#                     reward = [50, 100, 150, 200][min(ghost.death_count, 3)]
#                     ghost.set_returning_to_spawn(30)
#                 elif not ghost.returning_to_spawn and not ghost.waiting:
#                     reward = -100
#                     self.done = True

#         # 更新鬼魂狀態
#         for ghost in self.ghosts:
#             if ghost.move_towards_target(self.maze):
#                 if ghost.returning_to_spawn and self.maze.get_tile(ghost.x, ghost.y) == 'S':
#                     ghost.set_waiting(30)
#                 else:
#                     ghost.move(self.pacman, self.maze, 30)

#         return self._get_state(), reward, self.done

#     def render(self):
#         pass  # 這裡不實現渲染，交給 main.py
