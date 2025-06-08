# test_maze_generator.py
import pytest
from game.maze_generator import Map
from config import MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED

def test_map_initialization():
    maze = Map(MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED)
    assert maze.width == MAZE_WIDTH
    assert maze.height == MAZE_HEIGHT
    assert maze.get_tile(0, 0) == '#'

def test_generate_maze():
    maze = Map(MAZE_WIDTH, MAZE_HEIGHT, MAZE_SEED)
    maze.generate_maze()
    assert any(maze.get_tile(x, y) == 'E' for y in range(MAZE_HEIGHT) for x in range(MAZE_WIDTH))