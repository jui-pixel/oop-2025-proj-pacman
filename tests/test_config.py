# test_config.py
import pytest
from config import *

def test_maze_dimensions():
    assert MAZE_WIDTH == 21
    assert MAZE_HEIGHT == 21
    assert CELL_SIZE == 30

def test_tile_types():
    assert TILE_BOUNDARY == '#'
    assert TILE_WALL == 'X'
    assert TILE_PATH == '.'
    assert TILE_POWER_PELLET == 'E'
    assert TILE_GHOST_SPAWN == 'S'

def test_speed_constants():
    assert PACMAN_BASE_SPEED == 100.0
    assert PACMAN_AI_SPEED == 125.0
    assert GHOST_DEFAULT_SPEED == 50
    assert GHOST_RETURN_SPEED == 150

def test_color_definitions():
    assert BLACK == (0, 0, 0)
    assert WHITE == (255, 255, 255)
    assert YELLOW == (255, 255, 0)
    assert RED == (255, 0, 0)