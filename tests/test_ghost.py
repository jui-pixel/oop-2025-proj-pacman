# test_ghost.py
import pytest
from unittest.mock import Mock, patch
from game.entities.ghost import Ghost

@pytest.fixture
def mock_maze():
    maze = Mock()
    maze.xy_valid.return_value = True
    maze.get_tile.return_value = '.'
    return maze

def test_ghost_move(mock_maze):
    ghost = Ghost(1, 1)
    pacman = Mock()
    ghost.move(pacman, mock_maze, 60)
    assert not ghost.waiting
    assert not ghost.returning_to_spawn

def test_ghost_set_edible(mock_maze):
    ghost = Ghost(1, 1)
    ghost.set_edible(60)
    assert ghost.edible
    assert ghost.edible_timer == 60