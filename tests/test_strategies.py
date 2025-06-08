# test_strategies.py
import pytest
from unittest.mock import Mock, patch
from game.strategies import RuleBasedAIControl, DQNAIControl, ControlManager

def test_rule_based_ai_move():
    ai = RuleBasedAIControl()
    pacman = Mock()
    pacman.move_towards_target.return_value = True
    pacman.rule_based_ai_move.return_value = True
    result = ai.move(pacman, Mock(), [], [], [], False)
    assert result

def test_control_manager_switch():
    manager = ControlManager(21, 21)
    initial_mode = manager.get_mode_name()
    manager.switch_mode()
    assert manager.get_mode_name() != initial_mode