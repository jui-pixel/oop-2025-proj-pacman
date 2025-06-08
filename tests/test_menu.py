# test_menu.py
import pytest
import pygame
from unittest.mock import Mock, patch
from game.menu import MenuButton, show_menu

@pytest.fixture
def setup_screen():
    pygame.init()
    screen = pygame.display.set_mode((630, 630))  # 21*30
    font = pygame.font.SysFont(None, 36)
    yield screen, font
    pygame.quit()

def test_menu_button_draw(setup_screen):
    screen, font = setup_screen
    button = MenuButton("Test", 100, 100, 200, 50, font, (128, 128, 128), (173, 216, 230))
    button.draw(screen)
    assert True  # 檢查無異常

def test_show_menu_selection(setup_screen):
    screen, font = setup_screen
    with patch('pygame.event.get', return_value=[Mock(type=pygame.KEYDOWN, key=pygame.K_RETURN)]):
        mode = show_menu(screen, font, 630, 630)
        assert mode == "player"  # 預設第一個選項