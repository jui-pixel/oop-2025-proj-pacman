import sys
import pygame
import math
import json
from config import *

class MenuButton:
    """定義選單按鈕類，包含位置、文字和樣式。"""
    def __init__(self, text, x, y, width, height, font, inactive_color, active_color):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.inactive_color = inactive_color
        self.active_color = active_color
        self.is_hovered = False

    def draw(self, screen):
        """繪製按鈕，根據滑鼠懸停狀態改變顏色。"""
        color = self.active_color if self.is_hovered else self.inactive_color
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        text_surface = self.font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def check_hover(self, mouse_pos):
        """檢查滑鼠是否懸停在按鈕上。"""
        self.is_hovered = self.rect.collidepoint(mouse_pos)

def show_menu(screen, font, screen_width, screen_height):
    """
    顯示初始選單，讓使用者選擇遊戲模式。

    Args:
        screen: Pygame 畫面物件。
        font: Pygame 字體物件。
        screen_width: 畫面寬度。
        screen_height: 畫面高度。

    Returns:
        str: 選擇的模式（"player", "rule_ai", "dqn_ai", "leaderboard", "settings", "exit"）。
    """
    buttons = [
        MenuButton("Player Mode", screen_width // 2 - 100, 100, 200, 50, font, GRAY, LIGHT_BLUE),
        MenuButton("Rule AI Mode", screen_width // 2 - 100, 160, 200, 50, font, GRAY, LIGHT_BLUE),
        MenuButton("DQN AI Mode", screen_width // 2 - 100, 220, 200, 50, font, GRAY, LIGHT_BLUE),
        MenuButton("Leaderboard", screen_width // 2 - 100, 280, 200, 50, font, GRAY, LIGHT_BLUE),
        MenuButton("Settings", screen_width // 2 - 100, 340, 200, 50, font, GRAY, LIGHT_BLUE),
        MenuButton("Exit", screen_width // 2 - 100, 400, 200, 50, font, GRAY, LIGHT_BLUE),
    ]
    selected_index = 0
    buttons[selected_index].is_hovered = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index + 1) % len(buttons)
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_UP:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index - 1) % len(buttons)
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_RETURN:
                    return ["player", "rule_ai", "dqn_ai", "leaderboard", "settings", "exit"][selected_index]
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    button.check_hover(mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.rect.collidepoint(mouse_pos):
                        return ["player", "rule_ai", "dqn_ai", "leaderboard", "settings", "exit"][i]

        screen.fill(BLACK)
        title = font.render("Pac-Man Menu", True, YELLOW)
        screen.blit(title, (screen_width // 2 - title.get_width() // 2, 30))
        for button in buttons:
            button.draw(screen)
        pygame.display.flip()

def show_leaderboard(screen, font, screen_width, screen_height):
    """
    顯示排行榜，從 scores.json 讀取歷史分數、名稱、種子和遊玩時長。

    Args:
        screen: Pygame 畫面物件。
        font: Pygame 字體物件。
        screen_width: 畫面寬度。
        screen_height: 畫面高度。
    """
    records = []
    try:
        with open("scores.json", "r") as f:
            records = json.load(f)
            records = sorted(records, key=lambda x: x["score"], reverse=True)[:10]  # 按分數排序，取前 10 名
    except FileNotFoundError:
        records = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

        screen.fill(BLACK)
        title = font.render("Leaderboard", True, YELLOW)
        screen.blit(title, (screen_width // 2 - title.get_width() // 2, 30))

        for i, record in enumerate(records):
            name = record["name"]
            score = record["score"]
            seed = record["seed"]
            time = record["time"]
            # 將遊玩時長轉換為分:秒格式
            minutes = int(time // 60)
            seconds = int(time % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            record_text = font.render(f"{i+1}. {name}: {score} (Seed: {seed}, Time: {time_str})", True, WHITE)
            screen.blit(record_text, (screen_width // 2 - record_text.get_width() // 2, 100 + i * 40))

        back_text = font.render("Press ESC to return", True, WHITE)
        screen.blit(back_text, (screen_width // 2 - back_text.get_width() // 2, screen_height - 50))
        pygame.display.flip()

def show_settings(screen, font, screen_width, screen_height):
    """
    顯示設定頁面（占位符，未來可添加音量調整等功能）。

    Args:
        screen: Pygame 畫面物件。
        font: Pygame 字體物件。
        screen_width: 畫面寬度。
        screen_height: 畫面高度。
    """
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

        screen.fill(BLACK)
        title = font.render("Settings", True, YELLOW)
        screen.blit(title, (screen_width // 2 - title.get_width() // 2, 30))
        placeholder = font.render("Settings (Press ESC to return)", True, WHITE)
        screen.blit(placeholder, (screen_width // 2 - placeholder.get_width() // 2, screen_height // 2))
        pygame.display.flip()

def save_score(name, score, seed, play_time):
    """
    儲存分數、名稱、種子和遊玩時長到 scores.json。

    Args:
        name (str): 玩家名稱。
        score (int): 遊戲分數。
        seed (int): 迷宮種子。
        play_time (float): 遊玩時長（秒）。
    """
    records = []
    try:
        with open("scores.json", "r") as f:
            records = json.load(f)
    except FileNotFoundError:
        pass
    records.append({"name": name, "score": score, "seed": seed, "time": play_time})
    with open("scores.json", "w") as f:
        json.dump(records, f)

def get_player_name(screen, font, screen_width, screen_height, default_name="Player"):
    """
    顯示輸入框，讓使用者輸入玩家名稱。
    AI 模式（RuleAI 和 DQNAI）直接返回預設名稱，不進入輸入迴圈。

    Args:
        screen: Pygame 畫面物件。
        font: Pygame 字體物件。
        screen_width: 畫面寬度。
        screen_height: 畫面高度。
        default_name: 預設名稱（例如 "Player" 或 AI 名稱）。

    Returns:
        str: 玩家輸入的名稱。
    """
    # 如果是 AI 模式，直接返回預設名稱
    if default_name in ["RuleAI", "DQNAI"]:
        return default_name

    name = default_name
    input_active = True
    input_rect = pygame.Rect(screen_width // 2 - 150, screen_height // 2, 300, 50)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if input_active:
                    if event.key == pygame.K_RETURN:
                        if name.strip():
                            return name.strip()
                    elif event.key == pygame.K_BACKSPACE:
                        name = name[:-1]
                    elif event.unicode.isprintable() and len(name) < 15:
                        name += event.unicode
                if event.key == pygame.K_ESCAPE:
                    return name

        screen.fill(BLACK)
        prompt = font.render("Enter your name:", True, YELLOW)
        screen.blit(prompt, (screen_width // 2 - prompt.get_width() // 2, screen_height // 2 - 50))

        pygame.draw.rect(screen, WHITE, input_rect, 2)
        name_surface = font.render(name, True, WHITE)
        screen.blit(name_surface, (input_rect.x + 5, input_rect.y + 5))

        instruction = font.render("Press Enter to confirm, ESC to skip", True, WHITE)
        screen.blit(instruction, (screen_width // 2 - instruction.get_width() // 2, screen_height // 2 + 60))
        pygame.display.flip()
        
def show_loading_screen(screen, font, screen_width, screen_height):
    """
    顯示加載畫面，持續 2 秒。

    Args:
        screen: Pygame 畫面物件。
        font: Pygame 字體物件。
        screen_width: 畫面寬度。
        screen_height: 畫面高度。
    """
    screen.fill(BLACK)
    loading_text = font.render("Loading...", True, YELLOW)
    screen.blit(loading_text, (screen_width // 2 - loading_text.get_width() // 2, screen_height // 2))
    pygame.display.flip()
    pygame.time.wait(2000)  # 等待 2 秒
    
def show_game_result(screen, font, screen_width, screen_height, won, score):
    """
    顯示遊戲結束畫面（贏或輸），顯示分數，並提供返回主選單和退出遊戲的按鈕。

    Args:
        screen: Pygame 畫面物件。
        font: Pygame 字體物件。
        screen_width: 畫面寬度。
        screen_height: 畫面高度。
        won (bool): 是否贏得遊戲。
        score (int): 玩家最終分數。

    Returns:
        str: 選擇的動作（"menu" 或 "exit"）。
    """
    buttons = [
        MenuButton("Back to Menu", screen_width // 2 - 100, screen_height // 2 + 50, 200, 50, font, GRAY, LIGHT_BLUE),
        MenuButton("Exit", screen_width // 2 - 100, screen_height // 2 + 110, 200, 50, font, GRAY, LIGHT_BLUE),
    ]
    selected_index = 0
    buttons[selected_index].is_hovered = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index + 1) % len(buttons)
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_UP:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index - 1) % len(buttons)
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_RETURN:
                    return ["menu", "exit"][selected_index]
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    button.check_hover(mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.rect.collidepoint(mouse_pos):
                        return ["menu", "exit"][i]

        screen.fill(BLACK)
        result_text = font.render("You Win!" if won else "Game Over!", True, GREEN if won else RED)
        screen.blit(result_text, (screen_width // 2 - result_text.get_width() // 2, screen_height // 2 - 100))

        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (screen_width // 2 - score_text.get_width() // 2, screen_height // 2 - 50))

        for button in buttons:
            button.draw(screen)
        pygame.display.flip()
        