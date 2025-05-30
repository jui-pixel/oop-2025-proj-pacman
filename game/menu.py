import sys
import pygame
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
        screen.blit(text_surface, text_surface.get_rect(center=self.rect.center))

    def check_hover(self, mouse_pos):
        """檢查滑鼠是否懸停在按鈕上。"""
        self.is_hovered = self.rect.collidepoint(mouse_pos)

def show_menu(screen, font, screen_width, screen_height):
    """顯示初始選單並返回選擇的模式。"""
    buttons = [MenuButton(mode, screen_width // 2 - 100, 100 + i * 60, 200, 50, font, GRAY, LIGHT_BLUE) 
              for i, mode in enumerate(["Player Mode", "Rule AI Mode", "DQN AI Mode", "Leaderboard", "Settings", "Exit"])]
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
    """顯示排行榜。"""
    records = []
    try:
        with open("scores.json", "r") as f:
            records = json.load(f)
            records = sorted(records, key=lambda x: x["score"], reverse=True)[:10]
    except FileNotFoundError:
        pass

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
            text = font.render(f"{i+1}. {record['name']}: {record['score']} (Seed: {record['seed']}, Time: {int(record['time']//60):02d}:{int(record['time']%60):02d})", True, WHITE)
            screen.blit(text, (screen_width // 2 - text.get_width() // 2, 100 + i * 40))
        back_text = font.render("Press ESC to return", True, WHITE)
        screen.blit(back_text, (screen_width // 2 - back_text.get_width() // 2, screen_height - 50))
        pygame.display.flip()

def show_settings(screen, font, screen_width, screen_height):
    """顯示設定頁面（占位符）。"""
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
    """儲存分數數據到 scores.json。"""
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
    """獲取玩家名稱輸入。"""
    if default_name in ["RuleAI", "DQNAI"]:
        return default_name
    name = default_name
    input_rect = pygame.Rect(screen_width // 2 - 150, screen_height // 2, 300, 50)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and name.strip():
                    return name.strip()
                elif event.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                elif event.unicode.isprintable() and len(name) < 15:
                    name += event.unicode
                elif event.key == pygame.K_ESCAPE:
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
    """顯示加載畫面。"""
    screen.fill(BLACK)
    loading_text = font.render("Loading...", True, YELLOW)
    screen.blit(loading_text, (screen_width // 2 - loading_text.get_width() // 2, screen_height // 2))
    pygame.display.flip()
    pygame.time.wait(2000)

def show_game_result(screen, font, screen_width, screen_height, won, score):
    """顯示遊戲結束畫面。"""
    buttons = [MenuButton(action, screen_width // 2 - 100, screen_height // 2 + 50 + i * 60, 200, 50, font, GRAY, LIGHT_BLUE) 
              for i, action in enumerate(["Back to Menu", "Restart", "Exit"])]
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
                    return ["menu", "restart", "exit"][selected_index]
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    button.check_hover(mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.rect.collidepoint(mouse_pos):
                        return ["menu", "restart", "exit"][i]
        screen.fill(BLACK)
        result_text = font.render("You Win!" if won else "Game Over!", True, GREEN if won else RED)
        screen.blit(result_text, (screen_width // 2 - result_text.get_width() // 2, screen_height // 2 - 100))
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (screen_width // 2 - score_text.get_width() // 2, screen_height // 2 - 50))
        for button in buttons:
            button.draw(screen)
        pygame.display.flip()

def show_pause_menu(screen, font, screen_width, screen_height):
    """顯示暫停選單。"""
    buttons = [MenuButton(action, screen_width // 2 - 100, screen_height // 2 - 30 + i * 60, 200, 50, font, GRAY, LIGHT_BLUE) 
              for i, action in enumerate(["Continue", "Back to Menu", "Exit"])]
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
                    return ["continue", "menu", "exit"][selected_index]
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    button.check_hover(mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.rect.collidepoint(mouse_pos):
                        return ["continue", "menu", "exit"][i]
        screen.fill(BLACK)
        pause_text = font.render("Paused", True, YELLOW)
        screen.blit(pause_text, (screen_width // 2 - pause_text.get_width() // 2, screen_height // 2 - 100))
        for button in buttons:
            button.draw(screen)
        pygame.display.flip()