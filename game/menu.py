import sys
import pygame
import json
import os
import config
from config import *
import importlib

try:
    import torch
    PYTORCH_AVAILABLE = True
    PYTORCH_VERSION = torch.__version__
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "N/A"
except ImportError:
    PYTORCH_AVAILABLE = False
    PYTORCH_VERSION = "Not Installed"
    CUDA_AVAILABLE = False
    CUDA_DEVICE = "N/A"
BACKGROUND = pygame.Surface((500, 500))
BACKGROUND.fill(DARK_GRAY_BLUE)
for x in range(0, 500, 10):
    for y in range(0, 500, 10):
        if (x // 10 + y // 10) % 2 == 0:
            pygame.draw.rect(BACKGROUND, (255, 255, 255, 50), (x, y, 10, 10))
class MenuButton:
    """
    定義選單按鈕類，包含位置、文字和樣式。

    原理：
    - 每個按鈕是一個可交互的矩形區域，支援滑鼠懸停和鍵盤選擇。
    - 按鈕具有兩種顏色狀態：未懸停（inactive_color）和懸停（active_color）。
    - 提供繪製和滑鼠懸停檢測功能，用於選單交互。
    """
    def __init__(self, text, x, y, width, height, font, inactive_color, active_color):
        """
        初始化按鈕，設置文字、位置和樣式。

        Args:
            text (str): 按鈕顯示的文字。
            x (int): 按鈕左上角 x 坐標。
            y (int): 按鈕左上角 y 坐標。
            width (int): 按鈕寬度。
            height (int): 按鈕高度。
            font (pygame.font.Font): 文字渲染的字體。
            inactive_color (tuple): 未懸停時的顏色 (R, G, B)。
            active_color (tuple): 懸停時的顏色 (R, G, B)。
        """
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)  # 按鈕的矩形區域
        self.font = font
        self.inactive_color = inactive_color
        self.active_color = active_color
        self.is_hovered = False  # 是否被滑鼠懸停

    def draw(self, screen):
        """
        繪製按鈕，根據滑鼠懸停狀態改變顏色。

        原理：
        - 若按鈕被懸停，使用 active_color；否則使用 inactive_color。
        - 繪製圓角矩形（border_radius=10）作為按鈕背景。
        - 將文字渲染為表面並置於矩形中心。

        Args:
            screen (pygame.Surface): 繪製目標的螢幕表面。
        """
        shadow_rect = self.rect.copy()
        shadow_rect.move_ip(3, 3)
        pygame.draw.rect(screen, (50, 50, 50, 100), shadow_rect, border_radius=10)
        color = self.active_color if self.is_hovered else self.inactive_color
        scale = 1.05 if self.is_hovered else 1.0
        scaled_rect = self.rect.copy()
        scaled_rect.inflate_ip(int(self.rect.width * (scale - 1)), int(self.rect.height * (scale - 1)))
        pygame.draw.rect(screen, color, scaled_rect, border_radius=10)  # 繪製圓角矩形
        pygame.draw.rect(screen, WHITE, scaled_rect, width=2, border_radius=10)
        text_surface = self.font.render(self.text, True, WHITE)  # 渲染文字
        screen.blit(text_surface, text_surface.get_rect(center=self.rect.center))  # 置中顯示文字

    def check_hover(self, mouse_pos):
        """
        檢查滑鼠是否懸停在按鈕上。

        原理：
        - 使用矩形的 collidepoint 方法檢測滑鼠位置是否在按鈕區域內。
        - 更新 is_hovered 屬性，影響按鈕顏色。

        Args:
            mouse_pos (tuple): 滑鼠位置 (x, y)。
        """
        self.is_hovered = self.rect.collidepoint(mouse_pos)

def show_menu(screen, font, screen_width, screen_height):
    """
    顯示初始選單並返回選擇的模式。

    原理：
    - 創建六個按鈕，分別對應玩家模式、規則 AI 模式、DQN AI 模式、排行榜、設定和退出。
    - 支援鍵盤（上下鍵選擇，Enter 確認）和滑鼠（懸停、點擊）交互。
    - 按鈕垂直排列，預設第一個按鈕被選中（is_hovered=True）。
    - 返回對應的模式字串，例如 "player" 或 "exit"。

    Args:
        screen (pygame.Surface): 繪製目標的螢幕表面。
        font (pygame.font.Font): 文字渲染的字體。
        screen_width (int): 螢幕寬度。
        screen_height (int): 螢幕高度。

    Returns:
        str: 選擇的模式（"player", "rule_ai", "dqn_ai", "leaderboard", "settings", "exit"）。
    """
    buttons = [MenuButton(mode, screen_width // 2 - 100, 120 + i * 60, 200, 50, font, GRAY, LIGHT_BLUE) 
               for i, mode in enumerate(["Player Mode", "Rule AI Mode", "DQN AI Mode", "Leaderboard", "Settings", "Exit"])]
    selected_index = 0
    buttons[selected_index].is_hovered = True  # 預設第一個按鈕被選中

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index + 1) % len(buttons)  # 向下循環選擇
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_UP:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index - 1) % len(buttons)  # 向上循環選擇
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_RETURN:
                    return ["player", "rule_ai", "dqn_ai", "leaderboard", "settings", "exit"][selected_index]  # 確認選擇
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    button.check_hover(mouse_pos)  # 更新懸停狀態
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.rect.collidepoint(mouse_pos):
                        return ["player", "rule_ai", "dqn_ai", "leaderboard", "settings", "exit"][i]  # 滑鼠點擊選擇

        screen.fill(BLACK)  # 清空螢幕
        title = font.render("Pac-Man Menu", True, YELLOW)  # 渲染標題
        screen.blit(title, (screen_width // 2 - title.get_width() // 2, 30))  # 置中顯示標題

        load_menu = pygame.image.load(f"./assert/image/menu.png").convert_alpha()  # 加載選單背景圖片
        load_menu = pygame.transform.scale(load_menu, (screen_width, screen_height))  # 調整圖片大小
        screen.blit(load_menu, (0, 0))  # 繪製背景
        for button in buttons:
            button.draw(screen)  # 繪製所有按鈕
        pygame.display.flip()  # 更新螢幕

def show_leaderboard(screen, font, screen_width, screen_height):
    """
    顯示排行榜。

    原理：
    - 從 scores.json 讀取分數記錄，按分數降序排列，顯示前 10 名。
    - 每條記錄包括玩家名稱、分數、迷宮種子和遊玩時間（格式為 mm:ss）。
    - 時間格式化公式：minutes = time // 60, seconds = time % 60。
    - 支援 ESC 鍵返回主選單，若檔案不存在則顯示空排行榜。

    Args:
        screen (pygame.Surface): 繪製目標的螢幕表面。
        font (pygame.font.Font): 文字渲染的字體。
        screen_width (int): 螢幕寬度。
        screen_height (int): 螢幕高度。
    """
    records = []
    try:
        with open("scores.json", "r") as f:
            records = json.load(f)
            records = sorted(records, key=lambda x: x["score"], reverse=True)[:10]  # 按分數降序取前 10
    except FileNotFoundError:
        pass  # 若檔案不存在，顯示空排行榜

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return  # 返回主選單
        screen.fill(BLACK)
        load_leaderboard = pygame.image.load(f"./assert/image/leaderboard.png").convert_alpha()  # 加載排行榜背景圖片
        load_leaderboard = pygame.transform.scale(load_leaderboard, (screen_width, screen_height))  # 調整圖片大小
        screen.blit(load_leaderboard, (0, 0))

        small_font = pygame.font.SysFont(None,29)  # 使用較小的字體顯示排行榜
        for i, record in enumerate(records):
            text = small_font.render(f"{record['name']}: ", True, YELLOW)
            text_score = small_font.render(f" {record['score']} (Seed: {record['seed']}, Time: {int(record['time']//60):02d}:{int(record['time']%60):02d})", True, WHITE)
            screen.blit(text, (screen_width // 4, 142 + i * 33.5))  # 顯示每條記錄
            screen.blit(text_score, (screen_width // 4 + 120, 142 + i * 33.5))  # 顯示分數、種子和時間
        pygame.display.flip()

def update_config_seed(new_seed):
    """
    更新 config.py 中的 MAZE_SEED 值並儲存到檔案，然後重新加載 config 模組。

    Args:
        new_seed (int): 新的迷宮種子值。
    """
    global MAZE_SEED
    MAZE_SEED = new_seed
    try:
        with open("config.py", "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open("config.py", "w", encoding="utf-8") as f:
            for line in lines:
                if line.strip().startswith("MAZE_SEED ="):
                    f.write(f"MAZE_SEED = {new_seed}\n")
                else:
                    f.write(line)
        importlib.reload(config)  # 重新加載 config 模組
        # 更新 menu 模組中的 MAZE_SEED
        globals()['MAZE_SEED'] = config.MAZE_SEED
    except Exception as e:
        print(f"Error updating config.py: {e}")
        
def show_settings(screen, font, screen_width, screen_height):
    """
    顯示設定頁面，包含 FPS、迷宮尺寸、種子、DQN 模型、PyTorch 和 CUDA 狀態，並允許修改 MAZE_SEED。

    原理：
    - 從 config 模組讀取 FPS、MAZE_WIDTH、MAZE_HEIGHT、MAZE_SEED。
    - 檢查 pacman_dqn.pth 是否存在，判斷 DQN 模型是否可用。
    - 檢查 PyTorch 是否安裝（版本）以及 CUDA 是否可用（設備名稱）。
    - 垂直排列顯示參數，標題為黃色，內容為白色，與主選單風格一致。
    - 添加兩個按鈕（+1 和 -1）用於調整 MAZE_SEED，種子值不得小於 1。
    - 支援滑鼠點擊和鍵盤（上下鍵選擇，Enter 確認，左右鍵調整種子）交互。
    - 支援 ESC 鍵返回主選單，提示文字顯示於底部。

    Args:
        screen (pygame.Surface): 繪製目標的螢幕表面。
        font (pygame.font.Font): 文字渲染的字體。
        screen_width (int): 螢幕寬度。
        screen_height (int): 螢幕高度。
    """
    # 檢查 DQN 模型檔案
    dqn_available = "Available" if os.path.exists("pacman_dqn.pth") else "Not Available"

    # 創建按鈕：增加和減少 MAZE_SEED
    seed_plus_button = MenuButton("+1", screen_width // 2 + 120, 100 + 3 * 40, 50, 40, font, GRAY, LIGHT_BLUE)
    seed_minus_button = MenuButton("-1", screen_width // 2 + 180, 100 + 3 * 40, 50, 40, font, GRAY, LIGHT_BLUE)
    buttons = [seed_plus_button, seed_minus_button]
    selected_index = 0
    buttons[selected_index].is_hovered = True  # 預設第一個按鈕被選中

    # 本地變數追蹤當前種子值
    current_seed = MAZE_SEED

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return  # 返回主選單
                elif event.key == pygame.K_DOWN:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index + 1) % len(buttons)  # 向下循環選擇
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_UP:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index - 1) % len(buttons)  # 向上循環選擇
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_RETURN:
                    if buttons[selected_index] == seed_plus_button:
                        current_seed += 1
                        update_config_seed(current_seed)
                    elif buttons[selected_index] == seed_minus_button and current_seed > 1:
                        current_seed -= 1
                        update_config_seed(current_seed)
                elif event.key == pygame.K_RIGHT:
                    current_seed += 1
                    update_config_seed(current_seed)
                elif event.key == pygame.K_LEFT and current_seed > 1:
                    current_seed -= 1
                    update_config_seed(current_seed)
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    button.check_hover(mouse_pos)  # 更新懸停狀態
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    if button.rect.collidepoint(mouse_pos):
                        if button == seed_plus_button:
                            current_seed += 1
                            update_config_seed(current_seed)
                        elif button == seed_minus_button and current_seed > 1:
                            current_seed -= 1
                            update_config_seed(current_seed)

        # 更新設定清單
        settings = [
            f"FPS: {FPS}",
            f"Maze Width: {MAZE_WIDTH}",
            f"Maze Height: {MAZE_HEIGHT}",
            f"Maze Seed: {current_seed}",
            f"DQN Model: {dqn_available}",
            f"PyTorch: {PYTORCH_VERSION}",
            f"CUDA: {f'{CUDA_DEVICE}' if CUDA_AVAILABLE else 'Not Available'}"
        ]

        screen.fill(BLACK)  # 清空螢幕
        
        load_settings = pygame.image.load(f"./assert/image/setting.png").convert_alpha()  # 加載設定背景圖片
        load_settings = pygame.transform.scale(load_settings, (screen_width, screen_height))  # 調整圖片大小
        screen.blit(load_settings, (0, 0))  #

        # 繪製設定清單
        for i, setting in enumerate(settings):
            text = font.render(setting, True, WHITE)
            screen.blit(text, (screen_width // 2 - text.get_width() // 2, 107 + i * 40))

        # 繪製按鈕
        for button in buttons:
            button.draw(screen)

        pygame.display.flip()  # 更新螢幕

def save_score(name, score, seed, play_time):
    """
    儲存分數數據到 scores.json。

    原理：
    - 將新分數記錄（名稱、分數、種子、遊玩時間）添加到 scores.json。
    - 若同一玩家名稱已存在，僅保留分數最高的記錄。
    - 使用字典 best_scores 去重，鍵為清理後的玩家名稱，值為最高分數記錄。
    - 記錄格式：{"name": str, "score": int, "seed": int, "time": float}。

    Args:
        name (str): 玩家名稱。
        score (int): 遊戲分數。
        seed (int): 迷宮種子。
        play_time (float): 遊玩時間（秒）。
    """
    records = []
    try:
        with open("scores.json", "r") as f:
            records = json.load(f)  # 讀取現有記錄
    except FileNotFoundError:
        pass  # 若檔案不存在，創建新記錄

    new_record = {"name": name.strip(), "score": score, "seed": seed, "time": play_time} 
    best_scores = {}
    for record in records:
        cleaned_existing_name = record["name"].strip()
        best_scores[cleaned_existing_name] = record

    cleaned_new_name = new_record["name"]
    if cleaned_new_name in best_scores:
        if new_record["score"] > best_scores[cleaned_new_name]["score"]:
            best_scores[cleaned_new_name] = new_record  # 更新更高分數
    else:
        best_scores[cleaned_new_name] = new_record  # 添加新記錄
    updated_records = list(best_scores.values())
    with open("scores.json", "w") as f:
        json.dump(updated_records, f)  # 儲存更新記錄

def get_player_name(screen, font, screen_width, screen_height, default_name="Player"):
    """
    獲取玩家名稱輸入。

    原理：
    - 顯示輸入框，允許玩家輸入名稱（最多 15 個字元）。
    - 支援鍵盤輸入（可打印字元）、刪除（Backspace）、確認（Enter）和取消（ESC）。
    - 若 default_name 為 "RuleAI" 或 "DQNAI"，直接返回該名稱（跳過輸入）。
    - 輸入框為固定大小矩形，文字左對齊顯示。

    Args:
        screen (pygame.Surface): 繪製目標的螢幕表面。
        font (pygame.font.Font): 文字渲染的字體。
        screen_width (int): 螢幕寬度。
        screen_height (int): 螢幕高度。
        default_name (str): 預設玩家名稱（預設為 "Player"）。

    Returns:
        str: 清理後的玩家名稱（去除首尾空格）。
    """
    if default_name in ["RuleAI", "DQNAI"]:
        return default_name  # 直接返回 AI 名稱
    name = default_name
    input_rect = pygame.Rect(screen_width // 2 - 150, screen_height // 2, 300, 50)  # 輸入框矩形
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and name.strip():
                    return name.strip()  # 確認輸入
                elif event.key == pygame.K_BACKSPACE:
                    name = name[:-1]  # 刪除最後一個字元
                elif event.unicode.isprintable() and len(name) < 15:
                    name += event.unicode  # 添加可打印字元
                elif event.key == pygame.K_ESCAPE:
                    return name  # 取消輸入
        screen.fill(BLACK)
        player_name_image = pygame.image.load(f"./assert/image/player_name.png").convert_alpha()  # 加載玩家名稱圖片
        player_name_image = pygame.transform.scale(player_name_image, (screen_width, screen_height))  # 調整圖片大小
        screen.blit(player_name_image, (0, 0))  # 繪製背景圖片

        pygame.draw.rect(screen, WHITE, input_rect, 2)  # 繪製輸入框邊框
        name_surface = font.render(name, True, WHITE)
        screen.blit(name_surface, (input_rect.x + 5, input_rect.y + 5))  # 顯示輸入文字
        pygame.display.flip()

def show_loading_screen(screen, font, screen_width, screen_height):
    """
    顯示加載畫面。

    原理：
    - 顯示簡單的 "Loading..." 文字，持續 1 秒（1000 毫秒）。
    - 用於模式切換或遊戲初始化時，提供視覺反饋。

    Args:
        screen (pygame.Surface): 繪製目標的螢幕表面。
        font (pygame.font.Font): 文字渲染的字體。
        screen_width (int): 螢幕寬度。
        screen_height (int): 螢幕高度。
    """
    screen.fill(BLACK)
    loading3_image = pygame.image.load(f"./assert/image/loading/loading3.png").convert_alpha()  # 加載加載圖片
    screen.blit(loading3_image, (0, 0))  # 繪製背景圖片
    pygame.display.flip()
    pygame.time.wait(400)  # 延遲 0.4 秒
    loading2_image = pygame.image.load(f"./assert/image/loading/loading2.png").convert_alpha()  # 加載加載圖片
    screen.blit(loading2_image, (0, 0))  # 繪製背景圖片
    pygame.display.flip()
    pygame.time.wait(400)  # 延遲 0.4 秒
    loading1_image = pygame.image.load(f"./assert/image/loading/loading1.png").convert_alpha()  # 加載加載圖片
    screen.blit(loading1_image, (0, 0))  # 繪製背景圖片
    pygame.display.flip()
    pygame.time.wait(400)  # 延遲 0.4 秒
    loading0_image = pygame.image.load(f"./assert/image/loading/loading0.png").convert_alpha()  # 加載加載圖片
    screen.blit(loading0_image, (0, 0))  # 繪製背景圖片
    pygame.display.flip()
    pygame.time.wait(200)  # 延遲 0.2 秒

def show_game_result(screen, font, screen_width, screen_height, won, score):
    """
    顯示遊戲結束畫面。

    原理：
    - 顯示遊戲結果（"You Win!" 或 "Game Over!"）和最終分數。
    - 提供三個按鈕：返回選單、重啟遊戲、退出。
    - 支援鍵盤（上下鍵選擇，Enter 確認）和滑鼠（懸停、點擊）交互。
    - 根據 won 參數選擇不同的文字顏色（勝利為綠色，失敗為紅色）。

    Args:
        screen (pygame.Surface): 繪製目標的螢幕表面。
        font (pygame.font.Font): 文字渲染的字體。
        screen_width (int): 螢幕寬度。
        screen_height (int): 螢幕高度。
        won (bool): 是否贏得遊戲。
        score (int): 最終分數。

    Returns:
        str: 選擇的動作（"menu", "restart", "exit"）。
    """
    buttons = [MenuButton(action, screen_width // 2 - 100, screen_height // 2 + 50 + i * 60, 200, 50, font, GRAY, LIGHT_BLUE) 
               for i, action in enumerate(["Back to Menu", "Restart", "Exit"])]
    selected_index = 0
    buttons[selected_index].is_hovered = True  # 預設第一個按鈕被選中

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index + 1) % len(buttons)  # 向下循環選擇
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_UP:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index - 1) % len(buttons)  # 向上循環選擇
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_RETURN:
                    return ["menu", "restart", "exit"][selected_index]  # 確認選擇
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    button.check_hover(mouse_pos)  # 更新懸停狀態
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.rect.collidepoint(mouse_pos):
                        return ["menu", "restart", "exit"][i]  # 滑鼠點擊選擇
        screen.fill(BLACK)
        winnig_image = pygame.image.load(f"./assert/image/win.png").convert_alpha()
        losing_image = pygame.image.load(f"./assert/image/lose.png").convert_alpha()
        result_image = winnig_image if won else losing_image
        screen.blit(result_image, (0, 0))  # 繪製結果
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (screen_width // 2 - score_text.get_width() // 2, screen_height // 2 - 50))  # 顯示分數
        for button in buttons:
            button.draw(screen)  # 繪製按鈕
        pygame.display.flip()

def show_pause_menu(screen, font, screen_width, screen_height):
    """
    顯示暫停選單，使用定格的遊戲畫面作為半透明背景。

    原理：
    - 複製當前遊戲畫面作為背景，添加半透明黑色覆蓋層（透明度 180）。
    - 顯示三個按鈕：繼續遊戲、返回選單、退出。
    - 支援鍵盤（上下鍵選擇，Enter 確認）和滑鼠（懸停、點擊）交互。
    - 返回選擇的動作，用於遊戲主循環處理。

    Args:
        screen (pygame.Surface): 繪製目標的螢幕表面。
        font (pygame.font.Font): 文字渲染的字體。
        screen_width (int): 螢幕寬度。
        screen_height (int): 螢幕高度。

    Returns:
        str: 選擇的動作（"continue", "menu", "exit"）。
    """
    # 複製當前螢幕內容作為背景
    background = screen.copy()
    
    # 創建半透明覆蓋層
    overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))  # 黑色，透明度 180 (0=完全透明, 255=完全不透明)
    
    # 將覆蓋層應用到背景上以使其變暗
    background.blit(overlay, (0, 0))

    buttons = [MenuButton(action, screen_width // 2 - 100, screen_height // 2 - 30 + i * 60, 200, 50, font, GRAY, LIGHT_BLUE) 
               for i, action in enumerate(["Continue", "Back to Menu", "Exit"])]
    selected_index = 0
    buttons[selected_index].is_hovered = True  # 預設第一個按鈕被選中

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index + 1) % len(buttons)  # 向下循環選擇
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_UP:
                    buttons[selected_index].is_hovered = False
                    selected_index = (selected_index - 1) % len(buttons)  # 向上循環選擇
                    buttons[selected_index].is_hovered = True
                elif event.key == pygame.K_RETURN:
                    return ["continue", "menu", "exit"][selected_index]  # 確認選擇
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                for button in buttons:
                    button.check_hover(mouse_pos)  # 更新懸停狀態
            elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.rect.collidepoint(mouse_pos):
                        return ["continue", "menu", "exit"][i]  # 滑鼠點擊選擇
        
        # 繪製定格的背景
        screen.blit(background, (0, 0))

        paused_image = pygame.image.load(f"./assert/image/paused.png").convert_alpha()  # 加載暫停圖片
        paused_image = pygame.transform.scale(paused_image, (screen_width, screen_height))
        screen.blit(paused_image, (0, 0))  # 繪製暫停背景圖片
        for button in buttons:
            button.draw(screen)  # 繪製按鈕
        pygame.display.flip()