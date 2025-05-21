```mermaid
classDiagram
    class Entity {
        +int x
        +int y
        +str symbol
        +int target_x
        +int target_y
        +float current_x
        +float current_y
        +float speed
        +move_towards_target(maze) bool
        +set_new_target(dx, dy, maze) bool
    }

    class PacMan {
        +int score
        +bool alive
        +float speed
        +eat_pellet(pellets) int
        +eat_score_pellet(score_pellets) int
        +rule_based_ai_move(maze, power_pellets, score_pellets, ghosts) bool
    }
    PacMan --|> Entity

    class Ghost {
        +str name
        +Tuple~int~ color
        +float speed
        +bool edible
        +int edible_timer
        +int respawn_timer
        +bool returning_to_spawn
        +float return_speed
        +int death_count
        +bool waiting
        +int wait_timer
        +int alpha
        +move(pacman, maze, fps)
        +return_to_spawn(maze, fps)
        +escape_from_pacman(pacman, maze)
        +move_random(maze)
        +chase_pacman(pacman, maze)
        +set_edible(duration)
        +set_returning_to_spawn(fps)
        +set_waiting(fps)
        +reset_position(maze, respawn_points)
    }
    Ghost --|> Entity

    class BasicGhost {
        +bfs_path(start_x, start_y, target_x, target_y, maze) Optional[Tuple]
        +chase_pacman(pacman, maze, ghosts)
        +set_new_target(dx, dy, maze) bool
    }
    BasicGhost --|> Ghost

    class Ghost1 {
        +__init__(x, y, name="Ghost1")
        +chase_pacman(pacman, maze, ghosts)
    }
    Ghost1 --|> BasicGhost

    class Ghost2 {
        +__init__(x, y, name="Ghost2")
        +chase_pacman(pacman, maze)
    }
    Ghost2 --|> BasicGhost

    class Ghost3 {
        +__init__(x, y, name="Ghost3")
        +chase_pacman(pacman, maze)
    }
    Ghost3 --|> BasicGhost

    class Ghost4 {
        +__init__(x, y, name="Ghost4")
        +chase_pacman(pacman, maze)
    }
    Ghost4 --|> BasicGhost

    class PowerPellet {
        +int value
    }
    PowerPellet --|> Entity

    class ScorePellet {
        +int value
    }
    ScorePellet --|> Entity

    class Map {
        +int w
        +int h
        +List~str~ tiles
        +__init__(w, h, seed, tile_str)
        +setMap(w, h, tile_str)
        +__str__()
        +xy_to_i(x, y)
        +i_to_xy(i)
        +xy_valid(x, y)
        +get_tile(x, y)
        +set_tile(x, y, value)
        +generate_maze()
    }

    class Game {
        +Map maze
        +PacMan pacman
        +List~Ghost~ ghosts
        +List~PowerPellet~ power_pellets
        +List~ScorePellet~ score_pellets
        +List~Tuple~ respawn_points
        +int ghost_score_index
        +bool running
        +update(fps, move_pacman)
        +_check_collision(fps)
    }
    Game --> PacMan
    Game --> Map
    Game --> Ghost1
    Game --> Ghost2
    Game --> Ghost3
    Game --> Ghost4
    Game --> PowerPellet
    Game --> ScorePellet

    class PacManEnv {
        +Map maze
        +PacMan pacman
        +List~Ghost~ ghosts
        +List~PowerPellet~ power_pellets
        +List~ScorePellet~ score_pellets
        +bool done
        +List~int~ action_space
        +Tuple~int~ observation_space
        +reset()
        +step(action)
        +_get_state()
        +render()
    }
    PacManEnv --> Map
    PacManEnv --> PacMan
    PacManEnv --> Ghost
    PacManEnv --> PowerPellet
    PacManEnv --> ScorePellet

    class ControlStrategy {
        <<abstract>>
        +move(pacman, maze, power_pellets, score_pellets, ghosts, moving) bool
    }

    class PlayerControl {
        +int dx
        +int dy
        +handle_event(event)
        +move(pacman, maze, power_pellets, score_pellets, ghosts, moving) bool
    }
    PlayerControl --|> ControlStrategy

    class RuleBasedAIControl {
        +move(pacman, maze, power_pellets, score_pellets, ghosts, moving) bool
    }
    RuleBasedAIControl --|> ControlStrategy

    class DQNAIControl {
        +DQNAgent agent
        +torch.device device
        +move(pacman, maze, power_pellets, score_pellets, ghosts, moving) bool
    }
    DQNAIControl --|> ControlStrategy

    class ControlManager {
        +PlayerControl player_control
        +RuleBasedAIControl rule_based_ai
        +DQNAIControl dqn_ai
        +ControlStrategy current_strategy
        +bool moving
        +switch_mode()
        +handle_event(event)
        +move(pacman, maze, power_pellets, score_pellets, ghosts) bool
        +get_mode_name() str
    }
    ControlManager --> PlayerControl
    ControlManager --> RuleBasedAIControl
    ControlManager --> DQNAIControl

    class DQN {
        +Tuple~int~ input_dim
        +nn.Sequential conv
        +nn.Sequential fc
        +_get_conv_out(shape) int
        +forward(x) Tensor
    }

    class DQNAgent {
        +Tuple~int~ state_dim
        +int action_dim
        +torch.device device
        +deque memory
        +int batch_size
        +float gamma
        +float epsilon
        +float epsilon_min
        +float epsilon_decay
        +DQN model
        +DQN target_model
        +optim.Adam optimizer
        +int steps
        +int target_update_freq
        +update_target_model()
        +get_action(state) int
        +train() float
        +remember(state, action, reward, next_state, done)
        +save(path, memory_path)
        +load(path, memory_path)
    }
    DQNAgent --> DQN
```