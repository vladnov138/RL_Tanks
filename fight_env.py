import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from enum import IntEnum
import os

class ACTION(IntEnum):
    """Actions available in the Tank battle grid environment."""
    LEFT = 0
    RIGHT = 1
    TOP = 2
    DOWN = 3
    SHOT = 4

class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class FightGridEnv(gym.Env):
    """
    A simple grid environment
    
    Grid: 8x8
    Actions: left, right, top, down, shot
    Reward: +1 for shotting enemy (episode ends)
            -1 for shotting someone else
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    max_steps = 200
    current_step = 0
    bullet_speed = 1
    
    def __init__(self, render_mode=None, seed=None):
        super().__init__()
        
        self.grid_size = 8
        self.render_mode = render_mode
        self._seed = seed
        
        # Define action and observation spaces
        # 6 actions: left, right, top, down, pick up, put
        self.action_space = spaces.Discrete(len(ACTION))

        self.observation_space = spaces.MultiDiscrete([
            self.grid_size,
            self.grid_size,
            4,
            self.grid_size,
            self.grid_size
        ])
        
        # Load textures for rendering
        self._load_textures()
        
        # Initialize state
        self.reset()

    def _load_textures(self):
        """Load texture images from the textures directory"""
        textures_dir = os.path.join(os.path.dirname(__file__), "textures")
        
        # Load cell texture
        cell_path = os.path.join(textures_dir, "cell.png")
        self.cell_texture = Image.open(cell_path).convert("RGBA")
        self.cell_size = self.cell_texture.size[0]  # Assume square cells
        
        # Load main tank texture
        main_tank_path = os.path.join(textures_dir, "main_tank.png")
        self.main_tank_texture = Image.open(main_tank_path).convert("RGBA")
        # Resize main tank to fit in cell (assuming 80% of cell size)
        tank_size = int(self.cell_size * 0.8)
        self.main_tank_texture = self.main_tank_texture.resize((tank_size, tank_size), Image.Resampling.LANCZOS)

        # Load enemy tank texture
        enemy_tank_path = os.path.join(textures_dir, "enemy_tank.png")
        self.enemy_tank_texture = Image.open(enemy_tank_path).convert("RGBA")
        # Resize tank to fit in cell (assuming 80% of cell size)
        self.enemy_tank_texture = self.enemy_tank_texture.resize((tank_size, tank_size), Image.Resampling.LANCZOS)
        
        # Load destroyed enemy tank texture
        destroyed_enemy_tank_path = os.path.join(textures_dir, "destroyed_enemy_tank.png")
        self.destroyed_enemy_tank_texture = Image.open(destroyed_enemy_tank_path).convert("RGBA")
        self.destroyed_enemy_tank_texture = self.destroyed_enemy_tank_texture.resize((tank_size, tank_size), Image.Resampling.LANCZOS)

        # Load destroyed main tank texture
        destroyed_main_tank_path = os.path.join(textures_dir, "destroyed_main_tank.png")
        self.destroyed_main_tank_texture = Image.open(destroyed_main_tank_path).convert("RGBA")
        self.destroyed_main_tank_texture = self.destroyed_main_tank_texture.resize((tank_size, tank_size), Image.Resampling.LANCZOS)

        # Load destroyed shot texture
        shot_path = os.path.join(textures_dir, "shot.jpg")
        self.shot_texture = Image.open(shot_path).convert("RGBA")
        shot_size = int(self.cell_size * 0.25)
        self.shot_texture = self.shot_texture.resize(
            (shot_size, shot_size),
            Image.Resampling.LANCZOS
        )


    def reset(self, seed=None, options=None):
        if seed is None and self._seed is not None:
            seed = self._seed
        super().reset(seed=seed)
        self.steps = 0
        
        # Randomly place tanks
        n_entities = 2
        positions = self.np_random.choice(
            self.grid_size * self.grid_size,
            size=n_entities,
            replace=False
        )
        
        main_tank_pos = positions[0]
        enemy_tank_pos = positions[1]
        
        self.main_tank_x = main_tank_pos // self.grid_size
        self.main_tank_y = main_tank_pos % self.grid_size
        
        self.enemy_tank_x = enemy_tank_pos // self.grid_size
        self.enemy_tank_y = enemy_tank_pos % self.grid_size

         # Random tank directions
        self.main_tank_direction = int(self.np_random.integers(0, 4))
        self.enemy_tank_direction = int(self.np_random.integers(0, 4))

        # Tank status
        self.main_tank_alive = True
        self.enemy_tank_alive = True

        # Projectile state
        self.shot_active = False
        self.shot_x = -1
        self.shot_y = -1
        self.shot_direction = 0

        self._initial_state_id = f"s{self._state_to_idx(self.main_tank_x, self.main_tank_y, self.main_tank_direction, self.enemy_tank_x, self.enemy_tank_y)}"
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def _state_to_idx(self, main_tank_x, main_tank_y, main_direction, enemy_tank_x, enemy_tank_y):
        """Convert state to unique index."""
        cells = self.grid_size ** 2

        main_tank_pos = main_tank_x * self.grid_size + main_tank_y
        enemy_tank_pos = enemy_tank_x * self.grid_size + enemy_tank_y

        return (main_tank_pos * 4 + main_direction) * cells + enemy_tank_pos
    
    def _idx_to_state(self, idx):
        n = self.grid_size
        cells = n ** 2

        enemy_pos = idx % cells
        idx //= cells

        main_direction = idx % 4
        main_pos = idx // 4

        main_x, main_y = divmod(main_pos, n)
        enemy_x, enemy_y = divmod(enemy_pos, n)

        return main_x, main_y, main_direction, enemy_x, enemy_y
    
    def _compute_move_pos(self, main_tank_x, main_tank_y, action_int):
        """Compute next position from move action. Returns (new_x, new_y)."""
        if action_int == ACTION.LEFT:
            return main_tank_x, max(0, main_tank_y - 1)
        if action_int == ACTION.RIGHT:
            return main_tank_x, min(self.grid_size - 1, main_tank_y + 1)
        if action_int == ACTION.TOP:
            return max(0, main_tank_x - 1), main_tank_y
        if action_int == ACTION.DOWN:
            return min(self.grid_size - 1, main_tank_x + 1), main_tank_y
        return main_tank_x, main_tank_y

    def get_all_states(self):
        n_states = (self.grid_size ** 2) ** 2 * 4

        return tuple(f"s{i}" for i in range(n_states)) + ("terminal",)
        
    def get_possible_actions(self, state):
        """Return tuple of possible action names for the state."""
        return ("a0", "a1", "a2", "a3", "a4")  # LEFT, RIGHT, TOP, DOWN, SHOT

    def get_next_states(self, state, action):
        """Return dict mapping next_state -> transition probability."""

        # Terminal state
        if state == "terminal":
            return {"terminal": 1.0}

        idx = int(state.removeprefix("s"))

        main_x, main_y, main_direction, enemy_x, enemy_y = self._idx_to_state(idx)
        action_int = int(action.removeprefix("a"))

        # Movement
        if action_int in (ACTION.LEFT, ACTION.RIGHT, ACTION.TOP, ACTION.DOWN):

            new_x, new_y = self._compute_move_pos(
                main_x, main_y, action_int
            )

            # Prevent collision with enemy
            if (new_x, new_y) == (enemy_x, enemy_y):
                new_x, new_y = main_x, main_y

                # Update tank direction
                direction_map = {
                    ACTION.TOP: Direction.UP,
                    ACTION.RIGHT: Direction.RIGHT,
                    ACTION.DOWN: Direction.DOWN,
                    ACTION.LEFT: Direction.LEFT
                }

                new_direction = int(direction_map[action_int])

                next_idx = self._state_to_idx(
                    new_x,
                    new_y,
                    new_direction,
                    enemy_x,
                    enemy_y
                )

                return {f"s{next_idx}": 1.0}
            # Shooting
            elif action_int == ACTION.SHOT:

                hit = False

                if main_direction == Direction.UP:
                    hit = main_y == enemy_y and enemy_x < main_x
                elif main_direction == Direction.RIGHT:
                    hit = main_x == enemy_x and enemy_y > main_y
                elif main_direction == Direction.DOWN:
                    hit = main_y == enemy_y and enemy_x > main_x
                elif main_direction == Direction.LEFT:
                    hit = main_x == enemy_x and enemy_y < main_y
                if hit:
                    return {"terminal": 1.0}

                return {state: 1.0}
        raise ValueError(f"Unknown action: {action}")

    def get_reward(self, state, action, next_state):
        if state == "terminal":
            return 0.0

        if next_state == "terminal":
            return 10.0

        action_int = int(action.removeprefix("a"))

        if action_int == ACTION.SHOT:
            return -0.1

        return -0.01
    
    def get_transition_prob(self, state, action, next_state):
        """Return P(next_state | state, action)."""
        next_states = self.get_next_states(state, action)
        return next_states.get(next_state, 0.0)
    
    def is_terminal(self, state):
        """Return True if enemy tank is destroyed."""
        return state == "terminal"
    
    def get_initial_state(self):
        """Return the state id that the env is in right after reset()."""
        return self._initial_state_id
    
    def load_from_state_id(self, state_id):
        """Load environment from the given state ID."""

        if self.is_terminal(state_id):
            raise ValueError("Cannot load terminal state")

        idx = int(state_id.removeprefix("s"))

        (
            main_tank_x,
            main_tank_y,
            main_direction,
            enemy_tank_x,
            enemy_tank_y
        ) = self._idx_to_state(idx)

        # Validate state
        if not (0 <= idx < (self.grid_size ** 2) ** 2 * 4):
            raise ValueError(f"Invalid state ID: {state_id}")

        if (main_tank_x, main_tank_y) == (enemy_tank_x, enemy_tank_y):
            raise ValueError("Tanks cannot occupy the same cell")

        # Restore positions
        self.main_tank_x = main_tank_x
        self.main_tank_y = main_tank_y
        self.main_tank_direction = main_direction

        self.enemy_tank_x = enemy_tank_x
        self.enemy_tank_y = enemy_tank_y

        # Restore game status
        self.main_tank_alive = True
        self.enemy_tank_alive = True

        # Reset projectile
        self.shot_active = False
        self.shot_x = -1
        self.shot_y = -1
        self.shot_direction = 0

        self.steps = 0
    
    def seed(self, seed=None):
        """Set the seed for the environment's random number generator.
        For compatibility with older Gym API. Returns the seed in a list."""
        self._seed = seed
        if hasattr(self, "np_random") and self.np_random is not None:
            self.np_random = np.random.default_rng(seed)
        return [seed]

    def _get_observation(self):
        return np.array([
            self.main_tank_x,
            self.main_tank_y,
            self.main_tank_direction,
            self.enemy_tank_x,
            self.enemy_tank_y
        ], dtype=np.int64)
    
    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0

        if action in (ACTION.LEFT, ACTION.RIGHT, ACTION.TOP, ACTION.DOWN):
            new_x, new_y = self._compute_move_pos(
                self.main_tank_x,
                self.main_tank_y,
                action
            )

            if (new_x, new_y) != (self.enemy_tank_x, self.enemy_tank_y):
                self.main_tank_x = new_x
                self.main_tank_y = new_y

            if action == ACTION.LEFT:
                self.main_tank_direction = Direction.LEFT
            elif action == ACTION.RIGHT:
                self.main_tank_direction = Direction.RIGHT
            elif action == ACTION.TOP:
                self.main_tank_direction = Direction.UP
            elif action == ACTION.DOWN:
                self.main_tank_direction = Direction.DOWN

            reward = -0.01

        elif action == ACTION.SHOT:

            hit = False

            if self.main_tank_direction == Direction.UP:
                hit = (
                    self.main_tank_y == self.enemy_tank_y
                    and self.enemy_tank_x < self.main_tank_x
                )
            elif self.main_tank_direction == Direction.RIGHT:
                hit = (
                    self.main_tank_x == self.enemy_tank_x
                    and self.enemy_tank_y > self.main_tank_y
                )
            elif self.main_tank_direction == Direction.DOWN:
                hit = (
                    self.main_tank_y == self.enemy_tank_y
                    and self.enemy_tank_x > self.main_tank_x
                )
            elif self.main_tank_direction == Direction.LEFT:
                hit = (
                    self.main_tank_x == self.enemy_tank_x
                    and self.enemy_tank_y < self.main_tank_y
                )

            if hit:
                self.enemy_tank_alive = False
                reward = 10.0
                terminated = True
            else:
                reward = -0.1

        else:
            raise ValueError(f"Invalid action: {action}")

        # Episode limit
        self.steps += 1

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        observation = self._get_observation()

        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, show_coords=False):
        """Render the tank environment."""

        if self.render_mode == "human":
            # Create empty grid
            grid = [
                ['.' for _ in range(self.grid_size)]
                for _ in range(self.grid_size)
            ]

            # Place enemy tank
            if self.enemy_tank_alive:
                grid[self.enemy_tank_x][self.enemy_tank_y] = 'E'
            else:
                grid[self.enemy_tank_x][self.enemy_tank_y] = 'X'

            # Place main tank with direction
            if self.main_tank_alive:
                direction_symbols = {
                    Direction.UP: 'U',
                    Direction.RIGHT: 'R',
                    Direction.DOWN: 'D',
                    Direction.LEFT: 'L'
                }

                grid[self.main_tank_x][self.main_tank_y] = (
                    direction_symbols[self.main_tank_direction]
                )
            else:
                grid[self.main_tank_x][self.main_tank_y] = 'X'

            # Print grid
            print("\n" + "=" * 35)

            for row in grid:
                print(" ".join(f"{cell:>3}" for cell in row))

            print(f"Enemy alive: {self.enemy_tank_alive}")
            print(f"Steps: {self.steps}")

            print("=" * 35 + "\n")

        # Return image array
        if self.render_mode in ["human", "rgb_array"] or self.render_mode is None:
            return self._render_image(show_coords=show_coords)

        return None

    def _render_image(self, show_coords=False):
        """Render the tank environment as an image using textures."""

        # Create a blank image for the grid
        img_width = self.grid_size * self.cell_size
        img_height = self.grid_size * self.cell_size

        img = Image.new(
            "RGBA",
            (img_width, img_height),
            (255, 255, 255, 255)
        )

        # Draw each cell
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_x = y * self.cell_size
                cell_y = x * self.cell_size

                # Draw regular cell
                img.paste(
                    self.cell_texture,
                    (cell_x, cell_y),
                    self.cell_texture
                )

                # Place enemy tank
                if x == self.enemy_tank_x and y == self.enemy_tank_y:

                    if self.enemy_tank_alive:
                        enemy_img = self.enemy_tank_texture
                    else:
                        enemy_img = self.destroyed_enemy_tank_texture

                    enemy_offset_x = (self.cell_size - enemy_img.size[0]) // 2
                    enemy_offset_y = (self.cell_size - enemy_img.size[1]) // 2

                    img.paste(
                        enemy_img,
                        (cell_x + enemy_offset_x, cell_y + enemy_offset_y),
                        enemy_img
                    )

                # Place main tank
                if x == self.main_tank_x and y == self.main_tank_y:

                    if self.main_tank_alive:
                        main_img = self.main_tank_texture
                    else:
                        main_img = self.destroyed_main_tank_texture

                    # Rotate tank according to direction
                    rotation_angles = {
                        Direction.UP: 0,
                        Direction.RIGHT: -90,
                        Direction.DOWN: 180,
                        Direction.LEFT: 90
                    }

                    angle = rotation_angles[self.main_tank_direction]

                    main_img = main_img.rotate(
                        angle,
                        resample=Image.Resampling.NEAREST,
                        expand=True
                    )

                    main_offset_x = (self.cell_size - main_img.size[0]) // 2
                    main_offset_y = (self.cell_size - main_img.size[1]) // 2

                    img.paste(
                        main_img,
                        (cell_x + main_offset_x, cell_y + main_offset_y),
                        main_img
                    )

        # Draw cell coordinates
        if show_coords:
            font_size = max(12, self.cell_size // 6)
            font = None

            for path in [
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
            ]:
                try:
                    font = ImageFont.truetype(path, size=font_size)
                    break
                except (OSError, IOError):
                    continue

            if font is None:
                font = ImageFont.load_default()

            draw = ImageDraw.Draw(img)

            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    cell_x = y * self.cell_size
                    cell_y = x * self.cell_size

                    center_x = cell_x + self.cell_size // 2
                    center_y = cell_y + self.cell_size // 2

                    coord_text = f"({x},{y})"

                    draw.text(
                        (center_x, center_y),
                        coord_text,
                        fill=(0, 0, 0, 255),
                        font=font,
                        anchor="mm"
                    )

        # Convert RGBA to RGB
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])

        arr = np.array(rgb_img)

        # Display image in human mode
        if self.render_mode == "human":
            img.show()

        return arr