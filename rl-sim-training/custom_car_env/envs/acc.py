"""
Linear Adaptive Cruise Control in Relative Coordinates.
Adapation from S. Teuber's OpenAI Gym implementation at https://github.com/samysweb/VerSAILLE/blob/kikit/technical/docker/contents/libs/acc.py.
The leader car accelerates, breaks or idles stochastically.
"""

'''
TODO:
- check if initialisation is valid (proper distance between cars and to boundaries)
'''

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pygame
import os


class ACCEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.max_steps = 1000
        self.current_step = 0

        self.SCREEN_WIDTH = 2000
        self.SCREEN_HEIGHT = 400
        self.WORLD_VIEW_WIDTH = 100

        self.MAX_VALUE = 100.0  # ?
        self.CAR_LENGTH = 250.0
        self.CAR_HEIGHT = 60.0
        self.REL_CAR_LENGTH = (self.CAR_LENGTH * self.WORLD_VIEW_WIDTH) / self.SCREEN_WIDTH
        self.MIN_SEPARATION = 1.5 * self.REL_CAR_LENGTH

        # Makes the continuous fragment of the system deterministic by fixing the
        # amount of time that the ODE evolves.
        self.TIME_STEP = 0.1

        # Maximal forward acceleration
        self.Amax = 3
        # Maximal braking acceleration
        self.Bmin = 1
        self.Bmax = 5
        # Maximal velocity
        self.Vmax = 20.0

        bound = np.array(
            [np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])

        # Action Space: Choose Acceleration self.A, 0 or self.B
        self.action_space = spaces.Discrete(3)

        # obs = [rel_front_dist, rel_front_vel, rel_back_dist, rel_back_vel]
        self.observation_space = spaces.Box(-bound, bound, shape=(4,))

        self._seed()
        self.state = None

        # Rendering
        self.viewer = None

        self.render_mode = "rgb_array"
        self.metadata = {
            'render_modes': ['rgb_array'],
            'video.frames_per_second': 50
        }
        self.invert_loss = False

        # OTHER CARS ---
        self.last_front_action = 1  # idle default
        self.last_back_action = 1

        # to access action in render()
        self.ego_action = 1  # default idle
        self.front_action = 1
        self.back_action = 1

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def is_crash(self):
        rel_front_dist = self.state[6]
        rel_back_dist = self.state[8]
        # ego_pos, _, front_pos, _, back_pos, _, _, _, _, _ = self.state
        crash_front = abs(rel_front_dist) <= self.REL_CAR_LENGTH
        crash_back = abs(rel_back_dist) <= self.REL_CAR_LENGTH
        return crash_front or crash_back

    def update_other_action(self, prefix):
        '''
        Stochastic behaviour model for the front and back cars.
        Mostly cruising at high speed with occasional braking.
        '''
        behaviour_attr = f"{prefix}_behaviour"
        timer_attr = f"{prefix}_timer"
        emerg_attr = f"{prefix}_emergency_brake_active"
        action_attr = f"{prefix}_action"
        last_action_attr = f"last_{prefix}_action"

        # Initialise
        if not hasattr(self, behaviour_attr):
            setattr(self, behaviour_attr, "cruise")
            setattr(self, timer_attr, 0)
            setattr(self, emerg_attr, False)

        behaviour = getattr(self, behaviour_attr)
        timer = getattr(self, timer_attr)

        timer += 1
        setattr(self, timer_attr, timer)

        # Stochastic behaviour
        if timer > self.np_random.integers(20, 40):
            behaviour = self.np_random.choice(
                ["cruise", "accelerate", "brake", "emergency_brake"],
                p=[0.50, 0.40, 0.10, 0.00]
            )
            setattr(self, behaviour_attr, behaviour)
            setattr(self, timer_attr, 0)

            if behaviour == "emergency_brake":
                setattr(self, emerg_attr, True)

        if behaviour == "accelerate":
            action = 0
        elif behaviour == "brake":
            action = 2
        elif behaviour == "emergency_brake":
            action = 2
        else:  # cruise/idle
            action = 1

        # # FOR DEBUGGING --------
        # action = 0
        # # ----------------------

        setattr(self, action_attr, action)
        setattr(self, last_action_attr, action)

        return action

    def update_front_state(self, ego_pos, ego_vel, front_pos, front_vel, front_action):
        '''
        Update front car kinematics with special handling for emergency braking.
        '''
        if front_action == 0:
            acc = self.Amax * self.np_random.uniform(0.1, 1.0)  # variable acceleration
        elif front_action == 1:
            acc = 0
        elif front_action == 2:
            if getattr(self, "front_behaviour", None) == "emergency_brake":
                acc = -self.Bmax
            else:
                # acc = -self.B
                acc = -self.np_random.uniform(self.Bmin, self.Bmax)
        else:
            raise ValueError(f"Unknown action value {front_action}")

        # update velocity by integrating the new acceleration over time --
        # pos = acc*t^2/2 + vel_0*t + pos_0
        # vel = vel = acc*t + vel_0
        t = self.TIME_STEP

        front_vel_new = acc * t + front_vel
        front_vel_new = np.clip(front_vel_new, 0, self.Vmax)
        front_pos_new = acc * t ** 2 / 2 + front_vel_new * t + front_pos  # NOTE: use front_vel_new or front_vel here???

        rel_front_dist_new = ego_pos - front_pos_new
        rel_front_vel_new = ego_vel - front_vel_new

        self.state = (
            self.state[0], self.state[1],
            np.float32(front_pos_new), np.float32(front_vel_new),
            self.state[4], self.state[5],
            np.float32(rel_front_dist_new), np.float32(rel_front_vel_new),
            self.state[8], self.state[9],
        )

    def update_back_state(self, ego_pos, ego_vel, back_pos, back_vel, back_action):
        """
        Back car kinematics, keeps conservative spacing w.r.t. front car so ego can fit.
        """
        if back_action == 0:  # accelerate
            acc = self.Amax * self.np_random.uniform(0.1, 1.0)
        elif back_action == 1:  # idle
            acc = 0.0
        elif back_action == 2:  # brake
            if getattr(self, "rear_behaviour", None) == "emergency_brake":
                acc = -self.Bmax
            else:
                acc = -self.np_random.uniform(self.Bmin, self.Bmax)
        else:
            raise ValueError(f"Unknown action value {back_action}")

        # allow enougn space for ego (conservative)
        front_pos = self.state[2]
        back_pos = self.state[4]
        front_back_gap = front_pos - back_pos
        if front_back_gap < 2 * self.CAR_LENGTH:
            acc = -self.Bmax

        t = self.TIME_STEP
        back_vel_new = np.clip(back_vel + acc * t, 0, self.Vmax)
        back_pos_new = back_pos + back_vel * t + 0.5 * acc * t * t

        rel_back_dist_new = back_pos_new - ego_pos
        rel_back_vel_new = back_vel_new - ego_vel

        self.state = (
            self.state[0], self.state[1],
            self.state[2], self.state[3],
            np.float32(back_pos_new), np.float32(back_vel_new),
            self.state[6], self.state[7],
            np.float32(rel_back_dist_new), np.float32(rel_back_vel_new)
        )

    def step(self, action):
        self.current_step += 1

        # # Rescale SAC output actions [-1, 1] to asymmetric physical bounds
        # raw_action = float(np.clip(action, -1.0, 1.0)[0])

        # if raw_action >= 0.0:
        #     # Forward acceleration up to Amax
        #     acc = raw_action * self.Amax
        # else:
        #     # Braking up to -Bmax
        #     acc = raw_action * self.Bmax

        # OTHER CARS ---
        ego_pos, ego_vel, front_pos, front_vel, back_pos, back_vel, _, _, _, _ = self.state
        front_action = self.update_other_action("front")
        back_action = self.update_other_action("back")
        self.update_front_state(ego_pos, ego_vel, front_pos, front_vel, front_action)
        self.update_back_state(ego_pos, ego_vel, back_pos, back_vel, back_action)

        ego_pos, ego_vel = self.state[0], self.state[1]
        front_pos_new, front_vel_new = self.state[2], self.state[3]
        back_pos_new, back_vel_new = self.state[4], self.state[5]
        rel_front_dist_new, rel_front_vel_new = self.state[6], self.state[7]
        rel_back_dist_new, rel_back_vel_new = self.state[8], self.state[9]
        # -------------

        # EGO CAR -----
        if isinstance(action, (list, tuple, np.ndarray)):  # accept scalar or array-like actions
            try:
                action = int(np.asarray(action).reshape(-1)[0])  # take first elem if vectorised
            except Exception:
                action = int(action[0])

        self.ego_action = action
        acc = 0
        if action == 0:
            acc = self.Amax  # only allow max acceleration??
        elif action == 1:
            acc = 0.0
        elif action == 2:
            acc = -self.Bmax  # only allow max brake????
        else:
            raise ValueError(f"Unknown action value {action}")

        # update velocity by integrating the new acceleration over time --
        # pos = acc*t^2/2 + vel_0*t + pos_0
        # vel = vel = acc*t + vel_0
        t = self.TIME_STEP
        ego_vel_new = np.clip(ego_vel + acc * t, 0.0, self.Vmax)
        ego_pos_new = ego_pos + ego_vel * t + 0.5 * acc * t * t

        self.state = (
            np.float32(ego_pos_new), np.float32(ego_vel_new),
            np.float32(front_pos_new), np.float32(front_vel_new),
            np.float32(back_pos_new), np.float32(back_vel_new),
            np.float32(rel_front_dist_new), np.float32(rel_front_vel_new),
            np.float32(rel_back_dist_new), np.float32(rel_back_vel_new)
        )
        # -------------
        # Assigning reward
        crash = self.is_crash()
        terminated = crash
        truncated = self.current_step >= self.max_steps

        # Distances to front/back (always >= 0)
        front_gap = max(0.0, -rel_front_dist_new)  # distance from ego to front car
        back_gap = max(0.0, -rel_back_dist_new)  # distance from back car to ego

        # Desired safety gaps
        desired_front_gap = self.MIN_SEPARATION
        desired_back_gap = self.MIN_SEPARATION

        # --- 1) Headway term (front gap) ---
        front_gap_error = (front_gap - desired_front_gap) / max(desired_front_gap, 1e-6)
        front_gap_error = np.clip(front_gap_error, -1.0, 1.0)
        r_headway = 1.0 - (front_gap_error ** 2)  # in [0, 1]

        # --- 2) Relative speed term (speed matching) ---
        rel_speed_norm = rel_front_vel_new / max(self.Vmax, 1e-6)
        rel_speed_norm = np.clip(rel_speed_norm, -1.0, 1.0)
        r_speed = 1.0 - (rel_speed_norm ** 2)  # in [0, 1]

        # --- 3) Back car safety term ---
        back_safety_ratio = back_gap / max(desired_back_gap, 1e-6)
        back_safety_ratio = np.clip(back_safety_ratio, 0.0, 1.0)
        r_back = back_safety_ratio  # in [0, 1]

        # --- Combine into a single score in [0, 1] ---
        w_headway = 0.5
        w_speed = 0.3
        w_back = 0.2

        base_score = (
                w_headway * r_headway +
                w_speed * r_speed +
                w_back * r_back
        )  # in [0, 1]

        # Map [0, 1] -> [-1, 1]
        dense_term = 0.02*(base_score-0.5)

        # Survival bonus: explicitly reward being alive longer
        survival_bonus = 0.01
        reward = dense_term + survival_bonus

        danger_gap = self.REL_CAR_LENGTH

        if front_gap < danger_gap:
            frac = 1.0 - front_gap / max(danger_gap, 1e-6)
            reward -= 0.5 * np.clip(frac, 0.0, 1.0)
        # Crash penalty: clearly worse than any non-crash trajectory,
        # but not huge (keeps DQN targets stable).
        if crash:
            reward = -3.0

        if self.invert_loss:
            reward *= -1.0

        info = {
            'crash': crash,
            'front_action': getattr(self, 'last_front_action', 0)
        }

        # Provide RELATIVE metrics as observation to make infinite-time horizon manageable!
        observation = np.array([
            rel_front_dist_new, rel_front_vel_new,
            rel_back_dist_new, rel_back_vel_new
        ], dtype=np.float32)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        '''
        TODO: change docstring to include back car safety!!

        Safety constraints at initialisation:
        - min distance of L between cars
        - min dist of L between cars if both were to brake (front at max brake, ego at current brake)
        - other constraints enforced (min/max acceleration and velocity)
        
        pos_e(B_min) + L < pos_o
        => x_e - (v_e^2 / 2*a_e) + L < x_o - (v_o^2 / 2*a_o)
        => x_e - (v_e^2 / 2*B_min) + L < x_o - (v_o^2 / 2*B_max) # assume worst case scenario (max braking for front car, min for ego)
        '''

        # If you want to change the state initialization, this is the place to go...
        if seed is not None:
            self._seed(seed=seed)

        self.current_step = 0

        if options is not None and "new_state" in options:
            state = options["new_state"]
            assert (isinstance(state, list) or isinstance(state, tuple)) and len(
                state) == 2, "New state must be tuple/list with 2 components"
            self.state = (np.float32(state[0]), np.float32(state[1]), np.float32(state[2]), np.float32(state[3]),
                          np.float32(state[4]), np.float32(state[5]))

            observation = np.array([
                self.state[6], self.state[7],  # rel_front_dist, rel_front_vel
                self.state[8], self.state[9]  # rel_back_dist, rel_back_vel
            ], dtype=np.float32)

            return observation, {'crash': self.is_crash(state)}
            # return np.array(self.state), {'crash': self.is_crash(state)}

        # 1: set ego_pos x_e
        ego_pos = 0.0

        # 2: randomly set ego_vel v_e (positive, up to max V)
        ego_vel = self.np_random.uniform(low=0, high=self.Vmax, size=(1,))[0]
        # self.ego_state = (np.float32(ego_pos), np.float32(ego_vel))

        # 3: randomly set front_vel v_f and back_vel v_b (positive, up to max V)
        front_vel = self.np_random.uniform(low=0, high=self.Vmax, size=(1,))[0]
        back_vel = self.np_random.uniform(low=0, high=self.Vmax, size=(1,))[0]

        # 4: compute required safe distances so that crash is never inevitable at init
        # set front_pos x_o s.t. x_o > x_e - (v_e^2 / 2*B_min) + (v_o^2 / 2*B_max) + L
        # front brakes max, ego brakes min
        safe_front = (front_vel ** 2) / (2 * self.Bmax) - (ego_vel ** 2) / (2 * self.Bmin) + self.REL_CAR_LENGTH
        safe_front = max(safe_front, self.REL_CAR_LENGTH + 1e-3)

        # ego brakes max, back brakes min
        safe_back = (ego_vel ** 2) / (2 * self.Bmax) - (back_vel ** 2) / (2 * self.Bmin) + self.REL_CAR_LENGTH
        safe_back = max(safe_back, self.REL_CAR_LENGTH + 1e-3)

        front_pos = ego_pos + safe_front + self.np_random.uniform(0, self.MAX_VALUE * 0.5)
        back_pos = ego_pos - safe_back - self.np_random.uniform(5, 15)

        # sanity check
        assert back_pos < ego_pos < front_pos

        rel_front_dist = ego_pos - front_pos
        rel_front_vel = ego_vel - front_vel
        rel_back_dist = back_pos - ego_pos
        rel_back_vel = back_vel - ego_vel

        self.state = (
            ego_pos, ego_vel,
            front_pos, front_vel,
            back_pos, back_vel,
            rel_front_dist, rel_front_vel,
            rel_back_dist, rel_back_vel
        )

        info = {
            'crash': False,
            'front_action': 1  # default idle
        }

        observation = np.array([
            rel_front_dist, rel_front_vel,
            rel_back_dist, rel_back_vel
        ], dtype=np.float32)

        return observation, info

    def render(self, mode='rgb_array', close=False):
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # required when running on remote server without GUI

        # This determines how our videos are rendered
        assert mode == self.render_mode
        if close:
            if self.viewer is not None:
                pygame.display.quit()
                pygame.quit()
                self.isopen = False
                self.viewer = None

        screen_width = self.SCREEN_WIDTH
        screen_height = self.SCREEN_HEIGHT

        ego_pos, ego_vel = self.state[0], self.state[1]
        front_pos = self.state[2]
        back_pos = self.state[4]

        camera_centre_world = ego_pos  # camera follows ego car
        # camera_centre_world = front_pos # camera follows front car
        world_view_width = self.WORLD_VIEW_WIDTH  # how much world space is visible

        # Convert world coords to screen coords
        def world_to_screen(world_x):
            relative_to_camera = world_x - camera_centre_world
            screen_centre = screen_width * 0.50  # place ego car in the centre
            return screen_centre + (relative_to_camera / world_view_width) * screen_width

        scroll_base = ego_vel * 2.0
        # scroll_base = front_vel * 2.0

        pole_speed = scroll_base  # pixels per frame
        pole_spacing = 200  # distance between poles in pixels
        pole_width = 10
        pole_height = 60

        cloud_speed = scroll_base * 0.1  # slower for parallax
        cloud_spacing = 300

        hill_speed = scroll_base * 0.2  # Very slow for distant background
        hill_spacing = 600  # Distance between hills

        stripe_width = 40
        stripe_height = 5
        stripe_spacing = 80

        carty = 40  # BOTTOM OF CART

        x_scale = screen_width / world_view_width

        cart_pix_width = self.REL_CAR_LENGTH * x_scale
        cart_pix_height = self.CAR_HEIGHT

        if self.viewer is None:
            # pygame.init()
            # self.viewer = pygame.display.set_mode((screen_width, screen_height))
            pygame.init()
            pygame.display.set_mode((1, 1))
            self.viewer = True  # dummy flag, no window

            self.nn_cart = pygame.image.load("nn-car.png").convert_alpha()
            self.front_cart = pygame.image.load("other-car.png").convert_alpha()
            self.back_cart = pygame.image.load("other-car.png").convert_alpha()
            original_width, original_height = self.nn_cart.get_size()

            scale_factor = cart_pix_width / original_width
            new_height = int(original_height * scale_factor)
            self.nn_cart = pygame.transform.flip(
                pygame.transform.smoothscale(self.nn_cart, (int(cart_pix_width), new_height)), False, True)
            self.front_car = pygame.transform.flip(
                pygame.transform.smoothscale(self.front_cart, (int(cart_pix_width), new_height)), False, True)
            self.back_car = pygame.transform.flip(
                pygame.transform.smoothscale(self.back_cart, (int(cart_pix_width), new_height)), False, True)

            self.pole_positions = [x for x in range(0, screen_width + pole_spacing, pole_spacing)]
            self.cloud_positions = [(x, 300 + 50 * (i % 2)) for i, x in
                                    enumerate(range(0, screen_width + cloud_spacing, cloud_spacing))]
            self.hill_positions = [x for x in range(0, screen_width + hill_spacing, hill_spacing)]
            self.stripe_positions = [x for x in range(0, screen_width + stripe_spacing, stripe_spacing)]

        self.pole_positions = [
            x - pole_speed if x - pole_speed > -pole_width else screen_width
            for x in self.pole_positions
        ]
        self.cloud_positions = [
            ((x - cloud_speed) if (x - cloud_speed) > -90 else screen_width, y)
            for (x, y) in self.cloud_positions
        ]
        self.hill_positions = [
            (x - hill_speed) if (x - hill_speed) > -400 else screen_width
            for x in self.hill_positions
        ]
        self.stripe_positions = [
            x - pole_speed if x - pole_speed > -stripe_width else screen_width
            for x in self.stripe_positions
        ]

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((135, 206, 235))  # Sky blue background

        # Draw clouds
        for (x, y) in self.cloud_positions:
            pygame.draw.ellipse(self.surf, (255, 255, 255), pygame.Rect(int(x), int(y), 80, 40))

        for x in self.hill_positions:
            pygame.draw.ellipse(self.surf, (0, 128, 0), pygame.Rect(int(x), 10, 400, 120))  # Dark green hills

        pygame.draw.rect(self.surf, (34, 139, 34), pygame.Rect(0, int(carty), int(screen_width), int(carty)))  # road?
        pygame.draw.rect(self.surf, (192, 192, 192), pygame.Rect(0, 0, int(screen_width), int(carty)))

        for x in self.pole_positions:
            pygame.draw.rect(self.surf, (0, 0, 0), pygame.Rect(int(x), int(carty), int(pole_width), int(pole_height)))

        # CARS!
        # Convert world positions to screen positions
        ego_x = world_to_screen(ego_pos)
        front_x = world_to_screen(front_pos)
        back_x = world_to_screen(back_pos)

        # Converting centre to upper-left for blit (so image is centered)
        half_w = cart_pix_width / 2.0

        # Add ego car
        l, r = -half_w, 0.0
        t, b = cart_pix_height, 0.0
        l += ego_x
        b += carty * 0.25
        self.surf.blit(self.nn_cart, (int(l), int(b)))  # (l,b) is upper-left corner

        # Add front cart
        l, r = -half_w, 0.0
        t, b = cart_pix_height, 0.0
        l += front_x
        b += carty * 0.25
        self.surf.blit(self.front_car, (int(l), int(b)))

        # Add back car
        l, r = -half_w, 0.0
        t, b = cart_pix_height, 0.0
        l += back_x
        b += carty * 0.25
        self.surf.blit(self.back_car, (int(l), int(b)))

        stripe_color = (228, 228, 228)  # Yellow stripe
        for x in self.stripe_positions:
            pygame.draw.rect(
                self.surf,
                stripe_color,
                pygame.Rect(int(x), 0, stripe_width, stripe_height)
            )

        # Add colour dial to indicate action taken
        green = (0, 255, 0)
        yellow = (255, 255, 0)
        red = (255, 0, 0)

        if self.ego_action == 0:  # accelerate
            ego_colour = green
        elif self.ego_action == 1:  # idle
            ego_colour = yellow
        elif self.ego_action == 2:  # break
            ego_colour = red

        if self.front_action == 0:  # accelerate
            front_colour = green
        elif self.front_action == 1:  # idle
            front_colour = yellow
        elif self.front_action == 2:  # break
            front_colour = red

        if self.back_action == 0:  # accelerate
            back_colour = green
        elif self.back_action == 1:  # idle
            back_colour = yellow
        elif self.back_action == 2:  # break
            back_colour = red

        pygame.draw.circle(
            self.surf,
            ego_colour,
            (screen_width - 100, screen_height - 50),
            20)
        pygame.draw.circle(
            self.surf,
            front_colour,
            (screen_width - 50, screen_height - 50),
            20)
        pygame.draw.circle(
            self.surf,
            back_colour,
            (screen_width - 150, screen_height - 50),
            20)

        rgb_surface = pygame.transform.flip(self.surf, False, True)
        frame = pygame.surfarray.array3d(rgb_surface)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def close(self):
        if self.viewer is not None:
            pygame.display.quit()
            pygame.quit()
            self.viewer = None


gym.register(
    id='acc-continuous-v0',
    entry_point=ACCEnv,
    max_episode_steps=410,  # todo edit
    reward_threshold=400.0,  # todo edit
)
