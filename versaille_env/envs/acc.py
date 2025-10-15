"""
Linear Adaptive Cruise Control in Relative Coordinates.
Adapation from S. Teuber's OpenAI Gym implementation at https://github.com/samysweb/VerSAILLE/blob/kikit/technical/docker/contents/libs/acc.py.
The leader car accelerates, breaks or idles randomly.
"""

'''
TODO:
- check if initialisation is valid (proper distance between cars and to boundaries)
'''

import logging
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import random
import pygame
from pygame import gfxdraw
import os

class ACCEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.max_steps = 1000
        self.current_step = 0

        self.MAX_VALUE = 100
        self.CAR_LENGTH = 62.5
        self.CAR_HEIGHT = 15.0
        self.REL_CAR_LENGTH = self.CAR_LENGTH/6.5
        self.MIN_SEPARATION = 1.5 * self.REL_CAR_LENGTH
        
        # Makes the continuous fragment of the system deterministic by fixing the
        # amount of time that the ODE evolves.
        self.TIME_STEP = 0.1

        # Maximal forward acceleration
        self.A = 3.1
        # Maximal braking acceleration
        self.B = 5.5
        
        bound = np.array([np.finfo(np.float32).max,np.finfo(np.float32).max])

        # Action Space: Choose Acceleration self.A, 0 or self.B
        self.action_space = spaces.Discrete(3)

        # State Space: (position, velocity)
        self.observation_space = spaces.Box(-bound, bound)

        self._seed()
        self.state = None

        # Rendering
        self.viewer = None

        self.render_mode="rgb_array"
        self.metadata = {
            'render_modes': ['rgb_array'],
            'video.frames_per_second' : 50
        }
        self.invert_loss = False

        # FRONT CAR ---
        self.front_observation_space = spaces.Box(-bound, bound)
        self.front_state = None
        self.last_front_action = 1 # idle default

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def is_crash(self, ego_pos, front_pos):
        ego_front_bumper = ego_pos + self.REL_CAR_LENGTH/2
        front_rear_bumper = front_pos - self.REL_CAR_LENGTH/2
        
        crash = ego_front_bumper >= front_rear_bumper
        return crash

    # def is_leaving_frame(self, ego_pos, front_pos, screen_width=2000):
    #     '''
    #     Adapted for camera fixed on front car - checks if ego too far behind/ahead.
    #     '''
    #     camera_center_world = ego_pos
    #     # camera_center_world = front_pos
    #     world_view_width = 100.0  # same as in render!!
        
    #     world_left = camera_center_world - world_view_width / 2
    #     world_right = camera_center_world + world_view_width / 2
        
    #     front_left = front_pos - self.REL_CAR_LENGTH / 2
    #     # ego_left = ego_pos - self.REL_CAR_LENGTH / 2
    #     front_right = front_pos + self.REL_CAR_LENGTH / 2
    #     # ego_right = ego_pos + self.REL_CAR_LENGTH / 2
        
    #     is_leaving = front_left < world_left or front_right > world_right
    #     # is_leaving = ego_left < world_left or ego_right > world_right
    #     return is_leaving
    
    def update_front_action(self, front_pos, front_vel):
        '''
        Computes next action for the front car.
        Includes stochastic behaviour (emergency braking, gradual acceleration, inertia in behaviour changes).
        '''
        if not hasattr(self, "front_behaviour"):
            self.front_behaviour = "cruise"
            self.front_timer = 0

        # change behaviour every 20-40 steps
        self.front_timer += 1
        dist_l = front_pos
        dist_r = self.MAX_VALUE - front_pos
        if self.front_timer > self.np_random.integers(20, 40):
            if dist_l <= self.MIN_SEPARATION: # need enough space for at least the ego car to fit
                action_space = [0] # accelerate
            elif dist_r <= self.REL_CAR_LENGTH: # can go up to contact with right wall
                action_space = [2] # break
            else:
                action_space = [0,1,2] # full action space
            self.front_behaviour = self.np_random.choice(
                ["cruise", "accelerate", "brake", "emergency_brake"],
                p=[0.5, 0.3, 0.1, 0.1]
            )
            self.front_timer = 0
        
        if self.front_behaviour == "accelerate":
            action = 0
        elif self.front_behaviour == "brake":
            action = 2
        elif self.front_behaviour == "emergency_brake":
            action = 2
        else: # cruise
            action = 1

        self.last_front_action = action
        return action
    
    def update_front_state(self, front_pos, front_vel, front_action):
        '''
        Update front car kinematics with special handling for emergency braking.
        '''
        if front_action==0:
            acc = self.A * self.np_random.uniform(0.1, 1.0) # variable acceleration
        elif front_action==1:
            acc = 0
        elif front_action==2:
            if getattr(self, "front_behaviour", None) == "emergency_brake":
                acc = -self.B # strongest deceleration
            else:
                # acc = -self.B
                acc = -self.B * self.np_random.uniform(0.1, 0.9)
        else:
            raise ValueError(f"Unknown action value {front_action}")
        
        # update velocity by integrating the new acceleration over time --
        # pos = acc*t^2/2 + vel_0*t + pos_0
        # vel = vel = acc*t + vel_0
        t = self.TIME_STEP
        front_vel_new = acc*t + front_vel
        front_pos_new = acc*t**2/2 + front_vel_new*t + front_pos # TODO: use front_vel_new or front_vel here???

        # enforce boundary constraints
        front_pos_new = np.clip(front_pos_new, self.REL_CAR_LENGTH/2, self.MAX_VALUE - self.REL_CAR_LENGTH/2)
        # velocity constraints
        max_vel = np.sqrt(max(0.0, (self.MAX_VALUE - front_pos) * 2 * self.A))
        front_vel_new = np.clip(front_vel_new, 0, max_vel)

        self.front_state = (np.float32(front_pos_new), np.float32(front_vel_new))

    def step(self, action):
        '''
        TODO: remove distance penalty now that we have out of frame neg reward?
        '''
        self.current_step += 1
        assert self.action_space.contains(action), "%s (of type %s) invalid" % (str(action), type(action))

        # FRONT CAR ---
        front_pos,front_vel = self.front_state
        front_action = self.update_front_action(front_pos, front_vel)
        self.update_front_state(front_pos, front_vel, front_action)
        front_pos_new, front_vel_new = self.front_state
        # -------------
        
        # EGO CAR -----
        ego_pos, ego_vel = self.state
        if isinstance(action, (list, tuple, np.ndarray)): # accept scalar or array-like actions
            try:
                action = int(np.asarray(action).reshape(-1)[0]) # take first elem if vectorised
            except Exception:
                action = int(action[0])

        acc = 0
        if action==0:
            acc = self.A 
        elif action==1:
            acc = 0.0
        elif action==2:
            acc = -self.B
        else:
            raise ValueError(f"Unknown action value {action}")

        # update velocity by integrating the new acceleration over time --
        # pos = acc*t^2/2 + vel_0*t + pos_0
        # vel = vel = acc*t + vel_0
        t = self.TIME_STEP
        ego_vel_new = acc*t + ego_vel
        ego_vel_new = np.clip(ego_vel_new, 0, None)
        ego_pos_new = acc*t**2/2 + ego_vel_new*t + ego_pos # TODO: use ego_vel_new or ego_vel here???
        self.state = (np.float32(ego_pos_new), np.float32(ego_vel_new))
        # -------------

        # Assigning reward    
        crash = self.is_crash(ego_pos, front_pos_new)
        terminated = crash
        truncated = self.current_step >= self.max_steps

        if crash:
            reward = -10.0
        else:
            reward = 0.1
            
            # distance reward
            current_distance = front_pos_new - ego_pos_new
            ideal_min_distance = 2.0 * self.REL_CAR_LENGTH # somewhat arbitrary
            ideal_max_distance = 4.0 * self.REL_CAR_LENGTH
            
            if current_distance < ideal_min_distance: # too close
                reward -= 0.5
            elif current_distance > ideal_max_distance: # too far
                reward -= 0.2
            else:
                reward += 0.2

        # distance = front_pos_new - ego_pos_new
        # optimal_distance = 2 * self.REL_CAR_LENGTH # somewhat arbitrary
        # distance_error = abs(distance - optimal_distance)
        # distance_penalty = -0.05 * distance_error

        # if crash:
        #     reward = -50.0
        # # elif leaving_frame:
        # #     reward = -10.0
        # else:
        #     reward = 0.1 + distance_penalty

        if self.invert_loss:
            reward *= -1.0

        info = {
            'crash': crash,
            'front_state': getattr(self, 'front_state'),
            'front_action': getattr(self, 'last_front_action', 0)
        }

        return np.array(self.state,dtype=np.float32), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        # If you want to change the state initialization, this is the place to go...
        if seed is not None:
            self._seed(seed=seed)

        self.current_step = 0
            
        if options is not None and "new_state" in options:
            state = options["new_state"]
            assert (isinstance(state, list) or isinstance(state, tuple)) and len(state) == 2, "New state must be tuple/list with 2 components"
            self.state = (np.float32(state[0]), np.float32(state[1]))
            return np.array(self.state), {'crash': self.is_crash(state)}
        
        # EGO CAR ---
        min_ego_pos = self.REL_CAR_LENGTH / 2 # stay within frame
        max_ego_pos = self.MAX_VALUE / 2 - self.REL_CAR_LENGTH * 1.5 # some extra space for the front car to fit
        pos = self.np_random.uniform(low=min_ego_pos, high=max_ego_pos, size=(1,))[0]
        
        # We must not approach too fast (in which case braking would not stop us anymore)
        # min_velocity = -np.sqrt(pos*2*self.B)
        min_velocity = 0
        # Hypothetical constraint on the other side:
        # (MAX_VALUE-pos) <= vel^2 / (2*B)
        # We must not fall behind too fast (in which case accelerating would not help us anymore)
        max_velocity = np.sqrt((self.MAX_VALUE-pos)*2*self.A)
        vel = self.np_random.uniform(low=min_velocity,high=max_velocity, size=(1,))[0]
        self.state = (np.float32(pos), np.float32(vel))

        # FRONT CAR ---
        min_front_pos = pos + self.REL_CAR_LENGTH # reset within boundary
        max_front_pos = self.MAX_VALUE / 2 - self.REL_CAR_LENGTH / 2
        front_pos = self.np_random.uniform(low=min_front_pos, high=max_front_pos, size=(1,))[0]
        
        front_min_velocity = 0.0
        front_max_velocity = min(np.sqrt(max(0.0, (self.MAX_VALUE - front_pos) * 2 * self.A)), 20.0)
        front_vel = self.np_random.uniform(low=front_min_velocity, high=front_max_velocity, size=(1,))[0]
        self.front_state = (np.float32(front_pos), np.float32(front_vel))

        info = {
            'crash': False,
            'front_state': self.front_state,
            'front_action': 1 # default idle
        }

        return np.array(self.state), info

    def render(self, mode='rgb_array', close=False):
        os.environ["SDL_VIDEODRIVER"] = "dummy" # required when running on remote server without GUI
        
        # This determines how our videos are rendered
        assert mode==self.render_mode
        if close:
            if self.viewer is not None:    
                pygame.display.quit()
                pygame.quit()
                self.isopen = False
                self.viewer = None

        screen_width = 2000
        screen_height = 400

        ego_pos, ego_vel = self.state
        front_pos, front_vel = self.front_state

        camera_centre_world = ego_pos # camera follows ego car
        # camera_centre_world = front_pos # camera follows front car
        world_view_width = 100.0 # how much world space is visible

        # Convert world coords to screen coords
        def world_to_screen(world_x):
            relative_to_camera = world_x - camera_centre_world
            screen_centre = screen_width * 0.25
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

        carty = 40 # BOTTOM OF CART
        cartwidth = self.CAR_LENGTH
        cartheight = self.CAR_HEIGHT
        
        x_scale = screen_width / world_view_width
        # x_scale = screen_width / self.MAX_VALUE

        cart_pix_width = self.REL_CAR_LENGTH * x_scale
        cart_pix_height = cartheight

        if self.viewer is None:
            # pygame.init()
            # self.viewer = pygame.display.set_mode((screen_width, screen_height))
            pygame.init()
            pygame.display.set_mode((1,1))
            self.viewer = True  # dummy flag, no window

            self.nn_cart = pygame.image.load("/home/s2908217/autonomous_car_control/versaille_env/nn-car.png").convert_alpha()
            self.front_cart = pygame.image.load("/home/s2908217/autonomous_car_control/versaille_env/front-car.png").convert_alpha()
            original_width, original_height = self.nn_cart.get_size()

            scale_factor = cart_pix_width / original_width
            new_height = int(original_height * scale_factor)
            self.nn_cart = pygame.transform.flip(pygame.transform.smoothscale(self.nn_cart, (int(cart_pix_width), new_height)),False,True)
            self.front_car = pygame.transform.flip(pygame.transform.smoothscale(self.front_cart, (int(cart_pix_width), new_height)),False,True)

            self.pole_positions = [x for x in range(0, screen_width + pole_spacing, pole_spacing)]
            self.cloud_positions = [(x, 300 + 50 * (i % 2)) for i, x in enumerate(range(0, screen_width + cloud_spacing, cloud_spacing))]
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
            pygame.draw.ellipse(self.surf, (255, 255, 255), pygame.Rect(x, y, 80, 40))

        for x in self.hill_positions:
            pygame.draw.ellipse(self.surf, (0, 128, 0), pygame.Rect(x, 10, 400, 120))  # Dark green hills

        pygame.draw.rect(self.surf, (34, 139, 34), pygame.Rect(0, carty, screen_width, carty)) # road?
        pygame.draw.rect(self.surf, (192, 192, 192), pygame.Rect(0, 0, screen_width, carty))

        for x in self.pole_positions:
            pygame.draw.rect(self.surf, (0, 0, 0), pygame.Rect(x, carty, pole_width, pole_height))

        # CARS!
        # Convert world positions to screen positions
        ego_x = world_to_screen(ego_pos)
        front_x = world_to_screen(front_pos)
        # front_x_centre = screen_width * 0.75
        # ego_x = front_x_centre + (ego_pos - front_pos) * x_scale
        # front_x = front_x_centre
                   
        # Converting centre to upper-left for blit (so image is centered)
        half_w = cart_pix_width / 2.0
        # Add a follower cart
        l,r = -half_w, 0.0
        t,b = cartheight, 0.0
        l += ego_x
        b += carty*0.25
        self.surf.blit(self.nn_cart, (l,b)) # (l,b) is upper-left corner

        # Add leader cart
        l,r = -half_w, 0.0
        t,b = cartheight, 0.0
        l += front_x
        b += carty*0.25
        self.surf.blit(self.front_car, (l,b))

        stripe_color = (228, 228, 228)  # Yellow stripe
        for x in self.stripe_positions:
            pygame.draw.rect(
                self.surf,
                stripe_color,
                pygame.Rect(x, 0, stripe_width, stripe_height)
            )
        
        # self.surf = pygame.transform.flip(self.surf, False, True)
        # self.viewer.blit(self.surf, (0, 0))

        # return np.transpose(np.array(pygame.surfarray.pixels3d(self.viewer)), axes=(1, 0, 2))
        rgb_surface = pygame.transform.flip(self.surf, False, True)
        frame = pygame.surfarray.array3d(rgb_surface)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def close(self):
        if self.viewer is not None:    
            pygame.display.quit()
            pygame.quit()
            self.viewer = None

'''
    def render(self, mode='rgb_array', close=False):
        # This determines how our videos are rendered
        assert mode==self.render_mode
        if close:
            if self.viewer is not None:    
                pygame.display.quit()
                pygame.quit()
                self.isopen = False
                self.viewer = None
            return

        screen_width = 2000
        screen_height = 400

        ego_pos, ego_vel = self.state
        front_pos, front_vel = self.front_state

        camera_centre_world = ego_pos
        world_view_width = 100.0

        # Convert world coords to screen coords
        def world_to_screen(world_x):
            relative_to_camera = world_x - camera_centre_world
            screen_centre = screen_width * 0.25
            return screen_centre + (relative_to_camera / world_view_width) * screen_width

        # Initialize pygame if needed
        if self.viewer is None:
            # Use headless mode for servers
            if self.render_mode == "rgb_array":
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            
            pygame.init()
            # Create a minimal display surface
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
            self.surf = pygame.Surface((screen_width, screen_height))
            
            # Load and scale car images (with error handling)
            try:
                self.nn_cart = pygame.image.load("/home/s2908217/autonomous_car_control/versaille_env/nn-car.png").convert_alpha()
                self.front_cart = pygame.image.load("/home/s2908217/autonomous_car_control/versaille_env/front-car.png").convert_alpha()
                
                x_scale = screen_width / world_view_width
                cart_pix_width = self.REL_CAR_LENGTH * x_scale
                
                original_width, original_height = self.nn_cart.get_size()
                scale_factor = cart_pix_width / original_width
                new_height = int(original_height * scale_factor)
                
                self.nn_cart = pygame.transform.flip(
                    pygame.transform.smoothscale(self.nn_cart, (int(cart_pix_width), new_height)), False, True)
                self.front_car = pygame.transform.flip(
                    pygame.transform.smoothscale(self.front_cart, (int(cart_pix_width), new_height)), False, True)
            except Exception as e:
                print(f"Warning: Could not load car images: {e}")
                # Create placeholder rectangles if images fail to load
                self.nn_cart = pygame.Surface((50, 20))
                self.nn_cart.fill((255, 0, 0))  # Red for ego car
                self.front_car = pygame.Surface((50, 20))
                self.front_car.fill((0, 0, 255))  # Blue for front car

            # Initialize scrolling elements
            self.pole_positions = [x for x in range(0, screen_width + 200, 200)]
            self.cloud_positions = [(x, 300 + 50 * (i % 2)) for i, x in enumerate(range(0, screen_width + 300, 300))]
            self.hill_positions = [x for x in range(0, screen_width + 600, 600)]
            self.stripe_positions = [x for x in range(0, screen_width + 80, 80)]

        # Update scrolling elements
        scroll_base = ego_vel * 2.0
        pole_speed = scroll_base
        cloud_speed = scroll_base * 0.1
        hill_speed = scroll_base * 0.2

        self.pole_positions = [
            x - pole_speed if x - pole_speed > -10 else screen_width
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
            x - pole_speed if x - pole_speed > -40 else screen_width
            for x in self.stripe_positions
        ]

        # Clear surface
        self.surf.fill((135, 206, 235))  # Sky blue

        # Draw background elements
        for (x, y) in self.cloud_positions:
            pygame.draw.ellipse(self.surf, (255, 255, 255), pygame.Rect(x, y, 80, 40))

        for x in self.hill_positions:
            pygame.draw.ellipse(self.surf, (0, 128, 0), pygame.Rect(x, 10, 400, 120))

        # Draw road
        carty = 40
        pygame.draw.rect(self.surf, (34, 139, 34), pygame.Rect(0, carty, screen_width, carty))
        pygame.draw.rect(self.surf, (192, 192, 192), pygame.Rect(0, 0, screen_width, carty))

        # Draw poles
        for x in self.pole_positions:
            pygame.draw.rect(self.surf, (0, 0, 0), pygame.Rect(x, carty, 10, 60))

        # Draw cars
        ego_x = world_to_screen(ego_pos)
        front_x = world_to_screen(front_pos)
        
        half_w = self.nn_cart.get_width() / 2.0
        
        # Draw ego car
        ego_rect = self.nn_cart.get_rect(center=(ego_x, carty * 0.25 + self.nn_cart.get_height() / 2))
        self.surf.blit(self.nn_cart, ego_rect)
        
        # Draw front car
        front_rect = self.front_car.get_rect(center=(front_x, carty * 0.25 + self.front_car.get_height() / 2))
        self.surf.blit(self.front_car, front_rect)

        # Draw road stripes
        stripe_color = (228, 228, 228)
        for x in self.stripe_positions:
            pygame.draw.rect(self.surf, stripe_color, pygame.Rect(x, 0, 40, 5))

        # Convert to numpy array
        rgb_surface = pygame.transform.flip(self.surf, False, True)
        frame = pygame.surfarray.array3d(rgb_surface)
        frame = np.transpose(frame, (1, 0, 2))
        
        return frame

    def close(self):
        if self.viewer is not None:    
            pygame.display.quit()
            pygame.quit()
            self.viewer = None

'''


gym.register(
      id='acc-discrete-v0',
      entry_point=ACCEnv,
      max_episode_steps=410,  # todo edit
      reward_threshold=400.0, # todo edit
  )