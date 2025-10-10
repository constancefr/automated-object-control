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

class ACCEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        self.MAX_VALUE = 100
        self.CAR_LENGTH = 125/6.5 # 
        self.MIN_SEPARATION = 1.5 * self.CAR_LENGTH
        
        # Makes the continuous fragment of the system deterministic by fixing the
        # amount of time that the ODE evolves.
        self.TIME_STEP = 0.1

        # Maximal forward acceleration
        self.A = 3.1
        # Maximal braking acceleration
        self.B = 5.5
        
        bound = np.array([np.finfo(np.float32).max,np.finfo(np.float32).max])

        # Action Space: Choose Acceleration self.A, 0 or self.B
        # TODO: Change Action Space of the Model
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

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def is_crash(self, ego_pos, front_pos):
        ego_front_bumper = ego_pos + self.CAR_LENGTH/2
        front_rear_bumper = front_pos - self.CAR_LENGTH/2
        
        crash = ego_front_bumper >= front_rear_bumper
        return crash
    
    def update_front_action(self, front_pos, front_vel):
        # Calculate boundaries
        dist_l = front_pos
        dist_r = self.MAX_VALUE - front_pos

        if dist_l <= self.MIN_SEPARATION: # need enough space for at least the ego car to fit
            action_space = [0] # accelerate
        elif dist_r <= self.CAR_LENGTH / 2: # can go up to contact with right wall
            action_space = [2] # break
        else:
            action_space = [0,1,2] # full action space
        
        # action = random.choice(action_space)
        action = self.np_random.choice(action_space)
        return action
    
    def update_front_state(self, front_pos, front_vel, front_action):
        # TODO: select acceleration / braking amount continuously, not just max or min
        if front_action==0:
            acc = self.A
        elif front_action==1:
            acc = 0
        elif front_action==2:
            acc = -self.B
        else:
            raise ValueError(f"Unknown action value {front_action}")
        
        # update velocity by integrating the new acceleration over time --
        # pos = acc*t^2/2 + vel_0*t + pos_0
        # vel = vel = acc*t + vel_0
        t = self.TIME_STEP

        front_pos_0 = front_pos
        front_vel_0 = front_vel
        front_pos = acc*t**2/2 + front_vel_0*t + front_pos_0
        front_vel = acc*t + front_vel_0

        # enforce boundary constraints
        front_pos = np.clip(front_pos, self.CAR_LENGTH/2, self.MAX_VALUE - self.CAR_LENGTH/2)
        
        # velocity constraints
        max_vel = np.sqrt(max(0.0, (self.MAX_VALUE - front_pos) * 2 * self.A))
        front_vel = np.clip(front_vel, 0, max_vel)

        self.front_state = (np.float32(front_pos), np.float32(front_vel))

    def step(self, action):
        # What happens when the model takes a step
        assert self.action_space.contains(action), "%s (of type %s) invalid" % (str(action), type(action))

        # FRONT CAR ---
        front_state = self.front_state
        front_pos,front_vel = front_state[0],front_state[1]

        front_action = self.update_front_action(front_pos, front_vel)
        self.update_front_state(front_pos, front_vel, front_action)
        # -------------
        
        # Get ego state
        state = self.state
        ego_pos, ego_vel = state[0],state[1]

        # Determine acceleration:
        # TODO: Update choice of acceleration based on changed action space
        acc = 0
        if action==0:
            acc = self.A 
        elif action==1:
            acc = 0
        elif action==2:
            acc = -self.B
        else:
            raise ValueError(f"Unknown action value {action}")

        # update velocity by integrating the new acceleration over time --
        # pos = acc*t^2/2 + vel_0*t + pos_0
        # vel = vel = acc*t + vel_0
        t = self.TIME_STEP

        pos_0 = ego_pos
        vel_0 = ego_vel
        ego_pos = acc*t**2/2 + vel_0*t + pos_0
        ego_vel = acc*t + vel_0

        self.state = (ego_pos, ego_vel)

        # Assigning reward
        front_pos_new, _ = self.front_state
    
        crash = self.is_crash(ego_pos, front_pos_new)
        truncated = (ego_pos > self.MAX_VALUE - self.CAR_LENGTH/2 or 
                ego_pos < self.CAR_LENGTH/2 or
                front_pos_new > self.MAX_VALUE - self.CAR_LENGTH/2 or 
                front_pos_new < self.CAR_LENGTH/2)
        done = crash or truncated

        distance = front_pos_new - ego_pos
        optimal_distance = 2 * self.CAR_LENGTH # somewhat arbitrary
        distance_error = abs(distance - optimal_distance)

        distance_penalty = -0.01 * distance_error

        if crash:
            reward = -20.0
        elif truncated:
            reward = -10.0
        else:
            reward = 0.1 + distance_penalty

        if self.invert_loss:
            reward *= -1.0

        return np.array(self.state,dtype=np.float32), reward, done, truncated, {'crash': crash}
    
    def reset(self, seed=None, options=None):
        # If you want to change the state initialization, this is the place to go...
        if seed is not None:
            self._seed(seed=seed)
            
        if options is not None and "new_state" in options:
            state = options["new_state"]
            assert (isinstance(state, list) or isinstance(state, tuple)) and len(state) == 2, "New state must be tuple/list with 2 components"
            self.state = (np.float32(state[0]), np.float32(state[1]))
            return np.array(self.state), {'crash': self.is_crash(state)}
        
        # EGO CAR ---
        min_ego_pos = self.CAR_LENGTH / 2 # stay within frame
        max_ego_pos = self.MAX_VALUE - self.CAR_LENGTH * 1.5 # some extra space for the front car to fit
        pos = self.np_random.uniform(low=min_ego_pos, high=max_ego_pos, size=(1,))[0]
        
        # We must not approach too fast (in which case braking would not stop us anymore)
        min_velocity = -np.sqrt(pos*2*self.B)
        # Hypothetical constraint on the other side:
        # (MAX_VALUE-pos) <= vel^2 / (2*B)
        # We must not fall behind too fast (in which case accelerating would not help us anymore)
        max_velocity = np.sqrt((self.MAX_VALUE-pos)*2*self.A)
        vel = self.np_random.uniform(low=min_velocity,high=max_velocity, size=(1,))[0]
        self.state = (np.float32(pos), np.float32(vel))

        # FRONT CAR ---
        min_front_pos = pos + self.CAR_LENGTH
        max_front_pos = self.MAX_VALUE - self.CAR_LENGTH / 2 # stay within frame
        front_pos = self.np_random.uniform(low=min_front_pos, high=max_front_pos, size=(1,))[0]
        
        front_min_velocity = 0.0
        front_max_velocity = min(np.sqrt(max(0.0, (self.MAX_VALUE - front_pos) * 2 * self.A)), 20.0)
        front_vel = self.np_random.uniform(low=front_min_velocity, high=front_max_velocity, size=(1,))[0]
        self.front_state = (np.float32(front_pos), np.float32(front_vel))

        # Debug
        # actual_distance = front_pos - pos
        # print(f"Initialized: ego at {pos:.1f}, front at {front_pos:.1f}")
        # print(f"Ego is {'behind' if actual_distance > 0 else 'ahead of'} front car by {abs(actual_distance):.1f} units")
        
        return np.array(self.state), {'crash': False}

    def render(self, mode='rgb_array', close=False):
        # This determines how our videos are rendered
        assert mode==self.render_mode
        if close:
            if self.viewer is not None:    
                pygame.display.quit()
                pygame.quit()
                self.isopen = False
                self.viewer = None

        screen_width = 1000
        screen_height = 400

        pole_speed = 10  # pixels per frame
        pole_spacing = 200  # distance between poles in pixels
        pole_width = 10
        pole_height = 60
        
        cloud_speed = 1  # slower for parallax
        cloud_spacing = 300

        hill_speed = 2  # Very slow for distant background
        hill_spacing = 600  # Distance between hills

        stripe_width = 40
        stripe_height = 5
        stripe_spacing = 80

        carty = 40 # BOTTOM OF CART
        cartwidth = 125.0
        # cartwidth = 250.0
        cartheight = 30.0
        # cartheight = 60.0
        x_scale = screen_width / self.MAX_VALUE

        cart_pix_width = self.CAR_LENGTH * x_scale
        cart_pix_height = cartheight

        relativeDistance = cartwidth * 2

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))
            #pygame.Surface((screen_width, screen_height))

            self.nn_cart = pygame.image.load("/home/s2908217/autonomous_car_control/versaille_env/nn-car.png").convert_alpha()
            self.front_cart = pygame.image.load("/home/s2908217/autonomous_car_control/versaille_env/front-car.png").convert_alpha()
            original_width, original_height = self.nn_cart.get_size()

            scale_factor = cart_pix_width / original_width
            # scale_factor = cartwidth / original_width
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
        ego_pos, ego__vel = self.state
        front_pos, front_vel = self.front_state

        front_x = front_pos * x_scale
        ego_x = ego_pos * x_scale
                   
        # Converting centre to upper-left for blit (so image is centered)
        half_w = cart_pix_width / 2.0
        # Add a follower cart
        l,r = -half_w, 0.0
        # l,r = -cartwidth, 0.0
        t,b = cartheight, 0.0
        l += ego_x
        b += carty*0.75
        #gfxdraw.filled_polygon(self.surf, coords, (0,0,0))
        self.surf.blit(self.nn_cart, (l,b)) # (l,b) is upper-left corner

        # Add leader cart
        l,r = -half_w, 0.0
        # l,r = -cartwidth, 0.0
        t,b = cartheight, 0.0
        l += front_x
        b += carty*0.75
        self.surf.blit(self.front_car, (l,b))

        stripe_color = (228, 228, 228)  # Yellow stripe

        for x in self.stripe_positions:
            pygame.draw.rect(
                self.surf,
                stripe_color,
                pygame.Rect(x, 0, stripe_width, stripe_height)
            )

        # Display track
        #gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.viewer)), axes=(1, 0, 2))

    def close(self):
        if self.viewer is not None:    
            pygame.display.quit()
            pygame.quit()
            self.viewer = None


gym.register(
      id='acc-discrete-v0',
      entry_point=ACCEnv,
      max_episode_steps=410,  # todo edit
      reward_threshold=400.0, # todo edit
  )