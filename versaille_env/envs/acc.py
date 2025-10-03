"""
Linear Adaptive Cruise Control in Relative Coordinates.
The visualization fixes the position of the leader car.
Adapation from N. Fulton and A. Platzer, "Safe Reinforcement Learning via Formal Methods: Toward Safe Control through Proof and Learning", AAAI 2018.
OpenAI Gym implementation adapted from the classic control cart pole environment.
"""

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

    def __init__(self):
        self.MAX_VALUE = 100
        
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

    def is_crash(self, some_state):
      return some_state[0] <= 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # What happens when the model takes a step
        assert self.action_space.contains(action), "%s (of type %s) invalid" % (str(action), type(action))
        
        # Get State
        state = self.state
        pos, vel = state[0],state[1]

        # Determine acceleration:
        # TODO: Update choice of acceleration based on changed action space
        acc = 0
        if action==0:
            # Relative Position -> Acceleration decreases relative distance
            acc = -self.A 
        elif action==1:
            acc = 0
        elif action==2:
            # Relative Position -> Braking increases relative distance
            acc = self.B
        else:
            raise ValueError(f"Unknown action value {action}")

        # update velocity by integrating the new acceleration over time --
        # pos = acc*t^2/2 + vel_0*t + pos_0
        # vel = vel = acc*t + vel_0
        t = self.TIME_STEP

        pos_0 = pos
        vel_0 = vel
        pos = acc*t**2/2 + vel_0*t + pos_0
        vel = acc*t + vel_0

        self.state = (pos, vel)

        crash = self.is_crash(self.state)
        truncated = self.state[0] > self.MAX_VALUE
        done = crash or truncated
        done = bool(done)

        if not done:
            # Well done, you stayed alive!
            reward = 0.1
        elif done and self.state[0] <= 1:
            # TOO CLOSE
            reward = -200.0
        elif done and self.state[0] > self.MAX_VALUE - 0.5:
            # Fell too far behind
            reward = -50.0
        else:
            assert False, "Not sure why this should happen, and when it was previously there was a bug in the if/elif guards..."
            reward = 0.0

        if self.invert_loss:
            reward *= -1.0

        return np.array(self.state,dtype=np.float32), reward, done, truncated, {'crash': self.state[0] <= 0}

    def reset(self, seed=None,options=None):
        # If you want to change the state initialization, this is the place to go...
        if seed is not None:
            self._seed(seed=seed)
        if options is not None and "new_state" in options:
            state = options["new_state"]
            assert (isinstance(state,list) or isinstance(state,tuple)) and len(state)==2, "New state must be tuple/list with 2 components"
            self.state = (np.float32(state[0]),np.float32(state[1]))
            return np.array(self.state), {'crash': self.is_crash(state)}
        pos = self.np_random.uniform(low=5, high=0.8*self.MAX_VALUE, size=(1,))[0]
        # We must not approach too fast (in which case braking would not stop us anymore)
        min_velocity = -np.sqrt(pos*2*self.B)
        # Hypothetical constraint on the other side:
        # (MAX_VALUE-pos) <= vel^2 / (2*B)
        # We must not fall behind too fast (in which case accelerating would not help us anymore)
        max_velocity = np.sqrt((self.MAX_VALUE-pos)*2*self.A)
        vel = self.np_random.uniform(low=min_velocity,high=max_velocity, size=(1,))[0]
        self.state = (np.float32(pos), np.float32(vel))

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
        x_scale = (screen_width-100-2*cartwidth)/self.MAX_VALUE

        relativeDistance = cartwidth * 2

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))
            #pygame.Surface((screen_width, screen_height))

            self.nn_cart = pygame.image.load("/home/s2908217/autonomous_car_control/versaille_env/nn-car.png").convert_alpha()
            self.front_cart = pygame.image.load("/home/s2908217/autonomous_car_control/versaille_env/front-car.png").convert_alpha()
            original_width, original_height = self.nn_cart.get_size()

            scale_factor = cartwidth / original_width
            new_height = int(original_height * scale_factor)
            self.nn_cart = pygame.transform.flip(pygame.transform.smoothscale(self.nn_cart, (cartwidth, new_height)),False,True)
            self.front_car = pygame.transform.flip(pygame.transform.smoothscale(self.front_cart, (cartwidth, new_height)),False,True)

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
        pygame.draw.rect(self.surf, (34, 139, 34), pygame.Rect(0, carty, screen_width, carty))
        pygame.draw.rect(self.surf, (192, 192, 192), pygame.Rect(0, 0, screen_width, carty))


        for x in self.pole_positions:
            pygame.draw.rect(self.surf, (0, 0, 0), pygame.Rect(x, carty, pole_width, pole_height))

        relativeDistance, relativeVelocity = self.state
        followerx = screen_width - 100 - relativeDistance*x_scale - cartwidth
        leaderx = screen_width - 100
                   
        # Add a follower cart.
        l,r = -cartwidth, 0.0
        t,b = cartheight, 0.0
        l += followerx
        b += carty*0.75
        #gfxdraw.filled_polygon(self.surf, coords, (0,0,0))
        self.surf.blit(self.nn_cart, (l,b))

        # Add leader cart
        l,r = -cartwidth, 0.0
        t,b = cartheight, 0.0
        l += leaderx
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