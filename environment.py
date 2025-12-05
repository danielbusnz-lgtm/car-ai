import numpy as np
import pygame
import math


class CarEnvironment:

    def __init__(self, width=800, height=600):
        self.width=width
        self.height=height

        self.track_center_x = width// 2
        self.track_center_y = height// 2

        self.track_radius_x = 250
        self.track_radius_y = 150
        

        self.track_width = 80

        self.max_velocity = 8.0
        self.steering_speed = 5
        
        #How the car actually sees
        self.num_rays = 5
        self.ray_length = 200
        self.ray_angles = [-90, -45, 0, 45, 90]
        
        #Number of possible actions the car can take
        self.action_space = 3

        #What the agent can see, 5 directions(ray angles and speed)
        self.state_space = 7
        
        #Intialize the pygame screen
        self.screen = None
        self.clock = None

        self.reset()

    def reset(self):
        
        #Putting car back at the starting line
        self.car_x = self.track_center_x
        self.car_y = self.track_center_y - self.track_radius_y + self.track_width//2

        self.car_angle = 0

        self.car_velocity = 5.0
        
        #Amount of decisions and actions the car is making
        self.steps = 0
        self.total_reward = 0

        return self._get_state()
    
    def _get_state(self):


        ray_distance = self._cast_rays()
        

        #Normalize the rays 
        rays_norm = [d / self.ray_length for d in ray_distance]
        
        #Normalize velocity
        vel_norm = self.car_velocity / self.max_velocity
        
        #Normalize angles
        ang_norm = (self.car_angle % 360) / 360
        
        state = rays_norm +[vel_norm, ang_norm]

        return np.array(state, dtype = np.float32)

    def _cast_rays(self):
        ray_distances = []
        
        #Casting rays in all 5 direction
        for angle_offset in self.ray_angles:
            ray_angle = math.radians(self.car_angle + angle_offset)
            
            #How much to move in X
            ray_dx = math.cos(ray_angle)
            #How much to move in Y
            ray_dy = math.sin(ray_angle)
            
            min_distance = self.ray_length

            for step in range(1, int(self.ray_length)):
                #Check the ray for the x direction
                check_x = self.car_x + ray_dx * step
                #Check the ray for the y direction
                check_y = self.car_y + ray_dy * step
                
                if not self._is_on_track(check_x, check_y):
                    min_distance = step
                    break

            ray_distances.append(min_distance)

        return ray_distances

    def _is_on_track(self, x, y):
        dx = (x - self.track_center_x) / self.track_radius_x
        dy = (y - self.track_center_y) / self.track_radius_y

        distance_from_center = math.sqrt(dx**2 + dy**2)

        inner_ratio = (self.track_radius_x - self.track_width) / self.track_radius_x
          
        return inner_ratio <= distance_from_center <= 1.0



    def step(self, action):
        if action == 0:
            self.car_angle -= self.steering_speed
        elif action == 2: 
            self.car_angle += self.steering_speed
            
        self.car_angle = self.car_angle % 360
        
        #Convert from angle to radians
        angle_rad = math.radians(self.car_angle)
        
        #Move the car in the direction based on the action
        self.car_x += math.cos(angle_rad) * self.car_velocity
        self.car_y += math.sin(angle_rad) * self.car_velocity
        
        #Check if the car has crashed
        on_track = self._is_on_track(self.car_x, self.car_y)
        

        #Reward function
        if on_track:
            reward = 1.0
        else:
            reward = -100.0

        done = not on_track 
        
        self.steps += 1 
        self.total_reward += reward
        
        #Cast the rays again so the agent can see where it just went
        state = self._get_state()
        
        info = {
            'steps': self.steps,
            'total_reward': self.total_reward
        }

        return state, reward, done, info

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen =  pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Car RL Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((50, 50, 50))

        pygame.draw.ellipse(
            self.screen,
            (255, 255, 255),
            (
                self.track_center_x - self.track_radius_x,
                self.track_center_y - self.track_radius_y,
                self.track_radius_x * 2,
                self.track_radius_y * 2
            ),
            2
        )

        inner_radius_x = self.track_radius_x - self.track_width
        inner_radius_y = self.track_radius_y - self.track_width
        pygame.draw.ellipse(
            self.screen,
            (255,255,255),
            (
                self.track_center_x - inner_radius_x,
                self.track_center_y - inner_radius_y,
                inner_radius_x * 2,
                inner_radius_y * 2
            ),
            2
        )

        self._draw_rays()

        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (int(self.car_x), int(self.car_y)),
            8
        )

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_rays(self):
        for angle_offset in self.ray_angles:
            ray_angle = math.radians(self.car_angle + angle_offset)

            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            hit_distance = self.ray_length

            for step in range(1, int(self.ray_length)):
                check_x = self.car_x + ray_dx * step
                check_y = self.car_y + ray_dy * step

                if not self._is_on_track(check_x, check_y):
                    hit_distance = step
                    break
            end_x = self.car_x + hit_distance * ray_dx
            end_y = self.car_y + hit_distance * ray_dy

            pygame.draw.line(
                self.screen,
                (0, 255, 0),
                (int(self.car_x), int(self.car_y)),
                (int(end_x), int(end_y)),
                1
            )
