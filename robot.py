import numpy as np
import matplotlib.pyplot as plt

# Shared data storage for position, speed, angular speed, and timestamps
def data_meas(self):
    # get data from vicon and store it for other's to use
    return 0

class Robot:
    def __init__(self, robot_id,max_speed,max_angular_velocity,max_position,battDrainRate,delta_t):
        self.robot_id = robot_id
        self.max_position=max_position
        self.max_speed = max_speed
        self.max_angular_velocity = max_angular_velocity
        self.battDrainRate=battDrainRate
        self.delta_t=delta_t
        self.plot_initialized = False  # To check if the plot has been initialized
        self.plot_object = None  # To store the plot object
        
    def update_data(self, x, y, velocity, angular_velocity, theta, battery_level, timestamp):
        self.x = x
        self.y=y
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.theta = theta
        self.battery_level = battery_level
        self.timestamp = timestamp
        
    def step(self, v, om):
        # Update the robot's pose, simple kinematics
        self.x += v * np.cos(self.theta) * self.delta_t      
        self.y += v * np.sin(self.theta) * self.delta_t
        self.theta += om * self.delta_t
        Vr=v+(0.5*om)
        Vl=v-(0.5*om)
        self.battery_level -= self.battDrainRate*(np.absolute(Vr)+np.absolute(Vl))*self.delta_t

        # Apply position constraints
        self.x = min(self.max_position, max(-self.max_position, self.x))
        self.y = min(self.max_position, max(-self.max_position, self.y))
        
    def plot(self,col,xdes,ydes):
        if not self.plot_initialized:
            plt.ion()  # Enable interactive mode if the plot hasn't been initialized yet
            self.plot_object = plt.figure()
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Jonny Robot')
            plt.grid()
            #plt.xlim([-self.max_position, self.max_position])
            #plt.ylim([-self.max_position, self.max_position])
            plt.scatter(xdes,ydes,color='black',marker='s')
            self.plot_initialized = True

        # Plot the current position and orientation of the robot
        if self.plot_object is not None:
            plt.scatter(self.x, self.y, color = col,s=4,marker='o', label='Robot Position')
            
            
            
