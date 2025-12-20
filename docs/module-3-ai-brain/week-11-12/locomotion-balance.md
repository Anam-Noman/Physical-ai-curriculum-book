---
sidebar_position: 2
---

# Bipedal Locomotion and Balance

## Introduction to Humanoid Locomotion

Humanoid locomotion represents one of the most challenging problems in robotics, requiring the robot to maintain balance while performing complex, dynamic movements. The ability to walk on two legs is fundamental to humanoid robots, enabling them to navigate human-centric environments effectively. Unlike wheeled robots, bipedal robots must continuously negotiate the balance between stability and mobility, managing complex interactions between multiple degrees of freedom and the environment.

Successful locomotion in humanoid robots requires sophisticated control algorithms that can handle the inherent instability of bipedal movement while adapting to environmental changes in real-time. This challenge is amplified by the need to maintain balance during gait transitions, handle external disturbances, and manage energy efficiency during sustained locomotion.

## Principles of Bipedal Movement

### Biomechanics and Human Walking

Human walking involves a complex interplay of:
- **Double Support Phase**: Both feet in contact with ground
- **Single Support Phase**: One foot in contact, the other swinging
- **Heel Strike**: Initial contact of leading foot with ground
- **Toe Off**: Push-off action with trailing foot
- **Mid-Swing**: Highest point of swinging foot

The human body treats walking as a controlled falling process, where the center of mass (COM) oscillates laterally and vertically while maintaining forward momentum. This natural dynamic behavior is what robot designers strive to replicate.

### Dynamic vs Static Stability

- **Static Stability**: Center of pressure remains within the support polygon at all times. Applicable to slow movement or stationary postures.
  - Pros: Simple to implement, robust to disturbances
  - Cons: Energy inefficient, unnatural gait, slow movement

- **Dynamic Stability**: Center of mass is allowed to move outside the support polygon during portions of gait.
  - Pros: Energy efficient, natural gait, faster movement
  - Cons: Computationally complex, sensitive to disturbances

## Walking Gait Phases

### Gait Cycle Analysis

```
Gait Cycle (2 steps) -> 100%
├── Stance Phase (60%)
│   ├── Initial Double Support (8%)
│   ├── Single Support (40%)
│   ├── Terminal Double Support (8%)
└── Swing Phase (40%)
    ├── Early Swing (10%)
    ├── Mid Swing (20%)
    ├── Late Swing (10%)
```

### Single Support Phase

During single support, the stance leg bears the full body weight while the swing leg moves forward. This phase requires precise balance control:

```python
import numpy as np
import matplotlib.pyplot as plt

class SingleSupportController:
    def __init__(self, robot_params):
        self.robot_params = robot_params
        self.com_height = robot_params['com_height']  # Center of mass height
        self.gravity = 9.81  # m/s^2
        
    def calculate_pendulum_motion(self, com_position, com_velocity):
        """
        Approximate bipedal motion using inverted pendulum model
        """
        # Inverted pendulum model for walking
        # COM behaves like an inverted pendulum during single support
        x, y, z = com_position
        vx, vy, vz = com_velocity
        
        # Calculate angular velocity of COM around contact point
        # This is simplified - real implementation would consider full dynamics
        omega_x = vy / self.com_height
        omega_y = -vx / self.com_height
        
        # Calculate acceleration required to maintain pendulum motion
        ax = -omega_x * vy
        ay = omega_y * vx
        az = -self.gravity  # dominated by gravity
        
        return np.array([ax, ay, az])

    def balance_control(self, current_state, desired_state):
        """
        Calculate balance corrections during single support
        """
        # Current COM position and velocity
        current_com_pos = current_state['com_position']
        current_com_vel = current_state['com_velocity']
        
        # Desired COM trajectory
        desired_com_pos = desired_state['com_position']
        desired_com_vel = desired_state['com_velocity']
        
        # Calculate error
        pos_error = desired_com_pos - current_com_pos
        vel_error = desired_com_vel - current_com_vel
        
        # PID control for balance correction
        kp = 100.0  # Proportional gain for position
        kv = 20.0   # Derivative gain for velocity (damping)
        
        # Calculate corrective accelerations
        corrective_acc = kp * pos_error + kv * vel_error
        
        # Limit maximum correction to avoid aggressive movements
        max_correction = 2.0  # m/s^2
        magnitude = np.linalg.norm(corrective_acc)
        if magnitude > max_correction:
            corrective_acc = (corrective_acc / magnitude) * max_correction
        
        return corrective_acc

class WalkingPatternGenerator:
    def __init__(self, step_length=0.3, step_height=0.1, step_duration=1.0):
        self.step_length = step_length
        self.step_height = step_height
        self.step_duration = step_duration
        
    def generate_step_trajectory(self, start_pos, target_pos, support_foot_pos, 
                                 phase_percentage):
        """
        Generate trajectory for a single step including swing leg motion
        """
        # Calculate intermediate position based on gait phase
        t = phase_percentage  # 0.0 to 1.0
        
        # Horizontal movement (progressive from start to target)
        x = start_pos[0] * (1 - t) + target_pos[0] * t
        y = start_pos[1] * (1 - t) + target_pos[1] * t
        
        # Vertical movement (lift and land)
        # Using sine function for smooth lifting and landing
        vertical_lift = self.step_height * np.sin(np.pi * t)
        
        # Add slight arc to swing trajectory
        if t < 0.5:
            # Ascending part of step
            z = start_pos[2] + vertical_lift
        else:
            # Descending part of step
            z = target_pos[2] + vertical_lift
        
        # For lateral stepping, adjust with cosine for smooth side-to-side motion
        lateral_adjustment = 0.0  # Can be adjusted for side steps
        if hasattr(self, 'lateral_step') and self.lateral_step:
            lateral_adjustment = 0.1 * np.sin(np.pi * t)
        
        return [x, y + lateral_adjustment, z]
    
    def generate_gait_sequence(self, num_steps, forward_speed=0.3, step_height=0.1):
        """
        Generate complete gait sequence for multiple steps
        """
        gait_sequence = []
        
        # Starting position
        current_pos = np.array([0.0, 0.0, 0.8])  # Start at COM height
        
        for step_idx in range(num_steps):
            # Determine which foot steps for this step
            # Alternating pattern: step with left foot on odd steps, right on even
            stance_foot = 'right' if step_idx % 2 == 0 else 'left'
            swing_foot = 'left' if stance_foot == 'right' else 'right'
            
            # Generate trajectory for this step
            step_trajectory = self.create_single_step_trajectory(
                step_idx, stance_foot, forward_speed, step_height
            )
            
            gait_sequence.append({
                'step_number': step_idx,
                'stance_foot': stance_foot,
                'swing_foot': swing_foot,
                'trajectory': step_trajectory
            })
        
        return gait_sequence
    
    def create_single_step_trajectory(self, step_idx, stance_foot, forward_speed, step_height):
        """
        Create detailed trajectory for a single step
        """
        # Number of intermediate positions in the step
        num_intermediate_points = 50
        trajectory_points = []
        
        # Determine swing foot start and target positions
        if stance_foot == 'right':
            # Left foot swings forward
            start_pos = [step_idx * forward_speed, -0.1, 0.0]  # Left foot position
            target_pos = [(step_idx + 1) * forward_speed, -0.1, 0.0]  # Next left position
        else:
            # Right foot swings forward
            start_pos = [step_idx * forward_speed, 0.1, 0.0]  # Right foot position
            target_pos = [(step_idx + 1) * forward_speed, 0.1, 0.0]  # Next right position
        
        # Generate trajectory points
        for i in range(num_intermediate_points):
            t = i / (num_intermediate_points - 1)  # Normalized time (0 to 1)
            
            # Calculate position including lifting motion
            if t < 0.5:
                # First half: lifting and moving forward
                pos_x = start_pos[0] + (target_pos[0] - start_pos[0]) * t * 2
                pos_y = start_pos[1]
                pos_z = start_pos[2] + step_height * np.sin(np.pi * t * 2)
            else:
                # Second half: descending and reaching target
                pos_x = start_pos[0] + (target_pos[0] - start_pos[0]) * t * 2
                pos_y = start_pos[1] 
                pos_z = target_pos[2] + step_height * np.sin(np.pi + np.pi * (t - 0.5) * 2)
            
            trajectory_points.append([pos_x, pos_y, pos_z])
        
        return trajectory_points
```

## Balance Control Strategies

### Center of Mass (COM) Control

The center of mass is the key variable for balance control in humanoid robots. Different strategies have been developed to manage COM position and velocity.

```python
class COMBalancer:
    def __init__(self, robot_params):
        self.robot_params = robot_params
        self.com_position = robot_params['com_initial']
        self.com_velocity = np.zeros(3)
        self.com_history = []
        
        # PID controller parameters for COM control
        self.pid_x = {'kp': 80.0, 'ki': 0.1, 'kd': 15.0}
        self.pid_y = {'kp': 80.0, 'ki': 0.1, 'kd': 15.0}
        self.pid_z = {'kp': 10.0, 'ki': 0.01, 'kd': 5.0}  # For height control
        
    def calculate_zmp_from_com(self, com_pos, com_vel, com_acc):
        """
        Calculate Zero Moment Point (ZMP) from COM dynamics
        ZMP_x = x_com - h/g * x_com_ddot
        ZMP_y = y_com - h/g * y_com_ddot
        where h = height of COM above ground, g = gravity
        """
        h = self.robot_params['com_height']  # Height of COM above ground
        g = 9.81  # Gravity constant
        
        zmp_x = com_pos[0] - (h / g) * com_acc[0]
        zmp_y = com_pos[1] - (h / g) * com_acc[1]
        
        return np.array([zmp_x, zmp_y, 0.0])
    
    def track_com_trajectory(self, desired_com_traj, current_com_pos, dt):
        """
        Track a desired COM trajectory using feedback control
        """
        # Get desired position, velocity and acceleration
        desired_pos = desired_com_traj['position']
        desired_vel = desired_com_traj['velocity']
        desired_acc = desired_com_traj['acceleration']
        
        # Calculate errors
        pos_error = desired_pos - current_com_pos
        vel_error = desired_vel - self.com_velocity  # Need to track current velocity
        
        # Apply PID control
        control_force = np.zeros(3)
        control_force[0] = (self.pid_x['kp'] * pos_error[0] + 
                           self.pid_x['kd'] * vel_error[0])
        control_force[1] = (self.pid_y['kp'] * pos_error[1] + 
                           self.pid_y['kd'] * vel_error[1])
        control_force[2] = (self.pid_z['kp'] * pos_error[2] + 
                           self.pid_z['kd'] * vel_error[2])
        
        # Update COM dynamics
        total_force = control_force - np.array([0, 0, self.robot_params['total_mass'] * 9.81])
        com_acc = total_force / self.robot_params['total_mass']
        
        # Update velocity and position (Euler integration)
        self.com_velocity += com_acc * dt
        self.com_position += self.com_velocity * dt
        
        # Record for history
        self.com_history.append(self.com_position.copy())
        
        return control_force, com_acc

class CapturePointController:
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)
        
    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate capture point where robot should step to stop
        Capture Point = CoM position + CoM velocity / omega
        where omega = sqrt(g / h)
        """
        cp_x = com_pos[0] + com_vel[0] / self.omega
        cp_y = com_pos[1] + com_vel[1] / self.omega
        
        return np.array([cp_x, cp_y, 0.0])
    
    def should_step(self, com_pos, com_vel, support_polygon):
        """
        Determine if a step is needed based on capture point
        """
        capture_point = self.calculate_capture_point(com_pos, com_vel)
        
        # Check if capture point is outside support polygon
        if not self.point_in_polygon(capture_point[:2], support_polygon):
            return True, capture_point
        else:
            return False, capture_point
    
    def point_in_polygon(self, point, polygon):
        """
        Check if point is inside polygon using ray casting algorithm
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
```

### ZMP (Zero Moment Point) Based Control

ZMP-based control is a cornerstone of stable bipedal locomotion:

```python
class ZMPBalancer:
    def __init__(self, support_polygon=[[0, -0.1], [0.3, -0.1], [0.3, 0.1], [0, 0.1]], 
                 com_height=0.8):
        self.support_polygon = support_polygon
        self.com_height = com_height
        self.gravity = 9.81
        self.zmp_trajectory = []
        self.zmp_history = []
        
    def calculate_zmp(self, ground_reaction_forces, ground_reaction_moments):
        """
        Calculate ZMP from ground reaction forces and moments
        ZMP = (M_y - F_x * z_0) / F_z, (-M_x - F_y * z_0) / F_z
        where z_0 is height above ground, M are moments, F are forces
        """
        F_x, F_y, F_z = ground_reaction_forces
        M_x, M_y, M_z = ground_reaction_moments
        z_0 = 0  # Height of ZMP plane (ground level)
        
        if abs(F_z) < 0.01:  # Avoid division by near-zero force
            return np.array([0.0, 0.0, 0.0])
        
        zmp_x = (M_y - F_x * z_0) / F_z
        zmp_y = (-M_x - F_y * z_0) / F_z
        
        return np.array([zmp_x, zmp_y, zmp_0])
    
    def track_zmp_trajectory(self, desired_zmp_trajectory, current_com_state):
        """
        Track desired ZMP trajectory using COM control
        """
        com_pos, com_vel, com_acc = current_com_state
        
        # Calculate current ZMP
        current_zmp = self.calculate_zmp_from_com(com_pos, com_vel, com_acc)
        
        # Calculate ZMP error
        zmp_error = desired_zmp_trajectory - current_zmp
        
        # Design control law to track ZMP
        # For a simplified model: F = m*(g*zmp_error/h + com_ddot_desired)
        k_p = 100.0  # Proportional gain
        k_v = 20.0   # Velocity feedback gain
        
        # Calculate desired COM acceleration to correct ZMP
        desired_com_acc = np.zeros(3)
        desired_com_acc[0] = k_p * zmp_error[0] * self.gravity / self.com_height
        desired_com_acc[1] = k_p * zmp_error[1] * self.gravity / self.com_height
        
        # Also add velocity damping
        desired_com_acc[0] -= k_v * com_vel[0]
        desired_com_acc[1] -= k_v * com_vel[1]
        
        return desired_com_acc, zmp_error
    
    def calculate_zmp_from_com(self, com_pos, com_vel, com_acc):
        """
        Calculate ZMP from COM kinematics (inverse of the relationship)
        """
        h = self.com_height
        g = self.gravity
        
        zmp_x = com_pos[0] - (h / g) * com_acc[0]
        zmp_y = com_pos[1] - (h / g) * com_acc[1]
        
        return np.array([zmp_x, zmp_y, 0.0])
    
    def generate_zmp_trajectory(self, step_positions, step_duration=1.0, dt=0.01):
        """
        Generate ZMP trajectory following reference foot positions
        """
        num_steps = int(step_duration / dt)
        zmp_trajectory = []
        
        for i in range(len(step_positions) - 1):
            start_pos = step_positions[i]
            end_pos = step_positions[i + 1]
            
            for j in range(num_steps):
                t = j / num_steps  # Progress from 0 to 1
                
                # Smooth transition between foot positions
                current_zmp = (1 - t) * start_pos + t * end_pos
                
                # Add slight smoothing near foot placements
                if t < 0.1:  # Early in transition
                    # Stay near previous foot
                    current_zmp = start_pos * 0.9 + current_zmp * 0.1
                elif t > 0.9:  # Late in transition
                    # Approach next foot position
                    current_zmp = end_pos * 0.9 + current_zmp * 0.1
                
                zmp_trajectory.append(current_zmp)
        
        return zmp_trajectory
```

## Walking Pattern Generation

### Preview Control Method

Preview control uses future reference trajectories to anticipate and compensate for robot dynamics:

```python
class PreviewController:
    def __init__(self, preview_horizon=20, dt=0.01):
        self.preview_horizon = preview_horizon
        self.dt = dt
        self.com_height = 0.8
        self.gravity = 9.81
        
        # Calculate preview control gain matrix
        self.K_x, self.K_ref = self.calculate_preview_gains()
        
    def calculate_preview_gains(self):
        """
        Calculate preview control gains using Riccati equation solution
        This is a simplified implementation
        """
        # For inverted pendulum model: x_ddot = g/h * x - g/h * zmp
        # State: x = [com_pos, com_vel], u = zmp_ref
        # A = [[0, 1], [g/h, 0]], B = [0, -g/h]
        A = np.array([[0, 1], [self.gravity/self.com_height, 0]])
        B = np.array([0, -self.gravity/self.com_height])
        
        # Cost matrices (Q for state error, R for control effort)
        Q = np.diag([10.0, 1.0])  # Higher penalty on position error
        R = np.array([[0.1]])      # Penalty on ZMP deviation
        
        # Solve discrete Riccati equation
        # This requires more complex mathematics in practice
        # For simplicity, we'll use approximate gains
        K_x = np.array([3.16, 3.46])  # State feedback gains
        K_ref = np.array([3.16])      # Feedforward gains
        
        return K_x, K_ref
    
    def generate_reference_trajectory(self, step_sequence, dt=0.01):
        """
        Generate reference COM trajectory based on planned foot steps
        """
        # Calculate total simulation time
        total_time = len(step_sequence) * 1.0  # 1 second per step
        num_points = int(total_time / dt)
        
        # Initialize reference arrays
        ref_com_x = np.zeros(num_points)
        ref_com_y = np.zeros(num_points)
        ref_zmp_x = np.zeros(num_points)
        ref_zmp_y = np.zeros(num_points)
        
        # Generate smooth transition between foot placements
        for i, step in enumerate(step_sequence):
            start_idx = int(i / len(step_sequence) * num_points)
            end_idx = int((i + 1) / len(step_sequence) * num_points)
            
            if i < len(step_sequence) - 1:
                next_step = step_sequence[i + 1]
                # Smooth transition between current and next step
                for j in range(start_idx, min(end_idx, num_points)):
                    t = (j - start_idx) / (end_idx - start_idx)
                    # Use cosine interpolation for smooth transitions
                    ref_com_x[j] = step['com_x'] * (1 + np.cos(np.pi * t)) / 2 + \
                                   next_step['com_x'] * (1 - np.cos(np.pi * t)) / 2
                    ref_com_y[j] = step['com_y'] * (1 + np.cos(np.pi * t)) / 2 + \
                                   next_step['com_y'] * (1 - np.cos(np.pi * t)) / 2
            else:
                # Last step - hold position
                for j in range(start_idx, num_points):
                    ref_com_x[j] = step['com_x']
                    ref_com_y[j] = step['com_y']
        
        return {
            'ref_com_x': ref_com_x,
            'ref_com_y': ref_com_y,
            'ref_zmp_x': ref_zmp_x,
            'ref_zmp_y': ref_zmp_y
        }
    
    def preview_control_step(self, current_state, reference_sequence, current_time_idx):
        """
        Calculate control command using preview control
        """
        # Current state [com_pos, com_vel]
        x_current = np.array([
            current_state['com_pos'][0],  # x position
            current_state['com_vel'][0]   # x velocity
        ])
        
        y_current = np.array([
            current_state['com_pos'][1],  # y position
            current_state['com_vel'][1]   # y velocity
        ])
        
        # Extract preview reference (next few reference points)
        ref_start = current_time_idx
        ref_end = min(current_time_idx + self.preview_horizon, len(reference_sequence['ref_com_x']))
        
        ref_x_preview = reference_sequence['ref_com_x'][ref_start:ref_end]
        ref_y_preview = reference_sequence['ref_com_y'][ref_start:ref_end]
        
        # Calculate control command (simplified)
        # u = -K_x * x + sum(K_ref * (ref_future - ref_current))
        command_x = -np.dot(self.K_x, x_current)
        command_y = -np.dot(self.K_x, y_current)
        
        # Add preview compensation
        for i, (ref_x, ref_y) in enumerate(zip(ref_x_preview, ref_y_preview)):
            if i == 0:
                command_x += self.K_ref[0] * (ref_x - x_current[0])
                command_y += self.K_ref[0] * (ref_y - y_current[0])
            else:
                # Diminish effect of future previews
                weight = np.exp(-0.1 * i)
                command_x += weight * self.K_ref[0] * (ref_x - x_current[0])
                command_y += weight * self.K_ref[0] * (ref_y - y_current[0])
        
        return np.array([command_x, command_y])

class WalkingTrajectoryGenerator:
    def __init__(self, robot_params):
        self.params = robot_params
        self.preview_controller = PreviewController()
        self.zmp_controller = ZMPBalancer()
        self.com_controller = COMBalancer(robot_params)
    
    def generate_walk_trajectory(self, num_steps, step_length=0.3, step_height=0.1, 
                                 step_duration=1.0):
        """
        Generate complete walking trajectory with balance control
        """
        print(f"Generating {num_steps} walking steps...")
        
        # Generate basic step sequence
        pattern_gen = WalkingPatternGenerator(step_length, step_height, step_duration)
        step_sequence = pattern_gen.generate_gait_sequence(num_steps)
        
        # Generate reference trajectory
        reference_traj = self.preview_controller.generate_reference_trajectory(
            [{'com_x': i * step_length, 'com_y': 0.0} for i in range(num_steps)]
        )
        
        # Initialize simulation parameters
        dt = 0.01  # 100 Hz control loop
        total_time = num_steps * step_duration
        num_sim_steps = int(total_time / dt)
        
        # Initialize state variables
        com_pos = np.array([0.0, 0.0, self.params['com_height']])
        com_vel = np.array([0.0, 0.0, 0.0])
        com_acc = np.array([0.0, 0.0, 0.0])
        
        # Store trajectory for visualization
        trajectory_log = {
            'time': [],
            'com_pos': [],
            'com_vel': [],
            'com_acc': [],
            'zmp_pos': [],
            'support_polygon': []
        }
        
        # Walking simulation loop
        for sim_step in range(num_sim_steps):
            t = sim_step * dt
            time_idx = int(t / dt)
            
            # Determine support polygon based on gait phase
            support_poly = self.calculate_support_polygon(t, step_sequence)
            
            # Current state
            current_state = {
                'com_pos': com_pos,
                'com_vel': com_vel
            }
            
            # Use preview controller for trajectory following
            control_cmd = self.preview_controller.preview_control_step(
                current_state, reference_traj, time_idx
            )
            
            # Apply balance corrections
            current_zmp = self.zmp_controller.calculate_zmp_from_com(
                com_pos, com_vel, com_acc
            )
            
            # Check if ZMP is within support polygon
            if not self.point_in_support_polygon(current_zmp[:2], support_poly):
                # Apply emergency balance correction
                balance_corr = self.emergency_balance_correction(
                    com_pos, com_vel, support_poly
                )
                control_cmd += balance_corr
            
            # Update COM dynamics
            force = self.calculate_control_force(control_cmd, com_pos, com_vel)
            acc = force / self.params['total_mass']
            
            # Integrate dynamics
            com_vel += acc * dt
            com_pos += com_vel * dt
            
            # Log trajectory
            trajectory_log['time'].append(t)
            trajectory_log['com_pos'].append(com_pos.copy())
            trajectory_log['com_vel'].append(com_vel.copy())
            trajectory_log['com_acc'].append(acc.copy())
            trajectory_log['zmp_pos'].append(current_zmp)
            trajectory_log['support_polygon'].append(support_poly)
        
        print(f"Walking trajectory generation completed for {num_steps} steps")
        return trajectory_log
    
    def calculate_support_polygon(self, current_time, step_sequence):
        """
        Calculate current support polygon based on step timing
        """
        # Simplified: alternate between single and double support polygons
        step_duration = 1.0  # Fixed for this example
        step_num = int(current_time / step_duration)
        
        if step_num >= len(step_sequence):
            step_num = len(step_sequence) - 1
        
        # Determine feet positions based on timing within step cycle
        time_in_step = current_time % step_duration
        
        if time_in_step < step_duration * 0.1 or time_in_step > step_duration * 0.9:
            # Double support phase (both feet down)
            # Create polygon encompassing both feet
            left_pos = [step_num * 0.3, -0.1, 0.0]
            right_pos = [step_num * 0.3, 0.1, 0.0]
            return self.create_foot_polygon(left_pos, right_pos)
        else:
            # Single support phase
            stance_foot = 'left' if step_num % 2 != 0 else 'right'
            foot_pos = ([step_num * 0.3, -0.1, 0.0] if stance_foot == 'left' 
                       else [step_num * 0.3, 0.1, 0.0])
            return self.create_single_foot_polygon(foot_pos)
    
    def create_foot_polygon(self, left_pos, right_pos):
        """
        Create support polygon from foot positions
        """
        # Simplified rectangular approximation
        margin = 0.05  # Safety margin
        
        min_x = min(left_pos[0], right_pos[0]) - 0.1
        max_x = max(left_pos[0], right_pos[0]) + 0.1
        min_y = min(left_pos[1], right_pos[1]) - 0.1
        max_y = max(left_pos[1], right_pos[1]) + 0.1
        
        return [
            [min_x, min_y],
            [max_x, min_y], 
            [max_x, max_y],
            [min_x, max_y]
        ]
    
    def calculate_control_force(self, control_cmd, com_pos, com_vel):
        """
        Calculate control force from command
        """
        # Simplified control model
        kp = 100.0
        kv = 20.0
        
        force_x = kp * (control_cmd[0] - com_pos[0]) - kv * com_vel[0]
        force_y = kp * (control_cmd[1] - com_pos[1]) - kv * com_vel[1]
        force_z = self.params['total_mass'] * 9.81  # Compensate for gravity
        
        return np.array([force_x, force_y, force_z])
    
    def emergency_balance_correction(self, com_pos, com_vel, support_polygon):
        """
        Apply emergency balance correction if ZMP is outside support polygon
        """
        # Calculate correction to move ZMP back inside polygon
        zmp = self.zmp_controller.calculate_zmp_from_com(
            com_pos, com_vel, np.array([0, 0, 0])  # Assume zero acceleration initially
        )
        
        # Find closest valid ZMP position inside polygon
        closest_point = self.find_closest_point_in_polygon(zmp[:2], support_polygon)
        correction = closest_point - zmp[:2]
        
        # Amplify correction for urgency
        return correction * 10.0  # Scale factor for emergency response
```

## Advanced Locomotion Patterns

### Turning and Steering

Implementing turns and steering for humanoid robots adds complexity to the basic walking pattern:

```python
class TurningController:
    def __init__(self, robot_params):
        self.params = robot_params
        self.base_walking_generator = WalkingPatternGenerator()
        self.turn_radius = 1.0  # Default turn radius (meters)
        
    def generate_turn_trajectory(self, turn_angle, forward_steps=3, 
                                 turn_steps=5, step_length=0.3):
        """
        Generate trajectory for turning maneuver
        """
        total_steps = forward_steps + turn_steps
        trajectory = []
        
        # First, perform straight walking
        for i in range(forward_steps):
            straight_step = {
                'step_number': i,
                'position': [i * step_length, 0, 0],
                'orientation': 0,  # No turn yet
                'stance_foot': 'right' if i % 2 == 0 else 'left'
            }
            trajectory.append(straight_step)
        
        # Then, perform turning steps
        angle_increment = turn_angle / turn_steps
        current_angle = 0
        
        for i in range(turn_steps):
            current_angle += angle_increment
            
            # Calculate turn position using circular motion
            arc_length = self.turn_radius * current_angle
            x_pos = forward_steps * step_length + self.turn_radius * np.sin(current_angle)
            y_pos = self.turn_radius * (1 - np.cos(current_angle))
            
            turn_step = {
                'step_number': forward_steps + i,
                'position': [x_pos, y_pos, 0],
                'orientation': current_angle,
                'stance_foot': 'right' if (forward_steps + i) % 2 == 0 else 'left'
            }
            trajectory.append(turn_step)
        
        return trajectory
    
    def adjust_com_trajectory_for_turning(self, base_trajectory, turn_params):
        """
        Adjust COM trajectory to accommodate turning dynamics
        """
        # Add centripetal acceleration compensation during turns
        adjusted_trajectory = []
        
        for step in base_trajectory:
            # Calculate lateral forces needed for turning
            if abs(step['orientation']) > 0.1:  # Significant turn
                # Add lateral COM displacement for turning stability
                lat_displacement = 0.05 * np.sign(step['position'][1])  # Lean into turn
                step['com_adjustment'] = [0, lat_displacement, 0]
            else:
                step['com_adjustment'] = [0, 0, 0]
            
            adjusted_trajectory.append(step)
        
        return adjusted_trajectory

### Terrain Adaptation

Adapting to different terrains requires adjusting gait parameters:

class TerrainAdaptiveWalker:
    def __init__(self, robot_params):
        self.params = robot_params
        self.terrain_classifications = {
            'flat': {'max_slope': 0.05, 'height_var': 0.02},
            'uneven': {'max_slope': 0.15, 'height_var': 0.08},
            'rough': {'max_slope': 0.30, 'height_var': 0.15},
            'stairs': {'max_slope': 1.00, 'height_var': 0.20}  # Step height
        }
        
    def classify_terrain(self, terrain_data):
        """
        Classify terrain based on sensor data
        """
        # Calculate slope and height variation from terrain data
        slopes = np.gradient(terrain_data['elevation'])
        max_slope = np.max(np.abs(slopes))
        height_var = np.std(terrain_data['elevation'])
        
        # Classify terrain type
        if max_slope <= 0.05 and height_var <= 0.02:
            return 'flat'
        elif max_slope <= 0.15 and height_var <= 0.08:
            return 'uneven'
        elif max_slope <= 0.30 and height_var <= 0.15:
            return 'rough'
        else:
            return 'complex'  # Stairs or very rough
    
    def adapt_gait_for_terrain(self, base_gait, terrain_type):
        """
        Adapt gait parameters based on terrain type
        """
        adapted_gait = base_gait.copy()
        
        if terrain_type == 'flat':
            # Standard gait, no modifications
            pass
        elif terrain_type == 'uneven':
            # Increase step height and reduce step length
            for step in adapted_gait:
                step['step_height'] *= 1.5  # Higher clearance
                step['step_length'] *= 0.9  # Shorter steps
                step['stance_time'] *= 1.1  # Longer stance for stability
        elif terrain_type == 'rough':
            # Further increase step height, reduce speed, wider steps
            for step in adapted_gait:
                step['step_height'] *= 2.0  # Much higher clearance
                step['step_length'] *= 0.7  # Shorter steps
                step['stance_time'] *= 1.3  # Longer stance
                step['step_width'] *= 1.2  # Wider stance
        elif terrain_type == 'complex':
            # Use specialized step-by-step planning
            for step in adapted_gait:
                step['step_height'] *= 2.5  # Maximum clearance
                step['step_length'] *= 0.5  # Very short steps
                step['cautious_approach'] = True
        
        return adapted_gait
    
    def reactive_control_for_terrain(self, current_state, terrain_data):
        """
        Implement reactive control for terrain changes
        """
        # Detect terrain changes in real-time
        if terrain_data['slope'] > self.terrain_classifications['uneven']['max_slope']:
            # Switch to uneven terrain gait
            return self.adapt_gait_for_terrain(
                [current_state], 
                'uneven'
            )[0]
        elif terrain_data['height_variation'] > self.terrain_classifications['rough']['height_var']:
            # Switch to rough terrain gait
            return self.adapt_gait_for_terrain(
                [current_state], 
                'rough'
            )[0]
        else:
            # Maintain current gait
            return current_state
```

## Implementation in Simulation and Real Robots

### Simulation Setup

When implementing these algorithms in simulation:

```python
class SimulatorIntegration:
    def __init__(self, robot_model, controller):
        self.robot = robot_model
        self.controller = controller
        self.simulation_dt = 0.001  # Physics simulation timestep
        self.control_dt = 0.01      # Controller timestep (100 Hz)
        
    def simulation_loop(self, trajectory, duration=10.0):
        """
        Run walking simulation with controller
        """
        import time
        
        sim_time = 0.0
        control_update_counter = 0
        
        while sim_time < duration:
            # Physics simulation step
            self.update_physics()
            
            # Controller update (less frequent than physics)
            if control_update_counter % int(self.control_dt / self.simulation_dt) == 0:
                self.update_controller(trajectory, sim_time)
            
            # Update simulation time
            sim_time += self.simulation_dt
            control_update_counter += 1
            
            # Optionally, add real-time factor
            # time.sleep(0.0005)  # For real-time visualization
        
        return self.get_final_state()
    
    def update_controller(self, trajectory, sim_time):
        """
        Update controller with current robot state
        """
        # Get current robot state from simulation
        current_state = self.get_robot_state()
        
        # Calculate control commands based on desired trajectory
        control_commands = self.controller.calculate_control_commands(
            current_state, 
            self.get_trajectory_reference(sim_time, trajectory)
        )
        
        # Apply control commands to robot in simulation
        self.apply_control_commands(control_commands)
    
    def get_robot_state(self):
        """
        Get current state of robot from simulation
        """
        # This would interface with the physics engine
        state = {
            'com_position': self.robot.get_com_position(),
            'com_velocity': self.robot.get_com_velocity(),
            'joint_positions': self.robot.get_joint_positions(),
            'joint_velocities': self.robot.get_joint_velocities(),
            'foot_positions': self.robot.get_foot_positions(),
            'imu_data': self.robot.get_imu_data()
        }
        return state
    
    def apply_control_commands(self, commands):
        """
        Apply calculated control commands to robot
        """
        # Send commands to simulation robot model
        for joint_name, torque in commands.items():
            self.robot.apply_torque(joint_name, torque)
```

## Balance Recovery Strategies

### Disturbance Response

Humanoid robots need to handle unexpected disturbances during walking:

```python
class BalanceRecovery:
    def __init__(self, robot_params):
        self.params = robot_params
        self.state_estimator = StateEstimator()
        self.recovery_strategies = self.define_recovery_strategies()
        
    def define_recovery_strategies(self):
        """
        Define various balance recovery strategies
        """
        return {
            'ankle_strategy': {
                'activation_threshold': 0.05,  # meters COM displacement
                'duration': 0.1,  # seconds
                'effort': 'low'
            },
            'hip_strategy': {
                'activation_threshold': 0.10,
                'duration': 0.2,
                'effort': 'medium'
            },
            'stepping_strategy': {
                'activation_threshold': 0.15,
                'duration': 0.3,
                'effort': 'high'
            },
            'grabbing_strategy': {
                'activation_threshold': 0.20,
                'duration': 0.5,
                'effort': 'environment_dependent'
            }
        }
    
    def assess_disturbance_and_react(self, current_state, disturbance_force):
        """
        Assess disturbance magnitude and apply appropriate recovery
        """
        # Estimate state and calculate stability metrics
        estimated_state = self.state_estimator.estimate_state(current_state)
        
        # Calculate disturbance effect on COM
        com_displacement = self.calculate_com_effect(disturbance_force, estimated_state)
        
        # Select appropriate recovery strategy based on displacement
        for strategy_name, strategy_params in sorted(
            self.recovery_strategies.items(), 
            key=lambda x: x[1]['activation_threshold']
        ):
            if com_displacement > strategy_params['activation_threshold']:
                print(f"Activating {strategy_name} for disturbance recovery")
                return self.execute_recovery_strategy(strategy_name, current_state)
        
        return current_state  # No recovery needed
    
    def calculate_com_effect(self, force, state):
        """
        Calculate COM displacement caused by disturbance force
        """
        # Simplified model: impulse leads to COM displacement
        # In reality, this would consider full dynamics
        impulse = np.linalg.norm(force) * 0.01  # Assuming 0.01s duration
        mass = self.params['total_mass']
        acceleration = impulse / mass
        displacement = 0.5 * acceleration * (0.1**2)  # Assuming 0.1s to react
        
        return displacement
    
    def execute_recovery_strategy(self, strategy_name, current_state):
        """
        Execute the specified balance recovery strategy
        """
        if strategy_name == 'ankle_strategy':
            return self.ankle_balance_adjustment(current_state)
        elif strategy_name == 'hip_strategy':
            return self.hip_balance_adjustment(current_state)
        elif strategy_name == 'stepping_strategy':
            return self.emergency_stepping(current_state)
        elif strategy_name == 'grabbing_strategy':
            return self.emergency_grabbing(current_state)
        else:
            return current_state
    
    def ankle_balance_adjustment(self, current_state):
        """
        Shift balance using ankle joint torques
        """
        # Calculate required ankle torque to shift COM back
        current_com = current_state['com_position']
        desired_com = current_state['desired_com_position']  # Target position
        
        # PID control for ankle strategy
        kp = 200.0  # High gain for quick response
        kd = 50.0
        
        pos_error = desired_com - current_com
        # For ankle strategy, only adjust in sagittal and coronal planes
        
        # Calculate required ankle torques
        ankle_torques = {
            'left_ankle_pitch': kp * pos_error[0] * 0.5,  # Forward/back
            'left_ankle_roll': kp * pos_error[1] * 0.3,   # Side to side
            'right_ankle_pitch': kp * pos_error[0] * 0.5,
            'right_ankle_roll': kp * pos_error[1] * 0.3
        }
        
        return ankle_torques
    
    def hip_balance_adjustment(self, current_state):
        """
        Use hip torques for balance recovery
        """
        # More aggressive than ankle strategy - use hip joints
        current_com = current_state['com_position']
        desired_com = current_state['desired_com_position']
        
        pos_error = desired_com - current_com
        
        # Calculate hip torques needed
        hip_torques = {
            'left_hip_pitch': 150.0 * pos_error[0] + 40.0 * current_state['com_velocity'][0],
            'left_hip_roll': 100.0 * pos_error[1] + 30.0 * current_state['com_velocity'][1],
            'left_hip_yaw': 50.0 * pos_error[1] * 0.1,  # Small correction
            'right_hip_pitch': 150.0 * pos_error[0] + 40.0 * current_state['com_velocity'][0],
            'right_hip_roll': 100.0 * pos_error[1] + 30.0 * current_state['com_velocity'][1],
            'right_hip_yaw': 50.0 * pos_error[1] * 0.1
        }
        
        return hip_torques
    
    def emergency_stepping(self, current_state):
        """
        Execute emergency step to expand support polygon
        """
        # Calculate where to step based on COM position
        current_com = current_state['com_position']
        current_support_center = self.calculate_support_center(current_state)
        
        # Determine step location to bring COM back to stability
        capture_point = self.estimate_capture_point(current_com, current_state['com_velocity'])
        
        step_location = self.calculate_emergency_step_position(
            current_support_center, capture_point
        )
        
        # Generate step trajectory
        step_trajectory = self.generate_emergency_step_trajectory(
            current_state['swing_foot_position'], 
            step_location
        )
        
        return {
            'step_trajectory': step_trajectory,
            'early_termination': True,
            'adjust_support_polygon': True
        }
    
    def estimate_capture_point(self, com_pos, com_vel):
        """
        Estimate capture point where to step to stop motion
        """
        h = self.params['com_height']
        g = 9.81
        omega = np.sqrt(g / h)
        
        cp_x = com_pos[0] + com_vel[0] / omega
        cp_y = com_pos[1] + com_vel[1] / omega
        
        return np.array([cp_x, cp_y, 0.0])
    
    def calculate_emergency_step_position(self, current_support, desired_capture_point):
        """
        Calculate optimal emergency step position
        """
        # For emergency step, go beyond the capture point to ensure stability
        safety_margin = 0.1  # 10cm beyond capture point
        
        step_x = desired_capture_point[0] + np.sign(desired_capture_point[0] - current_support[0]) * safety_margin
        step_y = desired_capture_point[1] + np.sign(desired_capture_point[1] - current_support[1]) * safety_margin
        
        return np.array([step_x, step_y, 0.0])
```

## Troubleshooting Common Issues

### Issue 1: Walking Instability
- **Symptoms**: Robot falls during walking, oscillating motion
- **Solutions**:
  - Verify COM height estimate accuracy
  - Adjust ZMP tracking gains
  - Increase stance phase duration
  - Check feet friction parameters

### Issue 2: Foot Slipping
- **Symptoms**: Feet slide during stepping or stance phases
- **Solutions**:
  - Increase friction coefficients in simulation
  - Reduce step length and speed
  - Add ankle stiffness control
  - Verify foot geometry and contact points

### Issue 3: Excessive Joint Torques
- **Symptoms**: Unnatural joint movements, potential damage in real robots
- **Solutions**:
  - Limit control gains to reasonable values
  - Add torque saturation
  - Implement smooth torque transitions
  - Verify robot dynamics model accuracy

### Issue 4: ZMP Outside Support Polygon
- **Symptoms**: Unstable walking, frequent falls
- **Solutions**:
  - Verify ZMP calculation implementation
  - Improve COM trajectory planning
  - Increase step frequency during disturbances
  - Implement faster balance recovery strategies

## Best Practices

1. **Gradual Complexity**: Start with simple standing balance before attempting walking
2. **Parameter Tuning**: Use system identification techniques to determine optimal gains
3. **Safety First**: Always implement emergency stop procedures
4. **Simulation-to-Real**: Validate controllers in simulation before real robot testing
5. **Modular Design**: Separate balance control, gait generation, and motor control
6. **Real-time Performance**: Optimize algorithms for real-time execution constraints
7. **Sensor Fusion**: Use multiple sensors for robust state estimation
8. **Robust Control**: Implement controllers that handle parameter uncertainties
9. **Energy Efficiency**: Optimize trajectories for minimal energy consumption
10. **Continuous Monitoring**: Implement comprehensive monitoring and logging

## Summary

Bipedal locomotion and balance control represents one of the most challenging and rewarding areas in humanoid robotics. The complex interplay between mechanical design, control theory, and biological inspiration creates a rich field for innovation and development.

Successfully implementing bipedal walking requires:
- Understanding inverted pendulum dynamics and ZMP theory
- Developing robust balance control strategies
- Creating adaptive gait generation algorithms
- Implementing efficient real-time control systems
- Integrating multiple sensors for state estimation

These capabilities form the foundation for humanoid robots to operate effectively in human environments, bridging digital AI models with physical robotic bodies. The field continues to advance with new control techniques, better mechanical designs, and more sophisticated algorithms, bringing us closer to truly human-like robot locomotion.