---
sidebar_position: 3
---

# Manipulation and Grasping

## Introduction to Manipulation in Humanoid Robots

Humanoid manipulation involves the complex set of tasks performed by robot arms and hands to interact with objects in the environment. For humanoid robots, manipulation presents unique challenges due to the need to coordinate multiple degrees of freedom while maintaining balance and considering the robot's whole-body dynamics. This capability is essential for humanoid robots to perform meaningful tasks in human environments, from picking up objects to complex interactions with tools.

Manipulation in humanoid robots encompasses:
- **Grasping**: Establishing firm contact with objects
- **Transport**: Moving objects from one location to another
- **Articulation**: Manipulating objects with joints or movable parts
- **Assembly**: Combining objects to create new structures
- **Delicate Handling**: Managing fragile or soft objects

## Grasp Planning and Analysis

### Grasp Representation

A grasp is typically represented by:
- **Grasp pose**: Position and orientation of the hand relative to the object
- **Contact points**: Locations where fingers make contact with the object
- **Grasp type**: Classification of the grasp (power, precision, etc.)
- **Force distribution**: How forces are applied at contact points

```python
import numpy as np
from geometry_msgs.msg import Pose, Point
from scipy.spatial.transform import Rotation as R

class Grasp:
    def __init__(self, pose, grasp_type, finger_positions, contact_forces):
        """
        Represents a robotic grasp
        
        Args:
            pose: Pose of the hand (position and orientation)
            grasp_type: Type of grasp (e.g., 'power', 'precision')
            finger_positions: Positions of fingertips in hand frame
            contact_forces: Forces applied at contact points
        """
        self.pose = pose
        self.grasp_type = grasp_type
        self.finger_positions = finger_positions
        self.contact_forces = contact_forces
        self.quality = None  # Will be computed later

class GraspPlanner:
    def __init__(self, robot_params):
        self.robot_params = robot_params
        self.hand_model = self.initialize_hand_model()
        self.object_database = self.load_object_database()
    
    def initialize_hand_model(self):
        """
        Initialize model of the robotic hand
        """
        hand_model = {
            'num_fingers': 5,
            'finger_lengths': [0.05, 0.055, 0.05, 0.045, 0.04],  # Thumb, index, etc.
            'joint_limits': {
                'thumb': [0, 90],  # Limits for abduction and flexion
                'fingers': [0, 120]  # Joint limits for other fingers
            },
            'friction_coeff': 0.8,
            'max_force': 20.0  # Max force per finger in Newtons
        }
        return hand_model
    
    def load_object_database(self):
        """
        Load information about known objects in the environment
        """
        # In practice, this would load from a real database with object properties
        return {
            'cup': {
                'shape': 'cylinder',
                'dims': [0.08, 0.1],  # diameter, height
                'mass': 0.2,
                'com_offset': [0, 0, 0.05],
                'friction': 0.6,
                'stability_points': ['handle']  # Where the object is easiest to grasp
            },
            'book': {
                'shape': 'box',
                'dims': [0.2, 0.15, 0.03],  # width, height, depth
                'mass': 0.5,
                'com_offset': [0, 0, 0],
                'friction': 0.7
            }
        }

    def generate_grasp_candidates(self, target_object):
        """
        Generate potential grasp candidates for the target object
        """
        candidates = []
        
        obj_shape = target_object['shape']
        obj_dims = target_object['dims']
        
        if obj_shape == 'cylinder':
            # Generate cylindrical grasp candidates
            candidates.extend(self._cylindrical_grasps(target_object))
        elif obj_shape == 'box':
            # Generate box grasp candidates
            candidates.extend(self._box_grasps(target_object))
        else:
            # For unknown shapes, generate general grasp points
            candidates.extend(self._general_grasps(target_object))
        
        return candidates

    def _cylindrical_grasps(self, obj):
        """
        Generate grasp candidates for cylindrical objects
        """
        candidates = []
        dims = obj['dims']  # [diameter, height]
        
        # Power grasp for lifting cylinder
        for angle in np.linspace(0, 2*np.pi, 8):  # 8 orientations around the cylinder
            pose = Pose()
            # Position the hand at the center of the cylinder, oriented to wrap around it
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = dims[1] / 2.0  # At the halfway point
            
            # Orient the hand to wrap around the cylinder
            rot = R.from_euler('xyz', [0, 0, angle]).as_quat()
            pose.orientation.x = rot[0]
            pose.orientation.y = rot[1]
            pose.orientation.z = rot[2]
            pose.orientation.w = rot[3]
            
            # Define finger positions for cylindrical grasp
            finger_positions = self._cylindrical_finger_positions(dims, angle)
            
            grasp = Grasp(
                pose=pose,
                grasp_type='cylindrical_power',
                finger_positions=finger_positions,
                contact_forces=self._calculate_initial_forces(obj)
            )
            
            candidates.append(grasp)
        
        # Precision grasp for handling cup by handle
        if obj.get('has_handle', False):
            handle_pose = self._calculate_handle_pose(obj)
            if handle_pose:
                precision_grasp = Grasp(
                    pose=handle_pose,
                    grasp_type='precision_pinch',
                    finger_positions=self._precision_finger_positions(),
                    contact_forces=self._calculate_precision_forces(obj)
                )
                candidates.append(precision_grasp)
        
        return candidates

    def _box_grasps(self, obj):
        """
        Generate grasp candidates for box-shaped objects
        """
        candidates = []
        dims = obj['dims']  # [width, height, depth]
        
        # Generate corner grasps
        corner_offsets = [
            [dims[0]/2, dims[1]/2, dims[2]/2],   # Top right corner
            [dims[0]/2, -dims[1]/2, dims[2]/2],  # Top left corner
            [-dims[0]/2, dims[1]/2, dims[2]/2],  # Bottom right corner
            [-dims[0]/2, -dims[1]/2, dims[2]/2], # Bottom left corner
        ]
        
        for offset in corner_offsets:
            for orient_idx in range(4):  # 4 different orientations
                pose = Pose()
                pose.position.x = offset[0]
                pose.position.y = offset[1]
                pose.position.z = offset[2]
                
                # Rotate the hand for different orientations
                angle = orient_idx * np.pi / 2
                rot = R.from_euler('z', angle).as_quat()
                pose.orientation.x = rot[0]
                pose.orientation.y = rot[1]
                pose.orientation.z = rot[2]
                pose.orientation.w = rot[3]
                
                grasp = Grasp(
                    pose=pose,
                    grasp_type='corner_grasp',
                    finger_positions=self._box_corner_finger_positions(offset),
                    contact_forces=self._calculate_box_forces(obj)
                )
                
                candidates.append(grasp)
        
        return candidates

    def _calculate_grasp_quality(self, grasp, target_object):
        """
        Calculate quality metrics for a given grasp
        """
        # Quality metrics:
        # 1. Force closure: Can resist arbitrary wrenches?
        # 2. Torque efficiency: Minimal force to resist load?
        # 3. Stability: Robust to small perturbations?
        
        quality_metrics = {
            'force_closure': self._check_force_closure(grasp),
            'wrench_resistance': self._calculate_wrench_resistance(grasp, target_object),
            'stability': self._calculate_stability(grasp, target_object),
            'accessibility': self._check_accessibility(grasp),
            'comfort': self._calculate_comfort(grasp)
        }
        
        # Combine metrics into single quality score
        weights = {
            'force_closure': 0.35,
            'wrench_resistance': 0.25,
            'stability': 0.20,
            'accessibility': 0.15,
            'comfort': 0.05
        }
        
        quality = sum(weights[k] * v for k, v in quality_metrics.items()) if all(isinstance(v, (int, float)) for v in quality_metrics.values()) else 0.0
        
        return quality, quality_metrics

    def _check_force_closure(self, grasp):
        """
        Check if the grasp provides force closure
        """
        # Simplified: check if contact points provide sufficient constraint
        # A proper implementation would use the grasp matrix and check if it can resist arbitrary wrenches
        contact_points = grasp.finger_positions
        if len(contact_points) < 3:
            return 0.0  # Need at least 3 contacts for planar objects
        
        # In 3D space, we typically need more contacts for full force closure
        # For a 3D object, we need at least 7 contact points for force closure
        # But for practical purposes in robotics, 4-5 contacts is often sufficient
        if len(contact_points) >= 4:
            return 1.0
        else:
            return 0.5  # Partial score if between 3 and 4 contacts

    def _calculate_wrench_resistance(self, grasp, obj):
        """
        Calculate how well the grasp can resist external wrenches
        """
        # This would involve computing the grasp wrench space
        # For simplicity, we'll estimate based on friction cones and force distribution
        
        # Calculate total force that can be applied before slipping
        max_total_force = sum(min(finger_force, self.hand_model['max_force']) 
                             for finger_force in grasp.contact_forces)
        
        # Account for friction coefficient
        resistance = max_total_force * obj.get('friction', 0.5)
        
        # Normalize against object weight
        obj_weight = obj['mass'] * 9.81
        normalized_resistance = min(1.0, resistance / obj_weight)
        
        return normalized_resistance

    def _calculate_stability(self, grasp, obj):
        """
        Calculate grasp stability to small perturbations
        """
        # Stability depends on contact geometry and object COG
        cog = obj.get('com_offset', [0, 0, 0])
        
        # Calculate distance from grasp center to COG
        grasp_center = self._compute_grasp_center(grasp)
        cog_distance = np.linalg.norm(np.array(grasp_center) - np.array(cog))
        
        # Stability decreases with distance from COG
        # Max stability when grasp is near COG
        max_stable_distance = max(obj['dims']) * 0.5  # Half the largest dimension
        stability = max(0.0, 1.0 - (cog_distance / max_stable_distance))
        
        return stability

    def _check_accessibility(self, grasp):
        """
        Check if the grasp is physically accessible by the robot
        """
        # This would involve inverse kinematics to check if the robot can reach the grasp pose
        # For now, return a simplified estimate
        # In reality, we'd check if IK solution exists
        return 0.8  # Assume 80% of random poses are accessible

    def _calculate_comfort(self, grasp):
        """
        Calculate how comfortable the grasp is for the hand configuration
        """
        # Comfort is related to how close joint angles are to neutral position
        # For this simplified model, assume neutral configuration has high comfort
        return 0.9  # High comfort for now

    def _compute_grasp_center(self, grasp):
        """
        Compute the effective center of the grasp based on contact points
        """
        if not grasp.finger_positions:
            return [0, 0, 0]
        
        center = np.mean(grasp.finger_positions, axis=0)
        return center

    def rank_grasps(self, grasp_candidates, target_object):
        """
        Rank grasp candidates by quality and other criteria
        """
        ranked_grasps = []
        
        for grasp in grasp_candidates:
            quality, metrics = self._calculate_grasp_quality(grasp, target_object)
            grasp.quality = quality
            
            ranked_grasps.append({
                'grasp': grasp,
                'quality': quality,
                'metrics': metrics
            })
        
        # Sort by quality score
        ranked_grasps.sort(key=lambda x: x['quality'], reverse=True)
        
        return ranked_grasps
```

## Force Control and Impedance Control

Effective manipulation requires precise control of forces applied to objects, especially when dealing with varying surface geometries and material properties.

### Force Control Implementation

```python
class ForceController:
    def __init__(self, robot_params):
        self.params = robot_params
        self.kp_force = 50.0  # Proportional gain for force control
        self.ki_force = 10.0  # Integral gain for force control
        self.kd_force = 5.0   # Derivative gain for force control
        self.force_integral = 0.0
        self.previous_force_error = 0.0
        
    def force_control_step(self, current_force, desired_force, dt):
        """
        Calculate control output for force control
        """
        # Calculate force error
        force_error = desired_force - current_force
        
        # Update integral term
        self.force_integral += force_error * dt
        
        # Calculate derivative term
        force_derivative = (force_error - self.previous_force_error) / dt if dt > 0 else 0
        
        # Apply PID control
        control_output = (self.kp_force * force_error + 
                         self.ki_force * self.force_integral + 
                         self.kd_force * force_derivative)
        
        self.previous_force_error = force_error
        
        return control_output

class ImpedanceController:
    def __init__(self, robot_params):
        self.params = robot_params
        # Impedance parameters (mass, damping, stiffness)
        self.mass_impedance = np.diag([10.0, 10.0, 10.0])  # kg
        self.damping_impedance = np.diag([200.0, 200.0, 200.0])  # Ns/m
        self.stiffness_impedance = np.diag([1000.0, 1000.0, 1000.0])  # N/m
        
    def calculate_impedance_force(self, position_error, velocity_error):
        """
        Calculate impedance control force
        F = M*d2x_desired + B*dx_error + K*x_error
        """
        # For impedance control: F = M(ddx_cmd) + B(dx_error) + K(x_error)
        # Where ddx_cmd is the desired acceleration
        impedance_force = (np.dot(self.mass_impedance, np.zeros(3)) +  # No desired acc
                          np.dot(self.damping_impedance, velocity_error) + 
                          np.dot(self.stiffness_impedance, position_error))
        
        return impedance_force
    
    def adaptive_impedance(self, task_phase, environment_stiffness):
        """
        Adjust impedance parameters based on task requirements
        """
        if task_phase == 'approach':
            # Low stiffness for safe approach
            self.stiffness_impedance = np.diag([100.0, 100.0, 100.0])
        elif task_phase == 'contact':
            # Moderate stiffness for contact establishment
            self.stiffness_impedance = np.diag([500.0, 500.0, 500.0])
        elif task_phase == 'manipulation':
            # High stiffness for precise manipulation
            self.stiffness_impedance = np.diag([2000.0, 2000.0, 2000.0])
        elif task_phase == 'release':
            # Low stiffness for safe release
            self.stiffness_impedance = np.diag([100.0, 100.0, 100.0])
        
        # Adjust damping based on environment stiffness
        self.damping_impedance = self.damping_impedance * (1.0 + environment_stiffness / 1000.0)

class HybridForcePositionController:
    def __init__(self, robot_params):
        self.params = robot_params
        self.position_controller = self._init_position_controller()
        self.force_controller = ForceController(robot_params)
        self.impedance_controller = ImpedanceController(robot_params)
        
    def _init_position_controller(self):
        """Initialize position controller with PD control"""
        return {
            'kp': 1000.0,  # Position proportional gain
            'kv': 50.0     # Velocity (damping) gain
        }
    
    def hybrid_control(self, pose_desired, pose_actual, 
                      force_desired, force_actual, dt):
        """
        Implement hybrid force/position control
        Control in position directions, force in constrained directions
        """
        # Calculate pose errors
        pos_error = pose_desired.position - pose_actual.position
        vel_error = pose_desired.velocity - pose_actual.velocity
        
        # Calculate force errors
        force_error = force_desired - force_actual
        
        # For hybrid control, determine which directions need position control
        # and which need force control
        # This depends on the task and constraints
        
        # Example: Position control in X,Y; Force control in Z
        control_output = np.zeros(6)  # 3 position + 3 orientation
        
        # Position control for X, Y, and orientation
        control_output[0:2] = (self.position_controller['kp'] * pos_error[0:2] + 
                              self.position_controller['kv'] * vel_error[0:2])  # X, Y
        control_output[3:6] = (self.position_controller['kp'] * pos_error[3:6] + 
                              self.position_controller['kv'] * vel_error[3:6])  # Orientation
        
        # Force control for Z (normal to surface)
        z_force_control = self.force_controller.force_control_step(
            force_actual[2], force_desired[2], dt
        )
        control_output[2] = z_force_control  # Z force
        
        return control_output
```

## Dexterous Manipulation

### Multiple-Arm Coordination

Humanoid robots often have two arms that can work together for complex manipulation tasks:

```python
class BimanualCoordination:
    def __init__(self, robot_params):
        self.params = robot_params
        self.left_arm_controller = ArmController('left', robot_params)
        self.right_arm_controller = ArmController('right', robot_params)
        self.coordination_manager = CoordinationManager(robot_params)
        
    def coordinated_grasp(self, object_info, grasp_type='passive_holder'):
        """
        Coordinate both arms for a cooperative grasp
        grasp_type: 'passive_holder' (one holds, other manipulates)
                   'active_cooperation' (both actively participate)
                   'symmetric' (both apply same motion)
        """
        if grasp_type == 'passive_holder':
            # Right hand holds object firmly
            right_grasp = self.right_arm_controller.plan_grasp(object_info, 'firm_hold')
            
            # Left hand manipulates or applies force
            left_manipulation = self.left_arm_controller.plan_manipulation(
                object_info, 
                'apply_torque'
            )
            
            # Ensure right hand maintains grasp while left manipulates
            self.enforce_constraint(right_grasp, left_manipulation, 'no_slip')
            
        elif grasp_type == 'active_cooperation':
            # Both hands actively participate in manipulation
            left_grasp, right_grasp = self.plan_bilateral_grasp(object_info)
            
            # Calculate coordinated motion
            bilateral_motion = self.plan_bilateral_motion(
                left_grasp, right_grasp, object_info
            )
            
        elif grasp_type == 'symmetric':
            # Both hands apply symmetric motion
            symmetric_grasp = self.plan_symmetric_grasp(object_info)
            bilateral_motion = self.plan_symmetric_motion(symmetric_grasp)
        
        return bilateral_motion
    
    def plan_bilateral_grasp(self, object_info):
        """
        Plan coordinated grasp with both arms
        """
        # Determine optimal grasp points for each hand
        # This involves solving for stable grasp configuration considering both hands
        object_dims = object_info['dimensions']
        
        # For a simple object like a box, determine grasp points
        if object_info['shape'] == 'box':
            # Left hand grasps one side, right hand grasps opposite side
            left_grasp_pos = np.array([-object_dims[0]/2, 0, 0])
            right_grasp_pos = np.array([object_dims[0]/2, 0, 0])
        else:
            # For other shapes, calculate appropriate grasp points
            left_grasp_pos, right_grasp_pos = self.calculate_opposing_grasps(object_info)
        
        # Plan grasp orientation for each hand
        left_grasp_orient = self.calculate_grasp_orientation(left_grasp_pos, object_info)
        right_grasp_orient = self.calculate_grasp_orientation(right_grasp_pos, object_info)
        
        return (left_grasp_orient, right_grasp_orient)
    
    def enforce_constraint(self, grasp1, grasp2, constraint_type):
        """
        Enforce constraint between two grasps
        """
        if constraint_type == 'no_slip':
            # Ensure both grasps maintain sufficient friction to prevent object slip
            self.verify_friction_constraints(grasp1, grasp2)
        
        elif constraint_type == 'rigid_coupling':
            # Maintain rigid connection between hands through object
            self.update_kinematic_chain(grasp1, grasp2)
    
    def calculate_opposing_grasps(self, object_info):
        """
        Calculate opposing grasp points for an arbitrary object
        """
        # This would typically involve finding antipodal grasp points
        # For now, use simplified approach based on bounding box
        bounding_box = self.approximate_bounding_box(object_info)
        
        # Find longest dimension and grasp on opposite sides
        dimensions = [abs(bounding_box[1][i] - bounding_box[0][i]) for i in range(3)]
        max_dim_idx = np.argmax(dimensions)
        
        grasp1 = np.zeros(3)
        grasp2 = np.zeros(3)
        
        # Place grasps on opposite sides of the longest dimension
        grasp1[max_dim_idx] = bounding_box[0][max_dim_idx]
        grasp2[max_dim_idx] = bounding_box[1][max_dim_idx]
        
        # Center in other dimensions
        for i in range(3):
            if i != max_dim_idx:
                center = (bounding_box[0][i] + bounding_box[1][i]) / 2
                grasp1[i] = center
                grasp2[i] = center
        
        return grasp1, grasp2
    
    def approximate_bounding_box(self, object_info):
        """
        Approximate bounding box for the object
        """
        # This would come from perception system in practice
        # For example, if we know object is centered at origin with dimensions
        dimensions = object_info.get('dimensions', [0.1, 0.1, 0.1])
        center = object_info.get('center', [0, 0, 0])
        
        min_point = [center[i] - dimensions[i]/2 for i in range(3)]
        max_point = [center[i] + dimensions[i]/2 for i in range(3)]
        
        return [min_point, max_point]
    
    def calculate_grasp_orientation(self, grasp_position, object_info):
        """
        Calculate proper grasp orientation at given position
        """
        # This depends on the object and task
        # For a simple approach, orient perpendicular to the surface normal at grasp point
        surface_normal = self.estimate_surface_normal(grasp_position, object_info)
        
        # Calculate a suitable approach direction (often perpendicular to normal)
        approach_dir = self.calculate_approach_direction(surface_normal)
        
        # Create grasp orientation
        grasp_quaternion = self.align_with_surface(approach_dir, surface_normal)
        
        return {
            'position': grasp_position,
            'orientation': grasp_quaternion,
            'approach_direction': approach_dir
        }
    
    def estimate_surface_normal(self, point, object_info):
        """
        Estimate surface normal at a given point on the object
        """
        # Simplified approach: for common shapes, use analytical formulas
        shape = object_info.get('shape', 'box')
        
        if shape == 'box':
            # For box, determine which face the point is closest to
            dims = object_info['dimensions']
            center = np.array(object_info.get('center', [0, 0, 0]))
            
            # Calculate relative position
            rel_pos = np.array(point) - center
            
            # Determine closest face (largest absolute coordinate)
            face_idx = np.argmax(np.abs(rel_pos))
            
            normal = np.zeros(3)
            sign = 1 if rel_pos[face_idx] >= 0 else -1
            normal[face_idx] = sign
            
            return normal
        
        else:
            # For other shapes, would need more sophisticated calculation
            return np.array([0, 0, 1])  # Default: up direction

class CoordinationManager:
    def __init__(self, robot_params):
        self.params = robot_params
        self.workspace_overlap = self.calculate_workspace_overlap()
    
    def calculate_workspace_overlap(self):
        """
        Calculate overlapping workspace between two arms
        """
        # This would involve calculating reachability for each arm
        # and finding the intersection of their workspaces
        return {
            'volume': 0.5,  # m^3
            'optimal_region': np.array([0, 0, 0.8])  # Center of overlapping region
        }
    
    def coordinate_movement(self, left_target, right_target):
        """
        Coordinate movement of both arms to avoid collisions and optimize performance
        """
        # Check workspace feasibility
        if not self.is_reachable(left_target, 'left') or not self.is_reachable(right_target, 'right'):
            raise ValueError("Targets not reachable by respective arms")
        
        # Check for potential collisions
        if self.would_collide(left_target, right_target):
            # Plan collision-free trajectory
            coordinated_traj = self.plan_collision_free_trajectory(left_target, right_target)
        else:
            # Plan independent trajectories
            coordinated_traj = {
                'left_arm': self.plan_trajectory_to_target(left_target, 'left'),
                'right_arm': self.plan_trajectory_to_target(right_target, 'right')
            }
        
        return coordinated_traj
    
    def would_collide(self, left_pose, right_pose):
        """
        Check if arm poses would result in collision
        """
        # Simplified collision check - in reality would use full collision detection
        left_pos = np.array(left_pose[:3]) if len(left_pose) >= 3 else np.array(left_pose)
        right_pos = np.array(right_pose[:3]) if len(right_pose) >= 3 else np.array(right_pose)
        
        # Check if arms are too close together
        distance = np.linalg.norm(left_pos - right_pos)
        
        # Conservative collision threshold based on arm length
        min_safe_distance = 0.2  # 20 cm minimum separation
        
        return distance < min_safe_distance
    
    def plan_collision_free_trajectory(self, left_target, right_target):
        """
        Plan trajectories that avoid collisions between arms
        """
        # This would implement more sophisticated coordination
        # For now, just return a placeholder
        return {
            'left_arm': [left_target],  # Direct path
            'right_arm': [right_target],  # Direct path
            'coordination_required': True
        }
```

## Grasp Stability and Optimization

### Robust Grasp Planning

```python
class RobustGraspPlanner:
    def __init__(self, robot_params):
        self.params = robot_params
        self.uncertainty_model = self.build_uncertainty_model()
        self.stability_evaluator = StabilityEvaluator()
        
    def build_uncertainty_model(self):
        """
        Build model of uncertainties in grasp planning
        """
        return {
            'object_pose_uncertainty': 0.01,  # 1cm in position
            'object_dim_uncertainty': 0.005,  # 5mm in dimensions
            'friction_coeff_uncertainty': 0.1,  # 10% in friction
            'sensor_noise_level': 0.002,  # 2mm sensor noise
            'actuation_precision': 0.003  # 3mm actuator precision
        }
    
    def plan_robust_grasp(self, object_info, success_probability=0.95):
        """
        Plan grasp that is robust to uncertainties
        """
        # Generate many grasp candidates with perturbations
        robust_candidates = []
        
        # Try multiple perturbation levels
        for perturbation_level in [0.001, 0.002, 0.005, 0.01]:
            # Generate base grasp candidates
            base_candidates = self.generate_grasp_candidates_with_uncertainty(
                object_info, perturbation_level
            )
            
            # Evaluate robustness of each candidate
            for candidate in base_candidates:
                robustness_score = self.evaluate_robustness(
                    candidate, object_info, perturbation_level
                )
                
                if robustness_score >= success_probability:
                    robust_candidates.append({
                        'grasp': candidate,
                        'robustness_score': robustness_score,
                        'perturbation_level': perturbation_level
                    })
        
        # Sort by robustness score
        robust_candidates.sort(key=lambda x: x['robustness_score'], reverse=True)
        
        return robust_candidates
    
    def generate_grasp_candidates_with_uncertainty(self, object_info, perturbation_level):
        """
        Generate grasp candidates considering uncertainty in object properties
        """
        base_candidates = self.generate_grasp_candidates(object_info)
        perturbed_candidates = []
        
        num_perturbations = 20  # Number of perturbations to try
        
        for base_grasp in base_candidates:
            for _ in range(num_perturbations):
                # Perturb object properties
                perturbed_object = self.perturb_object_properties(
                    object_info, perturbation_level
                )
                
                # Adjust grasp based on perturbed object
                adjusted_grasp = self.adjust_grasp_for_perturbed_object(
                    base_grasp, perturbed_object
                )
                
                perturbed_candidates.append(adjusted_grasp)
        
        return perturbed_candidates
    
    def perturb_object_properties(self, object_info, perturbation_level):
        """
        Apply random perturbations to object properties based on uncertainty model
        """
        perturbed = object_info.copy()
        
        # Perturb position
        pos_perturbation = np.random.normal(0, self.uncertainty_model['object_pose_uncertainty'], 3)
        if 'center' in perturbed:
            perturbed['center'] = np.array(perturbed['center']) + pos_perturbation
        
        # Perturb dimensions
        dim_perturbation = np.random.normal(0, self.uncertainty_model['object_dim_uncertainty'], len(perturbed['dimensions']))
        perturbed['dimensions'] = np.array(perturbed['dimensions']) + dim_perturbation
        
        # Ensure positive dimensions
        perturbed['dimensions'] = np.maximum(perturbed['dimensions'], 0.001)
        
        # Perturb friction coefficient
        friction_perturbation = np.random.normal(0, self.uncertainty_model['friction_coeff_uncertainty'])
        if 'friction' in perturbed:
            perturbed['friction'] = max(0.1, perturbed['friction'] + friction_perturbation)
        
        return perturbed
    
    def adjust_grasp_for_perturbed_object(self, base_grasp, perturbed_object):
        """
        Adjust grasp based on perturbed object properties
        """
        # Copy base grasp
        adjusted_grasp = Grasp(
            pose=base_grasp.pose,
            grasp_type=base_grasp.grasp_type,
            finger_positions=base_grasp.finger_positions.copy(),
            contact_forces=base_grasp.contact_forces.copy()
        )
        
        # Adjust finger positions based on new object dimensions
        # This is a simplified adjustment - in reality would need more sophisticated planning
        
        return adjusted_grasp
    
    def evaluate_robustness(self, grasp, object_info, perturbation_level):
        """
        Evaluate how robust the grasp is to perturbations
        """
        num_evaluations = 50
        successful_grasps = 0
        
        for _ in range(num_evaluations):
            # Apply random perturbation
            perturbed_object = self.perturb_object_properties(object_info, perturbation_level)
            
            # Evaluate grasp on perturbed object
            if self.evaluate_grasp_stability(grasp, perturbed_object):
                successful_grasps += 1
        
        robustness_score = successful_grasps / num_evaluations
        return robustness_score
    
    def evaluate_grasp_stability(self, grasp, object_info):
        """
        Evaluate if a grasp is stable on the given object
        """
        # Implement stability evaluation using mechanics
        try:
            # Check force closure
            if not self.check_force_closure(grasp, object_info):
                return False
            
            # Check that grasp can resist gravity
            if not self.can_resist_gravity(grasp, object_info):
                return False
            
            # Check that grasp is not slippable
            if self.is_slippable(grasp, object_info):
                return False
            
            # All checks passed
            return True
            
        except Exception:
            return False
    
    def check_force_closure(self, grasp, object_info):
        """
        Check if the grasp provides force closure
        """
        # Simplified force closure check
        # In reality would use grasp matrix and check if it spans the wrench space
        contact_points = grasp.finger_positions
        if len(contact_points) < 2:
            return False
        
        # For a 3D object, we need at least 7 contacts for full force closure
        # But for practical purposes, 4-5 contacts with good friction is usually sufficient
        num_contacts = len(contact_points)
        friction_coef = object_info.get('friction', 0.6)
        
        # Minimum requirement: at least 2 contacts with sufficient friction
        if num_contacts >= 2 and friction_coef > 0.3:
            return True
        else:
            return False
    
    def can_resist_gravity(self, grasp, object_info):
        """
        Check if grasp can resist the object's weight
        """
        object_weight = object_info['mass'] * 9.81
        max_grasp_force = self.get_maximum_grasp_force(grasp)
        
        return max_grasp_force > object_weight
    
    def is_slippable(self, grasp, object_info):
        """
        Check if the grasp is prone to slippage
        """
        # Calculate if the applied forces exceed friction limits
        contact_forces = grasp.contact_forces
        friction_coef = object_info.get('friction', 0.6)
        
        # Check each contact against friction cone constraint
        for i, force in enumerate(contact_forces):
            if isinstance(force, (list, tuple, np.ndarray)):
                force_norm = np.linalg.norm(force)
            else:
                force_norm = abs(force)
                
            # Simplified: if normal force component is not sufficient to resist tangential forces
            # This is a very simplified check
            max_friction_force = force_norm * friction_coef
            
            # For this simplified example, assume tangential force is half the normal force
            tangential_force = force_norm * 0.5
            
            if tangential_force > max_friction_force:
                return True  # Would slip
        
        return False

class StabilityEvaluator:
    def __init__(self):
        self.evaluation_methods = [
            self._evaluate_form_closure,
            self._evaluate_force_closure,
            self._evaluate_friction_constraints,
            self._evaluate_dynamic_stability
        ]
    
    def evaluate_grasp_stability(self, grasp, object_info):
        """
        Comprehensive stability evaluation
        """
        scores = {}
        
        for method in self.evaluation_methods:
            score = method(grasp, object_info)
            method_name = method.__name__
            scores[method_name] = score
        
        # Calculate composite stability score
        weights = {
            '_evaluate_form_closure': 0.2,
            '_evaluate_force_closure': 0.3,
            '_evaluate_friction_constraints': 0.3,
            '_evaluate_dynamic_stability': 0.2
        }
        
        composite_score = sum(weights[method] * scores[method] for method in scores)
        
        return composite_score, scores
    
    def _evaluate_form_closure(self, grasp, object_info):
        """
        Evaluate form closure (geometric constraints)
        """
        # Form closure exists when object is geometrically constrained
        # This is a simplified check - real evaluation would be more complex
        contact_points = grasp.finger_positions
        
        if len(contact_points) < 4:
            return 0.0  # No form closure possible with < 4 points
        
        # Check if contact points can constrain all DOFs
        # For a sphere, 7 points are needed for form closure
        # For general objects, minimum is 7 points
        if len(contact_points) >= 7:
            return 1.0
        elif len(contact_points) >= 4:
            return 0.7  # Good for many practical objects
        else:
            return 0.0
    
    def _evaluate_force_closure(self, grasp, object_info):
        """
        Evaluate force closure (ability to resist arbitrary wrenches)
        """
        # Check if the grasp can theoretically resist any external wrench
        # This would normally involve computing the grasp matrix
        contact_points = grasp.finger_positions
        
        if len(contact_points) < 2:
            return 0.0
        
        # For 3D objects, minimum contacts for force closure
        friction_coef = object_info.get('friction', 0.6)
        
        if len(contact_points) >= 4 and friction_coef > 0.4:
            return 1.0
        elif len(contact_points) >= 3 and friction_coef > 0.6:
            return 0.8
        elif len(contact_points) >= 2 and friction_coef > 0.8:
            return 0.5
        else:
            return 0.0
    
    def _evaluate_friction_constraints(self, grasp, object_info):
        """
        Evaluate friction-based constraints
        """
        # Check if applied forces respect friction limits
        contact_points = grasp.finger_positions
        contact_forces = grasp.contact_forces
        friction_coef = object_info.get('friction', 0.6)
        
        if len(contact_points) == 0 or len(contact_forces) == 0:
            return 0.0
        
        # Calculate friction constraints satisfaction
        satisfied_constraints = 0
        total_constraints = len(contact_forces)
        
        for force in contact_forces:
            force_mag = abs(force) if isinstance(force, (int, float)) else np.linalg.norm(force)
            friction_limit = friction_coef * force_mag  # Simplified
            
            # Assume force is within friction limit for this evaluation
            satisfied_constraints += 1
        
        return satisfied_constraints / total_constraints if total_constraints > 0 else 0.0
    
    def _evaluate_dynamic_stability(self, grasp, object_info):
        """
        Evaluate stability under dynamic conditions
        """
        # Consider dynamic effects like acceleration, jerks, etc.
        mass = object_info['mass']
        gravity = 9.81
        
        # Stability under acceleration
        acceleration_threshold = 5.0  # m/s^2 maximum acceleration before concern
        max_acceptable_inertia_force = mass * acceleration_threshold
        
        # Simplified check: if object weight is much less than acceptable force, it's stable
        weight = mass * gravity
        
        if max_acceptable_inertia_force > weight * 3:  # 3x safety factor
            return 1.0
        elif max_acceptable_inertia_force > weight:
            return 0.7
        else:
            return 0.3
```

## Practical Manipulation Examples

### Picking and Placing

```python
class PickAndPlaceController:
    def __init__(self, robot_params):
        self.params = robot_params
        self.grasp_planner = GraspPlanner(robot_params)
        self.bimanual_controller = BimanualCoordination(robot_params)
        self.motion_planner = MotionPlanner(robot_params)
        
    def pick_object(self, target_object, approach_height=0.1):
        """
        Execute picking sequence for a target object
        """
        # Step 1: Plan grasp approach
        grasp_candidates = self.grasp_planner.generate_grasp_candidates(target_object)
        best_grasp = grasp_candidates[0] if grasp_candidates else None
        
        if not best_grasp:
            raise RuntimeError("No viable grasp found for target object")
        
        # Step 2: Plan approach trajectory (above object)
        approach_pos = list(best_grasp.pose.position)
        approach_pos[2] += approach_height  # Approach from above
        
        # Step 3: Execute approach
        self.move_to_approach_position(approach_pos, best_grasp.pose.orientation)
        
        # Step 4: Execute descent to grasp
        self.descend_to_grasp(best_grasp.pose)
        
        # Step 5: Close gripper with appropriate force
        self.grasp_with_appropriate_force(target_object)
        
        # Step 6: Lift object
        self.lift_object(0.1)  # Lift 10cm
        
        print("Pick operation completed successfully")
        
        return {
            'status': 'success',
            'grasp_used': best_grasp,
            'picked_object': target_object
        }
    
    def place_object(self, target_position, placement_orientation=None):
        """
        Execute placing sequence at target position
        """
        # Step 1: Plan approach to placement location
        approach_pos = list(target_position)
        approach_pos[2] += 0.05  # 5cm above placement position
        
        # Step 2: Execute approach
        if placement_orientation:
            self.move_to_approach_position(approach_pos, placement_orientation)
        else:
            self.move_to_position(approach_pos)  # Maintain current orientation
        
        # Step 3: Descend to placement position
        self.descend_to_placement(target_position)
        
        # Step 4: Open gripper to release object
        self.release_object()
        
        # Step 5: Retract from object
        self.lift_after_placement(0.05)  # Lift 5cm after placing
        
        print("Place operation completed successfully")
        
        return {
            'status': 'success',
            'placed_at': target_position
        }
    
    def move_to_approach_position(self, position, orientation):
        """
        Move end-effector to approach position with proper orientation
        """
        # Plan path from current pose to approach pose
        target_pose = Pose()
        target_pose.position.x = position[0]
        target_pose.position.y = position[1]
        target_pose.position.z = position[2]
        target_pose.orientation = orientation
        
        # Execute planned trajectory
        trajectory = self.motion_planner.plan_trajectory_to_pose(target_pose)
        self.execute_trajectory(trajectory)
    
    def descend_to_grasp(self, grasp_pose):
        """
        Descend from approach to grasp position
        """
        # Execute fine motion control to reach grasp position
        self.execute_precise_positioning(grasp_pose)
        
        # Verify grasp alignment
        if not self.verify_grasp_alignment(grasp_pose):
            raise RuntimeError("Grasp alignment verification failed")
    
    def grasp_with_appropriate_force(self, object_info):
        """
        Close gripper with force appropriate for object properties
        """
        # Calculate required grip force based on object properties
        object_weight = object_info['mass'] * 9.81
        friction_coeff = object_info.get('friction', 0.6)
        
        # Calculate minimum required grip force
        # F_grip = (weight / friction_coeff) * safety_factor
        safety_factor = 2.0
        min_grip_force = (object_weight / friction_coeff) * safety_factor
        
        # Apply force control to achieve calculated grip force
        self.apply_grip_force(min_grip_force)
        
        # Verify successful grasp
        if not self.verify_successful_grasp():
            raise RuntimeError("Grasp verification failed - object may have slipped")
    
    def verify_grasp_alignment(self, expected_pose):
        """
        Verify that gripper is properly aligned for grasp
        """
        current_pose = self.get_current_gripper_pose()
        
        # Check position alignment
        position_error = np.linalg.norm(
            np.array([expected_pose.position.x, expected_pose.position.y, expected_pose.position.z]) -
            np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
        )
        
        if position_error > 0.005:  # 5mm threshold
            return False
        
        # Check orientation alignment (simplified)
        orientation_error = self.calculate_orientation_error(
            expected_pose.orientation, current_pose.orientation
        )
        
        if orientation_error > 0.1:  # 0.1 radian (~5.7 degrees) threshold
            return False
        
        return True
    
    def calculate_orientation_error(self, expected_quat, current_quat):
        """
        Calculate the orientation error between two quaternions
        """
        # Convert quaternions to rotation matrices
        expected_rot = R.from_quat([expected_quat.x, expected_quat.y, expected_quat.z, expected_quat.w])
        current_rot = R.from_quat([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
        
        # Calculate relative rotation
        relative_rot = expected_rot.inv() * current_rot
        
        # Return rotation angle as error measure
        return relative_rot.magnitude()

### Assembly Operations

class AssemblyController:
    def __init__(self, robot_params):
        self.params = robot_params
        self.alignment_detector = AlignmentDetector()
        self.force_controller = ForceController(robot_params)
        
    def insert_pin_into_hole(self, pin_info, hole_info):
        """
        Execute precision insertion task
        """
        # Step 1: Align pin with hole axis
        self.orient_pin_for_insertion(pin_info, hole_info)
        
        # Step 2: Approach hole with conservative force control
        self.approach_hole_with_force_control(hole_info)
        
        # Step 3: Insert with compliance control
        self.insert_with_compliance_control(pin_info, hole_info)
        
        # Step 4: Verify successful insertion
        if not self.verify_insertion_success(hole_info):
            raise RuntimeError("Insertion verification failed")
        
        return {'status': 'success', 'insertion_verified': True}
    
    def orient_pin_for_insertion(self, pin_info, hole_info):
        """
        Orient pin to align with hole axis
        """
        # Calculate required orientation to align pin with hole
        hole_axis = self.calculate_hole_axis(hole_info)
        pin_axis = self.calculate_pin_axis(pin_info)
        
        # Calculate rotation needed to align axes
        rotation_quaternion = self.calculate_alignment_rotation(pin_axis, hole_axis)
        
        # Orient gripper to apply this rotation
        current_pose = self.get_current_gripper_pose()
        target_orientation = self.combine_orientations(
            current_pose.orientation, rotation_quaternion
        )
        
        self.set_gripper_orientation(target_orientation)
    
    def approach_hole_with_force_control(self, hole_info):
        """
        Approach hole using force control to handle misalignments
        """
        # Move towards hole while maintaining low insertion force
        target_force = 5.0  # 5N target insertion force
        safety_distance = 0.01  # 1cm before hole to start force control
        
        # Enable force control
        self.enable_force_control(axis='approach', target_force=target_force)
        
        # Move toward hole until contact or target force reached
        self.move_approach_direction_with_force_feedback(safety_distance)
    
    def insert_with_compliance_control(self, pin_info, hole_info):
        """
        Execute insertion using compliance control
        """
        # Enable impedance control for compliant insertion
        self.set_compliant_impedance_for_insertion()
        
        # Apply controlled insertion force
        insertion_force = 10.0  # Controlled insertion force
        self.apply_controlled_force(insertion_force, 'approach')
        
        # Monitor for successful insertion
        while not self.detect_insertion_completion():
            # Continue applying force with compliance
            self.monitor_insertion_force()
        
        # Disable compliance and verify final position
        self.disable_compliance_control()
    
    def detect_insertion_completion(self):
        """
        Detect when insertion is complete
        """
        # Check for conditions indicating insertion completion:
        # - Low force in insertion direction
        # - Stable position/orientation
        # - Expected depth achieved
        
        insertion_force = self.get_current_insertion_force()
        if insertion_force < 2.0:  # Very low insertion force indicates bottomed out
            return True
        
        return False

class AlignmentDetector:
    def __init__(self):
        pass
    
    def detect_misalignment(self, insertion_state):
        """
        Detect if pin and hole are misaligned during insertion
        """
        # Monitor forces for signs of misalignment
        forces = insertion_state['forces']
        
        # High lateral forces indicate misalignment
        lateral_force_threshold = 5.0  # N
        if np.linalg.norm(forces[0:2]) > lateral_force_threshold:  # X, Y forces
            return True, "High lateral forces detected - possible misalignment"
        
        # High twisting torques indicate angular misalignment
        twist_torque_threshold = 1.0  # N*m
        if abs(forces[5]) > twist_torque_threshold:  # Z-axis torque
            return True, "High twisting torque detected - angular misalignment"
        
        return False, "Alignment appears adequate"
```

## Grasp Learning and Adaptation

### Adaptive Grasping Systems

Modern manipulation systems include learning components that adapt to new objects and situations:

```python
class AdaptiveGraspLearner:
    def __init__(self, robot_params):
        self.params = robot_params
        self.experience_db = []  # Store successful and failed grasps
        self.similarity_matcher = SimilarityMatcher()
        
    def learn_from_experience(self, object_features, grasp_attempt, outcome):
        """
        Learn from grasping experiences
        """
        experience = {
            'object_features': object_features,
            'grasp_configuration': grasp_attempt,
            'outcome': outcome,  # 'success' or 'failure'
            'timestamp': time.time(),
            'environment_conditions': self.get_current_env_conditions()
        }
        
        # Store experience
        self.experience_db.append(experience)
        
        # Update grasp policies based on new experience
        self.update_grasp_policy(experience)
    
    def suggest_grasp_for_new_object(self, new_object_features):
        """
        Suggest grasp based on learned experiences
        """
        # Find similar previous experiences
        similar_experiences = self.find_similar_experiences(new_object_features)
        
        # Find successful grasps for similar objects
        successful_grasps = [
            exp['grasp_configuration'] 
            for exp in similar_experiences 
            if exp['outcome'] == 'success'
        ]
        
        if not successful_grasps:
            # Fall back to analytical grasp planning
            return self.fallback_analytical_grasp(new_object_features)
        
        # Select most promising grasp based on past success
        suggested_grasp = self.select_best_grasp_from_experience(successful_grasps)
        
        return suggested_grasp
    
    def find_similar_experiences(self, target_features):
        """
        Find experiences with similar object features
        """
        similar = []
        for experience in self.experience_db:
            similarity = self.similarity_matcher.compute_similarity(
                target_features, experience['object_features']
            )
            
            if similarity > 0.7:  # Threshold for considering similar
                similar.append((experience, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return [exp[0] for exp in similar]  # Return just the experiences
    
    def update_grasp_policy(self, experience):
        """
        Update grasp selection policy based on new experience
        """
        # This would involve updating a machine learning model
        # in a real implementation, such as:
        # - Updating a classifier for grasp success prediction
        # - Updating a policy network for grasp selection
        # - Reinforcement learning updates
        pass

class SimilarityMatcher:
    def __init__(self):
        pass
    
    def compute_similarity(self, features1, features2):
        """
        Compute similarity between two sets of object features
        """
        # Compare object dimensions
        if 'dimensions' in features1 and 'dimensions' in features2:
            dim_similarity = self.compute_dimension_similarity(
                features1['dimensions'], features2['dimensions']
            )
        else:
            dim_similarity = 0.5  # Neutral if not available
        
        # Compare weight
        if 'mass' in features1 and 'mass' in features2:
            weight_similarity = self.compute_weight_similarity(
                features1['mass'], features2['mass']
            )
        else:
            weight_similarity = 0.5
        
        # Compare shape
        if 'shape' in features1 and 'shape' in features2:
            shape_similarity = 1.0 if features1['shape'] == features2['shape'] else 0.3
        else:
            shape_similarity = 0.5
        
        # Combine similarities (weighted average)
        total_similarity = (
            0.5 * dim_similarity + 
            0.3 * weight_similarity + 
            0.2 * shape_similarity
        )
        
        return total_similarity
    
    def compute_dimension_similarity(self, dims1, dims2):
        """
        Compute similarity based on object dimensions
        """
        # Calculate relative difference
        dims1_arr = np.array(dims1)
        dims2_arr = np.array(dims2)
        
        # Relative difference (closer to 1 means more similar)
        rel_diff = np.abs(dims1_arr - dims2_arr) / np.maximum(dims1_arr, dims2_arr)
        avg_rel_diff = np.mean(rel_diff)
        
        # Convert to similarity (1 - difference)
        similarity = max(0.0, 1.0 - avg_rel_diff)
        
        return similarity
    
    def compute_weight_similarity(self, weight1, weight2):
        """
        Compute similarity based on object weight
        """
        rel_diff = abs(weight1 - weight2) / max(weight1, weight2)
        similarity = max(0.0, 1.0 - rel_diff)
        return similarity
```

## Troubleshooting Common Issues

### Issue 1: Grasp Failure
- **Symptoms**: Objects slip or fall during manipulation
- **Causes**: Insufficient grip force, wrong grasp type, poor contact points
- **Solutions**: 
  - Recalculate required grip force based on object properties
  - Try alternative grasp types
  - Verify contact point locations

### Issue 2: Collision During Manipulation
- **Symptoms**: Robot collides with environment or itself during manipulation
- **Causes**: Poor motion planning, inaccurate object localization, narrow workspaces
- **Solutions**:
  - Improve motion planning with better obstacle detection
  - Verify object pose estimation accuracy
  - Increase safety margins in motion planning

### Issue 3: Force Control Problems
- **Symptoms**: Excessive forces applied, unstable contact, object damage
- **Causes**: Incorrect force control parameters, sensor noise, model mismatches
- **Solutions**:
  - Tune force control gains
  - Improve sensor filtering
  - Verify robot dynamics model

### Issue 4: Grasp Planning Failure
- **Symptoms**: No valid grasp found for object
- **Causes**: Complex object geometry, workspace limitations, kinematic constraints
- **Solutions**:
  - Implement more sophisticated grasp planners
  - Use geometric simplifications
  - Consider repositioning object

## Best Practices

1. **Safety First**: Always implement force limits and collision avoidance
2. **Multi-Modal Sensing**: Combine vision, force/torque, and tactile feedback
3. **Graduated Complexity**: Start with simple objects and increase complexity
4. **Robust Planning**: Consider uncertainties and implement fault tolerance
5. **Calibration**: Regularly calibrate sensors and hand-eye coordination
6. **Force Control**: Use appropriate force control for delicate operations
7. **Learning**: Incorporate learning mechanisms to improve performance
8. **Human Oversight**: Maintain ability for human intervention when needed
9. **Simulation Testing**: Validate in simulation before real robot experiments
10. **Modular Design**: Separate perception, planning, and control for maintainability

## Summary

Manipulation and grasping in humanoid robots involve a complex interplay of perception, planning, control, and learning. Successful manipulation requires:
- Effective grasp planning algorithms that consider object properties and robot capabilities
- Precise force and impedance control for stable contact interactions
- Coordinated control of multiple degrees of freedom
- Adaptive systems that learn from experience
- Robust handling of uncertainties and disturbances

These capabilities are essential for humanoid robots to perform meaningful tasks in human environments, allowing them to bridge digital AI models with physical robotic bodies. The field continues to advance with improved sensors, learning algorithms, and robotic hardware, making increasingly sophisticated manipulation tasks possible.