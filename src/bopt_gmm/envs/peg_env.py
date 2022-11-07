import numpy as np

from functools   import lru_cache

from iai_bullet_sim import BasicSimulator,    \
                           MultiBody,         \
                           Link,              \
                           PerspectiveCamera, \
                           Transform,         \
                           Point3,            \
                           Vector3,           \
                           Quaternion,        \
                           Frame,             \
                           AABB,              \
                           MeshBody,          \
                           CartesianController, \
                           CartesianRelativePointController, \
                           CartesianRelativeVirtualPointController, \
                           CartesianRelativeVPointCOrientationController, \
                           CartesianRelativePointCOrientationController

from gym.spaces import Box  as BoxSpace, \
                       Dict as DictSpace
from gym        import Env

class BoxSampler(object):
    def __init__(self, b_min, b_max):
        if len(b_min) != len(b_max):
            raise Exception(f'Box bounds need to be the same size. Min: {len(b_min)} Max: {len(b_max)}')

        self.b_min = b_min
        self.b_max = b_max
    
    def sample(self):
        return (np.random.random(len(self.b_min)) * (np.asarray(self.b_max) - np.asarray(self.b_min))) + np.asarray(self.b_min)


class NoiseSampler(object):
    def __init__(self, dim, var, constant) -> None:
        self.constant = constant
        self.dim = dim
        self.var = var
        self._noise = None
        self.reset()
    
    def sample(self):
        if self.constant and self._noise is not None:
            return self._noise
        return np.random.normal(0, self.var, self.dim)
        
    def reset(self):
        self._noise = None
        self._noise = self.sample()


class PegEnv(Env):
    def __init__(self, cfg, show_gui=False):
        self.sim = BasicSimulator(cfg.action_frequency)
        self.sim.init('gui' if show_gui else 'direct')

        self.dt        = 1 / cfg.action_frequency
        self.workspace = AABB(Point3(0.3, -0.85, 0), 
                              Point3(0.85, 0.85, 0.8))

        self.robot = self.sim.load_urdf(cfg.robot.path, useFixedBase=True)
        self.eef   = self.robot.get_link(cfg.robot.eef)
        self.gripper_joints = [self.robot.joints[f] for f in cfg.robot.fingers]

        self.table = self.sim.create_box(Vector3(0.6, 1, 0.05), Transform.from_xyz(0.5, 0, -0.025), color=(1, 1, 1, 1), mass=0)
        self.board = self.sim.load_urdf(cfg.board.path, useFixedBase=True)
        self.peg   = self.sim.load_urdf(cfg.peg.path)

        if self.sim.visualizer is not None:
            self.sim.visualizer.set_camera_position(self.target_position, 0.6, -25, 65)

        self.board_sampler = BoxSampler(cfg.board.sampler.min, 
                                        cfg.board.sampler.max)

        robot_init_state = cfg.robot.initial_pose

        self._init_pose = Transform(Point3(*robot_init_state.position), 
                                    Quaternion(*robot_init_state.orientation))
        self.robot.set_joint_positions(cfg.robot.initial_pose.q, override_initial=True)
        self.robot.set_joint_positions(self.eef.ik(self._init_pose, 1000), override_initial=True)
        self.robot.set_joint_positions({j.name: robot_init_state.gripper_width / len(self.gripper_joints) for j in self.gripper_joints}, override_initial=True)

        self.eef_ft_sensor = self.robot.get_ft_sensor(cfg.robot.ft_joint)

        peg_position = self.eef.pose.position - Vector3(0, 0, 0.02)
        self.peg.initial_pose = Transform.from_xyz(*peg_position)
        self.peg.pose         = self.peg.initial_pose

        self.noise_samplers = {k: NoiseSampler(s.shape, 
                                               cfg.noise[k].variance, 
                                               cfg.noise[k].constant) for k, s in self.observation_space.sample().items() if k in cfg.noise}

        # print(f'Original: {temp_eef_pose}\nResolved EEF state: {self.eef.pose}\nDesired: {self._init_pose}\nPeg pos: {peg_position}')

        self.controller     = CartesianRelativePointCOrientationController(self.robot, self.eef)

        self._elapsed_steps = 0

    @property
    def visualizer(self):
        return self.sim.visualizer

    @property
    @lru_cache(1)
    def observation_space(self):
        return DictSpace({'position':      BoxSpace(low=self.workspace.min.numpy(), 
                                                    high=self.workspace.max.numpy()),
                          'gripper_width': BoxSpace(low=0.03, high=0.11, shape=(1,)),
                          'force':         BoxSpace(np.ones(3) * -5, np.ones(3) * 5),
                          'torque':        BoxSpace(np.ones(3) * -5, np.ones(3) * 5)
                         })

    @property
    @lru_cache(1)
    def action_space(self):
        """End effector position and gripper width relative displacement"""
        return DictSpace({'motion':  BoxSpace(-np.ones(3), np.ones(3)),
                          'gripper': BoxSpace(-1, 1, shape=(1,))})


    def reset(self):
        if self.visualizer is not None:
            dbg_pos, dbg_dist, dbg_pitch, dbg_yaw = self.visualizer.get_camera_position()

            dbg_rel_pos = dbg_pos - self.board.pose.position

        for v in self.noise_samplers.values():
            v.reset()

        self.sim.reset()

        board_position  = Point3(*self.board_sampler.sample())
        self.board.pose = Transform(board_position, Quaternion.from_euler(np.pi / 2, 0, 0))

        if self.visualizer is not None:
            self.visualizer.set_camera_position(self.board.pose.position + dbg_rel_pos, dbg_dist, dbg_pitch, dbg_yaw)

        x_goal = self.eef.pose
        # Only used to restore PID state after reset
        reset_controller = CartesianController(self.robot, self.eef)

        # Let the robot drop a bit
        for _ in range(5):
            reset_controller.act(x_goal)
            self._set_gripper_relative_goal(-self.dt)
            self.sim.update()

        # Wait for PID to restore the initial position
        while np.abs(reset_controller.delta).max() >= 1e-3:
            reset_controller.act(x_goal)
            self._set_gripper_relative_goal(-self.dt)
            self.sim.update()

        self.controller.reset()

        self._elapsed_steps = 0

        return self.observation()

    def step(self, action):
        if type(action) != dict:
            raise Exception(f'Action needs to be a dict with the fields "motion" and "gripper"')

        action_motion = action['motion'] / max(np.abs(action['motion']).max(), 1)
        self.controller.act(action_motion * self.dt)

        if 'gripper' in action:
            self._set_gripper_relative_goal(np.clip(action['gripper'], -1 * self.dt, 1 * self.dt))

        self.sim.update()

        obs = self.observation()
        done, success = self.is_terminated()
        reward = 100 * int(success)
        self._elapsed_steps += 1
        return obs, reward, done, {'success' : success}

    def observation(self):
        out = {'position'      : (self.peg.pose.position - self.target_position).numpy(),
               'gripper_width' : sum(self.robot.joint_state[j.name].position for j in self.gripper_joints),
               'force'         : self.eef_ft_sensor.get().linear.numpy(),
               'torque'        : self.eef_ft_sensor.get().angular.numpy()}
        for k in out:
            if k in self.noise_samplers:
                out[k] += self.noise_samplers[k].sample()
        
        return out

    def close(self):
        self.sim.kill()

    def is_terminated(self):
        """Checks if the episode is terminated

        Returns:
            tuple: (Done, Success)
        """        
        # Robot ran away
        if not self.workspace.inside(self.eef.pose.position):
            return True, False

        peg_pos_in_target = self.peg.pose.position - self.target_position

        # print(peg_pos_in_target)

        # Horizontal goal, vertical goal
        if (peg_pos_in_target * Vector3(1, 1, 0)).norm() < 0.005 and \
            peg_pos_in_target.z <= 0.005:
            return True, True
        elif (self.eef.pose.position - self.peg.pose.position).norm() > 0.25: # Peg was dropped
            return True, False
        return False, False

    @property
    def target_position(self):
        return self.board.links['target'].pose.position

    def _set_gripper_relative_goal(self, delta):
        self.robot.apply_joint_pos_cmds({j.name: self.robot.joint_state[j.name].position + delta for j in self.gripper_joints}, [800]*2)