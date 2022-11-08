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

from .utils     import BoxSampler, \
                       NoiseSampler


class DoorEnv(Env):
    def __init__(self, cfg, show_gui=False):
        self.sim = BasicSimulator(cfg.action_frequency)
        self.sim.init('gui' if show_gui else 'direct')

        self.dt        = 1 / cfg.action_frequency
        self.workspace = AABB(Point3(0.3, -0.85, 0), 
                              Point3(0.85, 0.85, 0.8))

        self.robot = self.sim.load_urdf(cfg.robot.path, useFixedBase=True)
        self.eef   = self.robot.get_link(cfg.robot.eef)
        self.gripper_joints = [self.robot.joints[f] for f in cfg.robot.fingers]

        self._target_position = np.deg2rad(cfg.open_threshold)

        self.table = self.sim.create_box(Vector3(0.6, 1, 0.05), 
                                         Transform.from_xyz(0.5, 0, -0.025), 
                                         color=(1, 1, 1, 1), 
                                         mass=0)
        # self.peg   = self.robot.links['peg'] #
        self.door  = self.sim.load_urdf(cfg.door.path, useFixedBase=True)
        self.reference_link = self.door.links[cfg.door.reference_link]
        self.board_sampler  = BoxSampler(cfg.door.sampler.min, 
                                         cfg.door.sampler.max)

        if self.sim.visualizer is not None:
            self.sim.visualizer.set_camera_position(self.door.pose.position, 0.6, -25, 65)


        robot_init_state = cfg.robot.initial_pose

        initial_rot = Quaternion(*robot_init_state.orientation) if len(robot_init_state.orientation) == 4 else Quaternion.from_euler(*robot_init_state.orientation)

        self._init_pose = Transform(Point3(*robot_init_state.position), initial_rot)
        self.robot.set_joint_positions(cfg.robot.initial_pose.q, override_initial=True)
        self.robot.set_joint_positions(self.eef.ik(self._init_pose, 1000), override_initial=True)
        self.robot.set_joint_positions({j.name: robot_init_state.gripper_width / len(self.gripper_joints) for j in self.gripper_joints}, override_initial=True)

        self.eef_ft_sensor = self.robot.get_ft_sensor(cfg.robot.ft_joint)

        self.noise_samplers = {k: NoiseSampler(s.shape, 
                                               cfg.noise[k].variance, 
                                               cfg.noise[k].constant) for k, s in self.observation_space.sample().items() if k in cfg.noise}

        # print(f'Original: {temp_eef_pose}\nResolved EEF state: {self.eef.pose}\nDesired: {self._init_pose}\nPeg pos: {peg_position}')

        # self.controller     = CartesianRelativePointCOrientationController(self.robot, self.eef)
        self.controller     = CartesianRelativeVPointCOrientationController(self.robot, self.eef, 0.02)

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

            dbg_rel_pos = dbg_pos - self.door.pose.position

        for v in self.noise_samplers.values():
            v.reset()

        self.sim.reset()

        door_position  = Point3(*self.board_sampler.sample())
        self.door.pose = Transform(door_position, Quaternion.from_euler(0, 0, 0))

        if self.visualizer is not None:
            self.visualizer.set_camera_position(self.door.pose.position + dbg_rel_pos, dbg_dist, dbg_pitch, dbg_yaw)

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

        # if 'gripper' in action:
        #     self._set_gripper_relative_goal(np.clip(action['gripper'], -1 * self.dt, 1 * self.dt))

        handle_pos = self.door.joint_state['handle_joint'].position
        switch = max(np.sign(self.door.joints['handle_joint'].q_max * 0.5 - handle_pos), 0.0)
        # print(switch)

        self.door.apply_joint_pos_cmds([0, 0], [5000 * switch, 1])

        self.sim.update()

        obs = self.observation()
        done, success = self.is_terminated()
        reward = 100 * int(success)
        self._elapsed_steps += 1
        return obs, reward, done, {'success' : success}

    def observation(self):
        out = {'position'      : (self.eef.pose.position - self.reference_link.pose.position).numpy(),
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

        door_pos = self.door.joint_state['hinge_joint'].position

        # print(peg_pos_in_target)

        # Horizontal goal, vertical goal
        if door_pos >= self._target_position:
            return True, True

        return False, False

    def _set_gripper_relative_goal(self, delta):
        self.robot.apply_joint_pos_cmds({j.name: self.robot.joint_state[j.name].position + delta for j in self.gripper_joints}, [800]*2)