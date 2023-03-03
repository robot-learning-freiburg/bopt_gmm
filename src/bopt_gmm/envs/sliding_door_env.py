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


class SlidingDoorEnv(Env):
    def __init__(self, cfg, show_gui=False):
        self.sim = BasicSimulator(cfg.action_frequency, use_egl=not show_gui)
        self.sim.init('gui' if show_gui else 'direct')

        self.dt        = 1 / cfg.action_frequency
        self.workspace = AABB(Point3(0.2, -0.65, -0.1), 
                              Point3(0.85, 0.65, 0.8))

        self.robot = self.sim.load_urdf(cfg.robot.path, useFixedBase=True)
        self.eef   = self.robot.get_link(cfg.robot.eef)
        self.gripper_joints = [self.robot.joints[f] for f in cfg.robot.fingers]

        self._target_position = cfg.goal_threshold

        self.table = self.sim.create_box(Vector3(0.6, 1, 0.05), 
                                         Transform.from_xyz(0.5, 0, -0.025), 
                                         color=(1, 1, 1, 1), 
                                         mass=0)
        # self.peg   = self.robot.links['peg'] #
        self.door  = self.sim.load_urdf(cfg.door.path,
                                        useFixedBase=False,
                                        use_self_collision='no_parents')

        self.frame = self.sim.load_urdf(cfg.frame.path,
                                        useFixedBase=True,
                                        use_self_collision='no_parents')

        self.reference_link = self.frame.links[cfg.frame.reference_link]
        self.board_sampler  = BoxSampler(cfg.frame.sampler.min, 
                                         cfg.frame.sampler.max)

        if self.sim.visualizer is not None:
            self.sim.visualizer.set_camera_position(self.frame.pose.position, 0.6, -25, 65)


        robot_init_state = cfg.robot.initial_pose

        initial_rot = Quaternion(*robot_init_state.orientation) if len(robot_init_state.orientation) == 4 else Quaternion.from_euler(*robot_init_state.orientation)

        self._robot_position_sampler = BoxSampler(robot_init_state.position.min,
                                                  robot_init_state.position.max)
        self._init_pose = Transform(Point3(*self._robot_position_sampler.center), initial_rot)
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

        self.render_camera  = PerspectiveCamera(self.sim, self.render_size[:2], 
                                                50, 0.1, 10.0, 
                                                Transform(Point3(-0.1, 0, 0.1), 
                                                          Quaternion.from_euler(0, np.deg2rad(30), -np.deg2rad(60))).dot(Transform.from_xyz(-1.2, 0, 0.1)),
                                                self.door)

        self._elapsed_steps = 0

    @property
    def config_space(self):
        return sum([[f'{k}_noise_{x}' for x in 'xyz'] for k in self.noise_samplers.keys()], []) + \
                    [f'door_pose_{x}' for x in 'x,y,z,qx,qy,qz,qw'.split(',')] + \
                    [f'ee_pose_{x}' for x in 'x,y,z,qx,qy,qz,qw'.split(',')]

    def config_dict(self):
        out = {} 
        for k, n in self.noise_samplers.items(): 
            out.update(dict(zip([f'{k}_noise_{x}' for x in 'xyz'], n.sample())))
        out.update(dict(zip([f'door_pose_{x}' for x in 'x,y,z,qx,qy,qz,qw'.split(',')], self.frame.pose.array())))
        out.update(dict(zip([f'ee_pose_{x}' for x in 'x,y,z,qx,qy,qz,qw'.split(',')], self.eef.pose.array())))
        return out

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
                          'torque':        BoxSpace(np.ones(3) * -5, np.ones(3) * 5),
                          'doorpos':       BoxSpace(low=-0.15, high=1.5, shape=(1,))
                         })

    @property
    @lru_cache(1)
    def action_space(self):
        """End effector position and gripper width relative displacement"""
        return DictSpace({'motion':  BoxSpace(-np.ones(3), np.ones(3)),
                          'gripper': BoxSpace(-1, 1, shape=(1,))})


    @property
    def render_size(self):
        return (640, 480, 3)

    def render(self, mode='rgb_array'):
        return self.render_camera.rgb()

    def reset(self):
        if self.visualizer is not None:
            dbg_pos, dbg_dist, dbg_pitch, dbg_yaw = self.visualizer.get_camera_position()

            dbg_rel_pos = dbg_pos - self.frame.pose.position

        for v in self.noise_samplers.values():
            v.reset()

        self.sim.reset()

        position_sample = self.board_sampler.sample()
        door_position  = Point3(*position_sample[:3])
        self.frame.pose = Transform(door_position, Quaternion.from_euler(0, np.deg2rad(0), position_sample[-1]))

        self.door.pose = self.frame.pose.dot(Transform.from_xyz_rpy(0, -0.15, 0, 0, 0, np.deg2rad(180)))

        if self.visualizer is not None:
            self.visualizer.set_camera_position(self.frame.pose.position + dbg_rel_pos, dbg_dist, dbg_pitch, dbg_yaw)

        x_goal = Transform(Point3(*self._robot_position_sampler.sample()), self.eef.pose.quaternion)
        # Only used to restore PID state after reset
        reset_controller = CartesianController(self.robot, self.eef)

        # Let the robot drop a bit
        for _ in range(5):
            reset_controller.act(x_goal)
            self._set_gripper_absolute_goal(0.5)
            self.sim.update()

        # Wait for PID to restore the initial position
        reset_steps = 0
        while (np.abs(reset_controller.delta) >= [1e-2, 0.1]).max():
            # print(f'EE: {self.eef.pose}\nDesired: {x_goal}\nDelta: {np.abs(reset_controller.delta).max()}')
            reset_controller.act(x_goal)
            self._set_gripper_absolute_goal(0.5)
            self.sim.update()
            if reset_steps > 1000:
                print('Initial sample seems to have been bad. Resetting again.')
                return self.reset()
            reset_steps += 1

        self.controller.reset()

        self._elapsed_steps = 0

        return self.observation()

    def step(self, action):
        if type(action) != dict:
            raise Exception(f'Action needs to be a dict with the fields "motion" and "gripper"')

        action_motion = action['motion'] # / max(np.abs(action['motion']).max(), 1)
        self.controller.act(action_motion * self.dt)

        if 'gripper' in action:
            self._set_gripper_absolute_goal(np.clip(action['gripper'], 0, 1))

        self.sim.update()

        obs = self.observation()
        done, success = self.is_terminated()
        reward = 100 * int(success)
        self._elapsed_steps += 1
        return obs, reward, done, {'success' : success}

    @property
    def door_position(self):
        return self.frame.pose.inv().dot(self.door.pose).position.y

    @property
    def reference_frame(self):
        return self.reference_link.pose

    def observation(self):
        out = {'position'      : self.reference_frame.inv().dot(self.eef.pose.position).numpy(),
               'gripper_width' : sum(self.robot.joint_state[j.name].position for j in self.gripper_joints),
               'force'         : self.eef_ft_sensor.get().linear.numpy(),
               'torque'        : self.eef_ft_sensor.get().angular.numpy(),
               'doorpos'       : self.door_position
               }
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

        door_pos = self.door_position

        # print(peg_pos_in_target)

        # Horizontal goal, vertical goal
        if door_pos >= self._target_position:
            return True, True

        return False, False

    def _set_gripper_absolute_goal(self, target):
        self.robot.apply_joint_pos_cmds({j.name: self.robot.joints[j.name].q_max * target for j in self.gripper_joints}, [800]*2)
