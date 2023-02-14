import numpy as np
import rospy
import tf2_ros
import roboticstoolbox as rp
import spatialmath     as sm

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

from multiprocessing import RLock

from gym.spaces import Box  as BoxSpace, \
                       Dict as DictSpace
from gym        import Env

from .utils     import BoxSampler, \
                       NoiseSampler

from rl_franka.panda import Panda

from geometry_msgs.msg import WrenchStamped as WrenchStampedMsg


class RealDrawerEnv(Env):
    def __init__(self, cfg, show_gui=False):
        # Only used for IK
        self._ik_model = rp.models.Panda()

        self.workspace = AABB(Point3(0.3, -0.85, 0), 
                              Point3(0.85, 0.85, 0.8))

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self._robot   = Panda(cfg.robot.state_prefix, 
                              cfg.robot.controllers)

        self._ref_frame    = cfg.reference_frame
        self._drawer_frame = cfg.drawer_frame
        
        self._ref_in_robot        = None
        self._current_drawer_pose = None

        self._robot_frame = cfg.robot.ref_frame
        self._ee_frame    = cfg.robot.ee_frame

        self._arm_reset_pose = cfg.robot.joint_reset_pose

        self._ee_rot = self._ik_model.fkine(self._arm_reset_pose).R

        self.starting_position_sampler = BoxSampler(cfg.initial_pose.position.min,
                                                    cfg.initial_pose.position.max)

        # self._init_pose = Transform(Point3(*robot_init_state.position), initial_rot)
        # self.robot.set_joint_positions(cfg.robot.initial_pose.q, override_initial=True)
        # self.robot.set_joint_positions(self.eef.ik(self._init_pose, 1000), override_initial=True)
        # self.robot.set_joint_positions({j.name: robot_init_state.gripper_width / len(self.gripper_joints) for j in self.gripper_joints}, override_initial=True)

        # self.eef_ft_sensor = self.robot.get_ft_sensor(cfg.robot.ft_joint)

        # print(f'Original: {temp_eef_pose}\nResolved EEF state: {self.eef.pose}\nDesired: {self._init_pose}\nPeg pos: {peg_position}')

        # self.controller     = CartesianRelativePointCOrientationController(self.robot, self.eef)

        self._elapsed_steps = 0
        self._n_reset = 0
        self._joint_reset_every = cfg.n_joint_reset

    @property
    def config_space(self):
        return sum([[f'{k}_noise_{x}' for x in 'xyz'] for k in self.noise_samplers.keys()], []) + \
                    [f'door_pose_{x}' for x in 'x,y,z,qx,qy,qz,qw'.split(',')] + \
                    [f'ee_pose_{x}' for x in 'x,y,z,qx,qy,qz,qw'.split(',')]

    def config_dict(self):
        out = {} 
        for k, n in self.noise_samplers.items(): 
            out.update(dict(zip([f'{k}_noise_{x}' for x in 'xyz'], n.sample())))
        out.update(dict(zip([f'door_pose_{x}' for x in 'x,y,z,qx,qy,qz,qw'.split(',')], self.door.pose.array())))
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
                        #   'gripper_width': BoxSpace(low=0.03, high=0.11, shape=(1,)),
                          'force':         BoxSpace(np.ones(3) * -5, np.ones(3) * 5),
                          'torque':        BoxSpace(np.ones(3) * -5, np.ones(3) * 5)
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
        return None

    def reset(self):
        # Establish reference frame if it's not existing
        if self._ref_in_robot is None:
            try:
                tf_stamped = self.tfBuffer.lookup_transform(self._robot_frame, self._ref_frame, rospy.Time(0))
                
                self._ref_in_panda = Transform(Point3(tf_stamped.transform.translation.x, 
                                                      tf_stamped.transform.translation.y, 
                                                      tf_stamped.transform.translation.z),
                                               Quaternion(tf_stamped.transform.rotation.x,
                                                          tf_stamped.transform.rotation.y,
                                                          tf_stamped.transform.rotation.z,
                                                          tf_stamped.transform.rotation.w))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pass

        # Reset joint space position every couple of episodes
        if self._n_reset % self._joint_reset_every == 0:
            self._robot.move_joints(self._arm_rest_pose)

        starting_position = Point3(*self.starting_position_sampler.sample())
        starting_pose     = sm.SE3.Rt(self._ee_rot, starting_position)

        q_start = self._ik_model.ikine(starting_pose)

        self._robot.move_joints(q_start)

        self._elapsed_steps = 0

        return self.observation()

    def step(self, action):
        if type(action) != dict:
            raise Exception(f'Action needs to be a dict with the fields "motion" and "gripper"')

        action_motion = action['motion'] / max(np.abs(action['motion']).max(), 1)
        self.controller.act(action_motion * self.dt)

        if 'gripper' in action:
            self._set_gripper_absolute_goal(np.clip(action['gripper'], 0, 1))

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
        out = {'position'      : (self._ref_in_panda.position - self._robot.ee_position).numpy(),
            #    'gripper_width' : sum(self.robot.joint_state[j.name].position for j in self.gripper_joints),
               'force'         : self._ref_in_panda.ext_force,
               'torque'        : self._ref_in_panda.ext_torque}
        
        return out

    def close(self):
        pass

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

    def _set_gripper_absolute_goal(self, target):
        self.robot.apply_joint_pos_cmds({j.name: self.robot.joints[j.name].q_max * target for j in self.gripper_joints}, [800]*2)