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
        rospy.init_node('bopt_gmm_real_drawer')

        # Only used for IK
        self._ik_model = rp.models.Panda()

        self.workspace = AABB(Point3(-0.4, -0.4, 0), 
                              Point3( 0.4,  0.4, 0.8))

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self._robot   = Panda(cfg.robot.state_prefix, 
                              cfg.robot.controllers_prefix)

        self._ref_frame    = cfg.reference_frame
        self._drawer_frame = cfg.drawer_frame
        self._target_position = cfg.open_position

        self._robot_T_ref         = None
        self._ref_T_robot         = None
        self._current_drawer_pose = None
        self._ref_P_drawer        = None

        self._robot_frame = cfg.robot.reference_frame
        self._ee_frame    = cfg.robot.endeffector_frame

        self._arm_reset_pose = np.asarray(cfg.robot.initial_pose.q)

        self._ee_rot = self._ik_model.fkine(self._arm_reset_pose).R

        self.starting_position_sampler = BoxSampler(cfg.robot.initial_pose.position.min,
                                                    cfg.robot.initial_pose.position.max)

        # self._init_pose = Transform(Point3(*robot_init_state.position), initial_rot)
        # self.robot.set_joint_positions(cfg.robot.initial_pose.q, override_initial=True)
        # self.robot.set_joint_positions(self.eef.ik(self._init_pose, 1000), override_initial=True)
        # self.robot.set_joint_positions({j.name: robot_init_state.gripper_width / len(self.gripper_joints) for j in self.gripper_joints}, override_initial=True)

        # self.eef_ft_sensor = self.robot.get_ft_sensor(cfg.robot.ft_joint)

        # print(f'Original: {temp_eef_pose}\nResolved EEF state: {self.eef.pose}\nDesired: {self._init_pose}\nPeg pos: {peg_position}')

        # self.controller     = CartesianRelativePointCOrientationController(self.robot, self.eef)

        self._elapsed_steps = 0
        self._n_reset = 0
        self._joint_reset_every = cfg.robot.joint_reset_frequency
        self._goal_lookup_timer = rospy.Timer(rospy.Duration(0.1), self._goal_look_up)

    def _goal_look_up(self, *args):
        try:
            tf_stamped = self.tfBuffer.lookup_transform(self._ref_frame, self._drawer_frame, rospy.Time(0))
            
            self._ref_P_drawer = Point3(tf_stamped.transform.translation.x, 
                                        tf_stamped.transform.translation.y, 
                                        tf_stamped.transform.translation.z)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    @property
    def config_space(self):
        return sum([[f'{k}_noise_{x}' for x in 'xyz'] for k in self.noise_samplers.keys()], []) + \
                    [f'ee_pose_{x}' for x in 'x,y,z,qx,qy,qz,qw'.split(',')]

    def config_dict(self):
        raise NotImplementedError
        out = {} 
        for k, n in self.noise_samplers.items(): 
            out.update(dict(zip([f'{k}_noise_{x}' for x in 'xyz'], n.sample())))
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

    def _is_tower_collision(self, q=None):
        q = self._robot.state.q if q is None else q
        # Experimentally found -0.6 * q0^2 - 0.45 as approximation for tower collisions
        q1_threshold = -1 * q[0]**6 - 0.45
        return q[1] < q1_threshold

    def reset(self):
        # Establish reference frame if it's not existing
        while self._robot_T_ref is None:
            try:
                tf_stamped = self.tfBuffer.lookup_transform(self._robot_frame, self._ref_frame, rospy.Time(0))
                
                quat = sm.UnitQuaternion(tf_stamped.transform.rotation.w,
                                         (tf_stamped.transform.rotation.x,
                                          tf_stamped.transform.rotation.y,
                                          tf_stamped.transform.rotation.z))

                self._robot_T_ref = sm.SE3.Rt(quat.R, (tf_stamped.transform.translation.x, 
                                                       tf_stamped.transform.translation.y, 
                                                       tf_stamped.transform.translation.z))
                self._ref_T_robot = self._robot_T_ref.inv()
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print(f'Trying to lookup reference frame {self._ref_frame} in {self._robot_frame}')

        print('Bla')
        while self._ref_P_drawer is None or self._ref_P_drawer.z > 0.015:
            if self._ref_P_drawer is None:
                print('Waiting for drawer to be registered')
            else:
                print(f'Waiting for drawer position to be less than 0.015. Currently: {self._ref_P_drawer.z}')
            rospy.sleep(0.3)

        print('Bla2')
        while not self._robot.is_operational:
            print('Waiting for robot to become operational again...')
            rospy.sleep(0.3)

        print('Bla3')
        # Reset joint space position every couple of episodes
        if self._n_reset % self._joint_reset_every == 0:
            self._robot.move_joint_position(self._arm_reset_pose, 0.15)
            print('FAKE MOVE ROBOT TO INITIAL Q')

        print('Bla4')
        while True:
            starting_position = Point3(*self.starting_position_sampler.sample())
            starting_pose     = sm.SE3.Rt(self._ee_rot, starting_position)

            q_start, ik_success, failure, _, residual = self._ik_model.ik_lm_chan(starting_pose, q0=self._arm_reset_pose)

            if ik_success:
                if self._is_tower_collision(q_start):
                    print('IK SOLUTION WOULD COLLIDE WITH TOWER. GOING AGAIN.')
                    continue

                print(f'FAKE MOVE ROBOT TO IK SOLUTION FOR SAMPLED\n{starting_pose}\nQ: {q_start}')
                break
            else:
                print(f'IK FAILED! RESIDUAL {residual} "{failure}"\n{starting_pose}')

        print('Bla5')
        self._robot.move_joint_position(q_start, vel_scale=0.15)

        print('Bla6')
        self._elapsed_steps = 0

        return self.observation()

    def step(self, action):
        if type(action) != dict:
            raise Exception(f'Action needs to be a dict with the fields "motion" and "gripper"')

        action_motion = action['motion'] / max(np.abs(action['motion']).max(), 1)
        # self.controller.act(action_motion * self.dt)

        # if 'gripper' in action:
        #     self._set_gripper_absolute_goal(np.clip(action['gripper'], 0, 1))

        rospy.sleep(1 / 30)

        obs = self.observation()
        done, success = self.is_terminated()
        reward = 100 * int(success)
        self._elapsed_steps += 1
        return obs, reward, done, {'success' : success}

    def observation(self):
        while self._ref_T_robot is None or self._robot.state.O_T_EE is None:
            print('Waiting for reference frame and endeffector frame')
            rospy.sleep(0.1)

        out = {'position'      : (self._ref_T_robot * self._robot.state.O_T_EE)[:3, 3].flatten(),
            #    'gripper_width' : sum(self.robot.joint_state[j.name].position for j in self.gripper_joints),
               'force'         : (self._ref_T_robot.R * self._robot.state.ext_force),
               'torque'        : (self._ref_T_robot.R * self._robot.state.ext_torque)}
        
        return out

    def close(self):
        pass

    def is_terminated(self):
        """Checks if the episode is terminated

        Returns:
            tuple: (Done, Success)
        """        
        # Robot has faulted
        if not self._robot.is_operational:
            print('Termination due to inoperative robot')
            return True, False

        # Robot ran away
        if not self.workspace.inside((self._ref_T_robot * self._robot.state.O_T_EE)[:3,3]):
            print('EE is not in safety area')
            return True, False

        if self._is_tower_collision():
            print('Robot is about to collide with the tower')
            return True, False

        # TODO: ADD FORCE SAFETY CHECK

        try:
            tf_stamped = self.tfBuffer.lookup_transform(self._robot_frame, self._ref_frame, rospy.Time(0))
            
            quat = sm.UnitQuaternion(tf_stamped.transform.rotation.w,
                                    (tf_stamped.transform.rotation.x,
                                        tf_stamped.transform.rotation.y,
                                        tf_stamped.transform.rotation.z))

            self._robot_T_ref = sm.SE3.Rt(quat.R, (tf_stamped.transform.translation.x, 
                                                    tf_stamped.transform.translation.y, 
                                                    tf_stamped.transform.translation.z))
            self._ref_T_robot = self._robot_T_ref.inv()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

        # print(peg_pos_in_target)

        # Horizontal goal, vertical goal
        if self._ref_P_drawer is not None and self._ref_P_drawer.z >= self._target_position:
            print('Terminated due to drawer being open')
            return True, True

        return False, False

    def _set_gripper_absolute_goal(self, target):
        self.robot.apply_joint_pos_cmds({j.name: self.robot.joints[j.name].q_max * target for j in self.gripper_joints}, [800]*2)