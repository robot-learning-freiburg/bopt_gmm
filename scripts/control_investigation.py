import numpy as np
import rospy
import roboticstoolbox as rp
import spatialmath     as sm
import tf2_ros

from iai_bullet_sim    import AABB, \
                              Point3, \
                              Vector3, \
                              Quaternion
from rl_franka.panda   import Panda
from roebots           import ROSVisualizer

from std_msgs.msg      import Float64 as Float64Msg
from geometry_msgs.msg import Vector3 as Vector3Msg


if __name__ == '__main__':
    rospy.init_node('bopt_gmm_control_investigatio')

    # Only used for IK
    ik_model = rp.models.Panda()

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    lin_stiffness = 500
    ang_stiffness = 20

    robot = Panda('/franka_state_controller', '/controllers')
    vis   = ROSVisualizer('~/vis', world_frame='panda_link0')

    pub_lin_stiffness = rospy.Publisher('/controllers/cartesian_impedance_controller/update_stiffness/linear/f', 
                                        Float64Msg, tcp_nodelay=True, queue_size=1)
    pub_ang_stiffness = rospy.Publisher('/controllers/cartesian_impedance_controller/update_stiffness/angular/f', 
                                        Float64Msg, tcp_nodelay=True, queue_size=1)
    pub_pos_gt        = rospy.Publisher('~/pos_gt',    Vector3Msg, tcp_nodelay=True, queue_size=1)
    pub_pos_goal      = rospy.Publisher('~/pos_goal',  Vector3Msg, tcp_nodelay=True, queue_size=1)
    pub_pos_delta     = rospy.Publisher('~/pos_delta', Vector3Msg, tcp_nodelay=True, queue_size=1)
    
    while robot.state.O_T_EE is None:
        print('Waiting for robot state')
        rospy.sleep(0.1)

    starting_pose = robot.state.O_T_EE

    rospy.sleep(0.3)
    robot.cm.activate_controller(robot.CART_IMPEDANCE_CONTROLLER)
    rospy.sleep(0.3)
    msg_lin_stiffness = Float64Msg(data=lin_stiffness)
    msg_ang_stiffness = Float64Msg(data=ang_stiffness)
    pub_lin_stiffness.publish(msg_lin_stiffness)
    pub_ang_stiffness.publish(msg_ang_stiffness)
    rospy.sleep(0.1)

    rotation_center = Point3(*starting_pose[:3, 3].flatten()) + Vector3(0.075, 0, 0)
    offset_vector   = Point3(*starting_pose[:3, 3].flatten()) - rotation_center

    t_start = rospy.Time.now()
    while not rospy.is_shutdown():
        vis.begin_draw_cycle('debug')
        now = rospy.Time.now()
        rotation = Quaternion.from_euler(0, 0, (now - t_start).to_sec() * 1.2).matrix()

        rv   = rotation.dot(offset_vector)
        goal_pos = rotation_center + rv

        goal_pose = starting_pose.copy()
        goal_pose[:3, 3] = goal_pos

        goal_msg = Vector3Msg()
        goal_msg.x, goal_msg.y, goal_msg.z = rotation_center - goal_pos
        pub_pos_goal.publish(goal_msg)

        gt_msg = Vector3Msg()
        gt_msg.x, gt_msg.y, gt_msg.z = rotation_center - robot.state.O_T_EE[:3, 3].flatten()
        pub_pos_gt.publish(gt_msg)

        delta_msg = Vector3Msg()
        delta_msg.x, delta_msg.y, delta_msg.z = goal_pos - robot.state.O_T_EE[:3, 3].flatten()
        pub_pos_delta.publish(delta_msg)

        robot.async_move_ee_cart_absolute(goal_pose)

        vis.draw_poses('debug', np.eye(4), 0.1, 0.005, [goal_pose])
        vis.render('debug')
        rospy.sleep(0.1)
