import hydra
import numpy as np
import rospy
import tf2_ros
import yaml

from argparse        import ArgumentParser
from dataclasses     import dataclass
from multiprocessing import RLock
from roebots         import ROSVisualizer

from sensor_msgs.msg   import JointState       as JointStateMsg
from std_msgs.msg      import String           as StringMsg
from geometry_msgs.msg import TransformStamped as TransformStampedMsg, \
                              WrenchStamped    as WrenchStampedMsg

from bopt_gmm.envs   import ENV_TYPES
from bopt_gmm.gmm    import GMM, \
                            utils as gmm_utils
from bopt_gmm.common import AgentWrapper
from bopt_gmm.bopt   import GMMOptAgent


@dataclass
class AppState:
    last_update = None
    agent_lock  = RLock()
    visualizer  = None
    successes   = 0
    runs        = 0


if __name__ == '__main__':
    parser = ArgumentParser(description='Using hydra without losing control of your code.')
    parser.add_argument('hy', type=str, help='Hydra config to use. Relative to root of "config" dir')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    args = parser.parse_args()

    rospy.init_node('interactive_gmm')

    # Point hydra to the root of your config dir. Here it's hard-coded, but you can also
    # use "MY_MODULE.__path__" to localize it relative to your python package
    hydra.initialize(config_path="../config")
    cfg   = hydra.compose(args.hy, overrides=args.overrides)
    
    app_state = AppState()
    app_state.visualizer = ROSVisualizer('vis', world_frame='gmm_reference')

    gmm   = GMM.load_model(cfg.bopt_agent.gmm.model)
    agent = GMMOptAgent(gmm, cfg.bopt_agent)

    print(f'Activations: {gmm._priors}')

    parameter_description = '<robot name="gmm">\n{}\n</robot>'.format('\n'.join([
        f'<joint name="{n}" type="prismatic">\n   <limit lower="{v.lower}" upper="{v.upper}"/>\n</joint>' for n, v in sorted(agent.config_space.items())
    ]))

    rospy.set_param('/gmm_description', parameter_description)


    def cb_js(msg : JointStateMsg):
        new_update = dict(zip(msg.name, msg.position))
        if app_state.last_update is None or max([app_state.last_update[k] != v for k, v in new_update.items()]):
            app_state.last_update = new_update
            with app_state.agent_lock:
                agent.update_model(new_update)
                app_state.successes = 0
                app_state.runs = 0
            
            if app_state.visualizer is not None:
                opt_dims = list(cfg.bopt_agent.opt_dims.cvars.unary) if cfg.bopt_agent.opt_dims.cvars.unary is not None else []
                if cfg.bopt_agent.opt_dims.cvars.nary is not None:
                    opt_dims += cfg.bopt_agent.opt_dims.cvars.nary

                gmm_utils.draw_gmm(app_state.visualizer, 'gmm', agent.model, opt_dims, 1e-1)

    def cb_save(msg : StringMsg):
        with open(f'{msg.data}_n{int(cfg.env.noise.position.variance * 100):02d}_{int(app_state.successes / app_state.runs * 100):03d}.yaml', 'w') as f:
            yaml.dump(app_state.last_update, f)

    sub_save = rospy.Subscriber('/save_params', StringMsg, callback=cb_save, queue_size=1, tcp_nodelay=True)
    sub = rospy.Subscriber('/gmm_updates', JointStateMsg, callback=cb_js, queue_size=1, tcp_nodelay=True)    
    pub = rospy.Publisher('/joint_states', JointStateMsg, queue_size=1, tcp_nodelay=True)    
    pub_wrench = rospy.Publisher('/wrist_wrench', WrenchStampedMsg, tcp_nodelay=True, queue_size=1)

    env = ENV_TYPES[cfg.env.type](cfg.env, show_gui=True)

    obs = env.reset()

    tf_broadcaster = tf2_ros.TransformBroadcaster()
    ref_msg  = TransformStampedMsg()
    ref_msg.header.frame_id = 'panda_link0'
    ref_msg.child_frame_id  = 'gmm_reference'
    
    wrench_msg = WrenchStampedMsg()
    wrench_msg.header.frame_id = env.robot.joints[cfg.env.robot.ft_joint].link._name

    env_steps = 0
    while not rospy.is_shutdown():
        frame_start = rospy.Time.now()
        
        with app_state.agent_lock:
            motion = agent.predict(obs)
            gmm_utils.draw_gmm_stats(app_state.visualizer, 'gmm', agent.model, obs)

        action = {'motion'  : motion, 
                  'gripper' : cfg.bopt_agent.gripper_command}

        
        obs, reward, done, info = env.step(action)
        env_steps += 1

        # Publish robot state for visualization in RVIZ
        js_msg = JointStateMsg()
        js_msg.header.stamp = rospy.Time.now()
        js_msg.name = env.robot.dynamic_joint_names
        js_msg.position = env.robot.q

        pub.publish(js_msg)
    
        wrench_msg.header.stamp = js_msg.header.stamp
        wrench_msg.wrench.force.x, wrench_msg.wrench.force.y, wrench_msg.wrench.force.z    = obs['force']
        wrench_msg.wrench.torque.x, wrench_msg.wrench.torque.y, wrench_msg.wrench.torque.z = obs['torque']
        pub_wrench.publish(wrench_msg)

        ref_pose = env.reference_frame
        ref_msg.transform.translation.x, \
        ref_msg.transform.translation.y, \
        ref_msg.transform.translation.z = ref_pose.position
        
        ref_msg.transform.rotation.x, \
        ref_msg.transform.rotation.y, \
        ref_msg.transform.rotation.z, \
        ref_msg.transform.rotation.w = ref_pose.quaternion

        ref_msg.header.stamp = js_msg.header.stamp
        tf_broadcaster.sendTransform(ref_msg)

        # print(env.observation()
        # print(f'Is terminated: {terminated}\nIs success: {success}')
        if done or env_steps > cfg.bopt_agent.num_episode_steps:
            if info['success']:
                app_state.successes += 1
            app_state.runs += 1
            obs = env.reset()
            env_steps = 0

            print(f'Accuracy since last parameter update: {app_state.successes / app_state.runs} ({app_state.successes}/{app_state.runs})')
        
        frame_remainder = (1 / 30) - (rospy.Time.now() - frame_start).to_sec()
        if frame_remainder > 0:
            rospy.sleep(frame_remainder)

        
