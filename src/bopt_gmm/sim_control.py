import numpy as np
import os
import signal
import socketio
import time
import threading

from datetime import datetime
from aiohttp import web
from prime_bullet import Vector3, \
                         res_pkg_path, \
                         DebugVisualizer
from omegaconf import ListConfig

from bopt_gmm.utils import save_demo_npz



class SimState(object):
    """Class used to control the current state of the simulation"""

    def __init__(self, input_mapping=np.eye(3)):
        self.should_reset    = True
        self.last_action     = None
        self.should_shutdown = False
        self.should_save     = False
        self.input_mapping   = input_mapping


def sim_loop(env, sim_state, save_dir):
    print("Sim loop has started")

    vis = env.visualizer
    if vis is None:
        raise Exception('Teleop can only performed with a visualizer')

    observations = []
    actions      = []
    while not sim_state.should_shutdown:
        start = datetime.now()
        if sim_state.should_reset or sim_state.should_save:
            if sim_state.should_save:  # and len(observations) >= 15:
                print('Remote signaled save...')
                save_demo_npz(observations=observations, save_dir=save_dir)
            else:
                print('Remote signaled reset...')
            observations = []
            env.reset()
            sim_state.should_reset = False
            sim_state.should_save  = False
        else:
            action = {'motion': np.zeros(3),
                      'gripper': 0.5}
            if sim_state.last_action is not None:
                raw_action = Vector3(*sim_state.last_action[:3])
                raw_action = sim_state.input_mapping.dot(raw_action)

                if isinstance(vis, DebugVisualizer):
                    cam_pose  = vis.get_camera_pose()
                    right_dir = cam_pose.dot(Vector3.unit_x()) * Vector3(1, 1, 0)
                    # print(cam_pose.matrix())
                    right_dir /= right_dir.norm()
                    fwd_dir    = Vector3.unit_z().cross(right_dir)

                    # print(raw_action)
                    cat_action = fwd_dir   * sim_state.last_action[1] + \
                                 right_dir * sim_state.last_action[0] + \
                                 Vector3.unit_z() * sim_state.last_action[2]
                else:
                    cat_action = Vector3(*raw_action)

                action['motion'] = cat_action.numpy() * 0.5
                sim_state.last_action = None
            observation, reward, done, info = env.step(action)
            observation.update(action)
            observations.append(observation)
            if done:
                sim_state.should_reset = True
                if info["success"]:
                    save_demo_npz(observations=observations, save_dir=save_dir)
        delta = datetime.now() - start
        time.sleep(max(0.0, env.dt - delta.total_seconds()))
    print("Sim loop has ended")
    env.close()


def get_index(name):
    async def index(request):
        """serve the html webpage used to control the robot"""
        with open(res_pkg_path(f"package://bopt_gmm/web/static/{name}.html"), 'r') as f:
            return web.Response(text=f.read(), content_type="text/html")

    return index


def get_process_sigint(sim_state, app):
    """Returns function to shutdown app"""

    def process_sigint(*args):
        """Shutdown the app"""
        sim_state.should_shutdown = True
        app.shutdown()

    return process_sigint


def get_sio(env, sim_state):
    sio = socketio.AsyncServer()

    @sio.event
    def connect(sid, env):
        print(f"A socket.io client connected: {sid}")

    @sio.event
    def disconnect(sid):
        print(f"A socket.io client disconnected: {sid}")

    @sio.event
    def command(sid, data):
        sim_state.last_action = [data["x"], data["y"], data["z"]]
        if "pitch" in data:
            sim_state.last_action.append(data["pitch"])
        if "yaw" in data:
            sim_state.last_action.append(data["yaw"])
        if "gripper" in data:
            sim_state.last_action.append(data["gripper"])

        if data["save"]:
            sim_state.should_save = True

        if data["reset"]:
            sim_state.should_reset = True

        if data["shutdown"]:
            sim_state.should_shutdown = True

    return sio


def start_web_app(cfg, env, save_dir):
    """
    Starts a web app using socketio
    such that the robot can be controlled
    using the keyboard
    """
    if 'lin_input_mapping' in cfg:
        val = cfg.lin_input_mapping
        if type(val) in {int, float}:
            mapping = np.eye(3) * val
        elif type(val) in {list, ListConfig}:
            if type(val[0]) in {list, ListConfig}:
                mapping = np.array(val)
                if mapping.shape != (3, 3):
                    raise Exception(f'Matrix mapping needs to be of (3, 3) shape. Got {mapping.shape}.')
            else:
                mapping = np.diag(val)
        else:
            raise Exception(f'Incompatible type "{type(val)}" for encoding input mappings.')
    else:
        mapping = np.eye(3)

    sim_state = SimState(mapping)
    app = web.Application()
    sio = get_sio(env, sim_state)
    sio.attach(app)
    process_sigint = get_process_sigint(sim_state, app)
    sim_thread = threading.Thread(target=sim_loop, args=(env, sim_state, save_dir))
    sim_thread.start()
    signal.signal(signal.SIGINT, process_sigint)
    index = get_index('control')
    app.router.add_get("/", index)
    web.run_app(app, port=5000)
    sim_thread.join()
