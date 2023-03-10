from dataclasses import dataclass

from bopt_gmm.logging import MP4VideoLogger


@dataclass
class RunAccumulator:
    _steps     : int   = 0
    _reward    : float = 0.0
    _successes : int   = 0
    _runs      : int   = 0

    def log_run(self, steps, reward, success):
        self._runs      += 1
        self._steps    += steps
        self._reward    += reward
        self._successes += int(success)

    def get_stats(self):
        if self._runs == 0:
            return 0.0, 0, 0.0
        return self._steps / self._runs, self._reward, self._successes / self._runs


class AgentWrapper(object):
    def __init__(self, model, gripper_command=0.0) -> None:
        self.model = model
        self.pseudo_bopt_step = 0
        self._gripper_command = gripper_command

    def predict(self, obs):
        # if callable(self._force_norm):
        #     obs = self._force_norm(obs)
        # elif 'force' in obs:
        #     obs['force'] = obs['force'] * self._force_norm
        return {'motion': self.model.predict(obs).flatten(), 'gripper': self._gripper_command}

    def step(self, *args):
        pass
    
    def has_gp_stage(self):
        return False

    def is_in_gp_stage(self):
        return False

    def get_bopt_step(self):
        self.pseudo_bopt_step += 1
        return self.pseudo_bopt_step


def run_episode(env, agent, max_steps, post_step_hook=None):
    observation    = env.reset()
    episode_return = 0.0

    initial_conditions = env.config_dict()

    for step in range(max_steps):
        action = agent.predict(observation)
        # print(observation)
        post_observation, reward, done, info = env.step(action)
        episode_return += reward
        done = done or (step == max_steps - 1)

        if post_step_hook is not None:
            post_step_hook(step, env, agent, observation, post_observation, action, reward, done, info)

        observation = post_observation
        
        if done:
            break
    
    info['initial_conditions'] = initial_conditions

    return episode_return, step, info


# Standard Hooks
def post_step_hook_dispatcher(*hooks):
    """Executes a list of post-step hooks in order"""
    def dispatcher(step, env, agent, obs, post_obs, action, reward, done, info):
        for h in hooks:
            h(step, env, agent, obs, post_obs, action, reward, done, info)
    return dispatcher


def post_step_hook_bopt(_, env, agent, obs, post_obs, action, reward, done, info):
    """Post-step hook calling step() on the agent"""
    agent.step(obs, post_obs, action, reward, done)


def gen_video_logger_and_hook(dir_path, filename, image_size, frame_rate=30.0):
    """Generates a MP4 video logger and a post-step hook that writes the image  
       returned by env.render() to this logger.
    """
    logger = MP4VideoLogger(dir_path, filename, image_size)

    def video_hook(_, env, *args):
        logger.write_image(env.render()[:,:,::-1])
    
    return logger, video_hook
