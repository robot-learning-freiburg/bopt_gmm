import wandb

from .logger import LoggerBase


class WBLogger(LoggerBase):
    def __init__(self, project, run_id, allow_reinit=False):
        self.run = wandb.init(project=project, reinit=allow_reinit)
        self.run.name = run_id
        self.run.save()

    def __del__(self):
        if self.run is not None:
            self.run.finish()

    def log_config(self, config):
        self.run.config.update(config)

    def log(self, values : dict):
        self.run.log(values)

    def define_metric(self, metric, step=None):
        if step is None:
            return

        self.run.define_metric(metric, step)
