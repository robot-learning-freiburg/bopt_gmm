
class LoggerBase(object):
    def log_config(self, config):
        raise NotImplementedError

    def log(self, values : dict):
        raise NotImplementedError

    def define_metric(self, metric, step=None):
        raise NotImplementedError


class BlankLogger(LoggerBase):
    def log_config(self, config):
        pass

    def log(self, values : dict):
        pass

    def define_metric(self, metric, step=None):
        pass
