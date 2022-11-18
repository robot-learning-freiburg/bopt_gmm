from dataclasses import dataclass

from .bopt_agent_base import BOPTGMMAgentBase, \
                             BOPTAgentConfig, \
                             no_op

@dataclass
class BOPTAgentGMMConfig(BOPTAgentConfig):
    base_accuracy : float = 0.3

class BOPTGMMAgent(BOPTGMMAgentBase):
    def __init__(self, base_gmm, config : BOPTAgentGMMConfig, obs_transform=no_op, logger=None):
        super().__init__(base_gmm, config, obs_transform, logger)

        self.init_optimizer(config.base_accuracy)
