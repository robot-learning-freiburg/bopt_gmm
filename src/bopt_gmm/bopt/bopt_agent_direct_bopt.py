from .bopt_agent_base import BOPTGMMAgentBase, \
                             BOPTAgentConfig

@dataclass
class BOPTAgentGMMConfig(BOPTAgentConfig):
    base_accuracy : float = 0.3

class BOPTGMMAgent(BOPTGMMAgentBase):
    def __init__(self, base_gmm, config : BOPTAgentGMMConfig, logger=None):
        super().__init__(base_gmm, config, logger)

        self.init_optimizer(config.base_accuracy)
