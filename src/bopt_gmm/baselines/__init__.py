from .        import sac_gmm       

try:
    import torch

    from .mlp     import MLP
    from .bc_lstm import LSTMPolicy, \
                         LSTMPolicyConfig
except ModuleNotFoundError:
    class MLP():
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError
    
    class LSTMPolicy():
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError
    
    class LSTMPolicyConfig():
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError
