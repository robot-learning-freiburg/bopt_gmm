try:
    import torch

    from .actor  import DenseNetActor
    from .agent  import SACAgent
    from .critic import DenseNetCritic
    from .env    import SACGMMEnv
except ModuleNotFoundError:
    class DenseNetActor():
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError

    class SACAgent():
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError

    class DenseNetCritic():
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError

    class SACGMMEnv():
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError

