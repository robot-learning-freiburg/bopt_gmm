from .peg_env  import PegEnv
from .door_env import DoorEnv
from .real_drawer_env  import RealDrawerEnv
from .real_door_env    import RealDoorEnv
from .sliding_door_env import SlidingDoorEnv

ENV_TYPES = {'door': DoorEnv,
             'peg' : PegEnv,
             'real_drawer'  : RealDrawerEnv,
             'real_door'    : RealDoorEnv,
             'sliding_door' : SlidingDoorEnv}
