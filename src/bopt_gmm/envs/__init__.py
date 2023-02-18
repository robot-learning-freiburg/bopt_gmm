from .peg_env  import PegEnv
from .door_env import DoorEnv
from .real_drawer_env import RealDrawerEnv
from .sliding_door_env import SlidingDoorEnv

ENV_TYPES = {'door': DoorEnv,
             'peg' : PegEnv,
             'real_drawer'  : RealDrawerEnv,
             'sliding_door' : SlidingDoorEnv}
