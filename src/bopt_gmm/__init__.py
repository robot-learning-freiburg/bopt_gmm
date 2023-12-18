try:
    type(profile)
except NameError:
    import builtins

    # Blank line profiler annotator if line profiler is not active
    def profile(f):
        return f
    
    builtins.profile = profile

from . import gmm
from . import envs
from . import bopt
from . import common
from . import utils
from . import logging

def _attach_prime_bullet_serialization():
    try:
        from roebots.ros_serializer import ROS_SERIALIZER, \
                                           serialize_4_quaternion, \
                                           serialize_3_point, \
                                           serialize_3_vector
        from prime_bullet import Transform, \
                                 Quaternion, \
                                 Point3, \
                                 Vector3

        from geometry_msgs.msg import Pose        as PoseMsg, \
                                      Point       as PointMsg, \
                                      Vector3     as Vector3Msg, \
                                      Quaternion  as QuaternionMsg, \
                                      PoseStamped as PoseStampedMsg

        def serialize_pb_transform(tf : Transform):
            return PoseMsg(serialize_3_point(tf.position),
                           serialize_4_quaternion(tf.quaternion))

        ROS_SERIALIZER.add_serializer(serialize_4_quaternion, {Quaternion}, {QuaternionMsg})
        ROS_SERIALIZER.add_serializer(serialize_3_point, {Point3, Vector3}, {PointMsg})
        ROS_SERIALIZER.add_serializer(serialize_3_vector, {Point3, Vector3}, {Vector3Msg})
        ROS_SERIALIZER.add_serializer(serialize_pb_transform, {Transform}, {PoseMsg})
    except ModuleNotFoundError:
        pass

_attach_prime_bullet_serialization()
