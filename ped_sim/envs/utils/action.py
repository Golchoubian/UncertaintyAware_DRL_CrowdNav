from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])
ActionAcc = namedtuple('ActionAcc', ['u_a', 'u_alpha'])
ActionBicycle = namedtuple('ActionBicycle', ['a', 'steering_change_rate'])
