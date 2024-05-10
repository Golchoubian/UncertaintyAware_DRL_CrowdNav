class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'


class Danger(object):
    def __init__(self, min_dist, speedAtdmin):
        self.min_dist = min_dist
        self.speedAtdmin = speedAtdmin

    def __str__(self):
        return 'Intrusion'


class Collision(object):
    def __init__(self, col_speed):
        self.col_speed = col_speed

    def __str__(self):
        return 'Collision'

class OutRoad(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Out of road'

class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''
    
class Collision_Vehicle(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Vehicle Collision'
