
class Policy(object):
    def __init__(self, config):
        """
        Base class for all policies, has an abstract method predict().
        """
        self.config = config