policy_factory = dict()
def none_policy(config):
    return None

from move_plan.policy.SocialMovePlan import SocialMovePlan

policy_factory['none'] = none_policy
policy_factory['social_move_plan'] = SocialMovePlan