import numpy as np
from move_plan.policy.policy import Policy
from ped_sim.envs.utils.action import ActionRot, ActionXY, ActionAcc, ActionBicycle

class SRNN(Policy):
	def __init__(self, config):
		super().__init__(config)
		self.time_step = self.config.env.time_step
		self.name = 'srnn'
		self.trainable = True
		self.multiagent_training = True


	# clip the self.raw_action and return the clipped action
	def clip_action(self, raw_action, v_pref):

		kinematics = self.config.action_space.kinematics
		
		# clip the action
		if kinematics == 'holonomic':
			act_norm = np.linalg.norm(raw_action)
			if act_norm > v_pref:
				raw_action[0] = raw_action[0] / act_norm * v_pref
				raw_action[1] = raw_action[1] / act_norm * v_pref
			return ActionXY(raw_action[0], raw_action[1])
		
		elif kinematics == 'unicycle':
			max_heaing_change = 0.1
			raw_action[0] = v_pref*raw_action[0]
			raw_action[1] = max_heaing_change*raw_action[1]

			raw_action[0] = np.clip(raw_action[0], -v_pref, v_pref)
			raw_action[1] = np.clip(raw_action[1], -max_heaing_change, max_heaing_change)
			# action[1] is change of theta


			return ActionRot(raw_action[0], raw_action[1])
		
		elif kinematics == 'double_integrator_unicycle':

			max_linear_acc = 1.75 # m/s^2 from literature
			max_angular_acc = 0.1 # rad/s^2
			raw_action[0] = max_linear_acc*raw_action[0]
			raw_action[1] = max_angular_acc*raw_action[1]

			raw_action[0] = np.clip(raw_action[0], -max_linear_acc, max_linear_acc)
			raw_action[1] = np.clip(raw_action[1], -max_angular_acc, max_angular_acc)

			return ActionAcc(raw_action[0], raw_action[1])
		
		elif kinematics == 'bicycle':

			max_linear_acc = 1.75 # m/s^2 from literature
			max_steering_change = 0.1 # rad/s
			raw_action[0] = max_linear_acc*raw_action[0]
			raw_action[1] = max_steering_change*raw_action[1]

			raw_action[0] = np.clip(raw_action[0], -max_linear_acc, max_linear_acc)
			raw_action[1] = np.clip(raw_action[1], -max_steering_change, max_steering_change)

			return ActionBicycle(raw_action[0], raw_action[1])


class selfAttn_merge_SRNN(SRNN):
	def __init__(self, config):
		super().__init__(config)
		self.name = 'selfAttn_merge_srnn'

class SocialMovePlan(SRNN):
	def __init__(self, config):
		super().__init__(config)

		self.name = 'social_move_plan'
		
