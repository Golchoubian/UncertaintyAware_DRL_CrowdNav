from gym.envs.registration import register

register(
    id='PedSimPred-v0',
    entry_point='ped_sim.envs:PedSimPred',
)

register(
    id='PedSim-v0',
    entry_point='ped_sim.envs:PedSim',
)