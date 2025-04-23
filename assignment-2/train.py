from torchrl.envs import GymEnv
import gym
from torchviz import make_dot
from torchrl.modules import Actor, MLP, ValueOperator
from torchrl.objectives import DDPGLoss

gym.register(
    id="Chess-v0",
    entry_point="env:ChessEnv",
    reward_threshold=100,
)

env = GymEnv("Chess-v0")

n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.n

actor = Actor(MLP(in_features=n_obs, out_features=n_act), in_keys=["observation"], out_keys=["action"])
#value_net = ValueOperator(
 #   MLP(in_features=n_obs + n_act, out_features=1),
  #  in_keys=["observation", "action"],
#)

#ddpg_loss = DDPGLoss(actor_network=actor, value_network=value_net)
rollout = env.rollout(max_steps=100, policy=actor)
#loss_vals = ddpg_loss(rollout)
#print(loss_vals)