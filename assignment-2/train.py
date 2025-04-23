from torchrl.envs import GymEnv
import gym
from torchviz import make_dot
from torchrl.modules import ProbabilisticActor, MLP
from tensordict.nn import TensorDictModule
import torch.distributions as dist

gym.register(
    id="Chess-v0",
    entry_point="env:ChessEnv",
    reward_threshold=100,
)

env = GymEnv("Chess-v0")

n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.n

actor_mlp = MLP(
    in_features=n_obs,
    out_features=n_act,
)

actor_module = TensorDictModule(
    actor_mlp,
    in_keys=["observation"],
    out_keys=["logits"],
)

actor = ProbabilisticActor(
    module=actor_module,
    distribution_class=dist.Categorical,
    in_keys=["logits"],
)

critic_mlp = MLP(
    in_features=n_obs,
    out_features=1,
)
print(critic_mlp)


rollout = env.rollout(max_steps=100, policy=actor)
