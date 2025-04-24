import gymnasium
import torch.nn as nn
from torch.optim import Adam
from torchrl.envs import GymEnv, TransformedEnv, FlattenObservation
from torchrl.modules import SafeModule
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator, ActorCriticWrapper
from torch.distributions import Categorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# Register custom gym environment
# Register custom gym environment
gymnasium.register(id="Chess-v0", entry_point="env:ChessEnv", reward_threshold=100)
env = GymEnv("Chess-v0")
env = TransformedEnv(
    env, 
    FlattenObservation(-2,-1, in_keys=["observation"], out_keys=["observation"]),
)
# Get observation and action sizes
obs_shape = env.observation_space.shape  # (8, 8)
obs_dim = obs_shape[-1] * obs_shape[0]  # 64
n_actions = env.action_space.n          # 4096 for 64x64

# Actor network
actor_net = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, n_actions),
)

# Wrap actor in SafeModule with distribution
actor_module = SafeModule(actor_net, in_keys=["observation"], out_keys=["logits"])
actor = ProbabilisticActor(module=actor_module, in_keys=["logits"], spec=env.action_spec, distribution_class=Categorical)

# Critic network
critic_net = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

value_operator = ValueOperator(
    module=critic_net,
    in_keys=["observation"],
    out_keys=["state_value"],
)

# PPO loss and optimizer
loss_fn = ClipPPOLoss(actor=actor, critic=value_operator, clip_epsilon=0.2)
optimizer = Adam(loss_fn.parameters(), lr=2e-4)
advantage_module = GAE(gamma=0.99, lmbda=0.95, value_network=value_operator)

policy = ActorCriticWrapper(actor, value_operator)

epochs = 10
for epoch in range(epochs):
    rollout = env.rollout(
        policy=policy,
        max_steps=50)
    loss_fn(rollout)
    
    