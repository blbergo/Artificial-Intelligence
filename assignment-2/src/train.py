import gymnasium
import torch.nn as nn
from torch.optim import Adam
from torchrl.envs import GymEnv, TransformedEnv, FlattenObservation, StepCounter
from torchrl.modules import SafeModule, MLP
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator, ActorCriticWrapper
from torch.distributions import Categorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
import torch

# Register custom gym environment
gymnasium.register(id="Chess-v0", entry_point="env:ChessEnv", reward_threshold=100)
env = GymEnv("Chess-v0")
env = TransformedEnv(
    env, 
    FlattenObservation(-2,-1, in_keys=["observation"], out_keys=["observation"]),
    StepCounter(),
)
# Get observation and action sizes
obs_shape = env.observation_space.shape  # (8, 8)
obs_dim = obs_shape[-1] * obs_shape[0]  # 64
n_actions = env.action_space.n          # 4096 for 64x64

actor_net = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, n_actions),
)

# Wrap actor in SafeModule with distribution
actor_module = SafeModule(actor_net, in_keys=["observation"], out_keys=["logits"])
actor = ProbabilisticActor(module=actor_module, in_keys=["logits"], spec=env.action_spec, distribution_class=Categorical, return_log_prob=True)

# Critic network
critic_net = MLP(
    in_features=obs_dim,
    out_features=1,
    num_cells=[32,32]
)

value_operator = ValueOperator(
    module=critic_net,
    in_keys=["observation"],
    out_keys=["state_value"],
)

# PPO loss and optimizer
policy = ActorCriticWrapper(actor, value_operator)

# Hyperparameters
clip_epsilon = 0.1
entropy_coef = 0.005
critic_coef = 0.5
normalize_advantage = True

lr = 1e-5
gamma = 0.99
lmbda = 0.95

loss_fn = ClipPPOLoss(
    actor=policy.get_policy_operator(),
    critic=policy.get_value_operator(),
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_coef,
    critic_coef=critic_coef,
    normalize_advantage=normalize_advantage,
    )

advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=policy.get_value_operator(),
)

optimizer = Adam(loss_fn.parameters(), lr=lr)

def training_loop(epochs=10, max_steps=1):
    # Actor network
    with open("epochs.csv", "w") as file:
        file.write("epoch,loss\n")
        
        for epoch in range(epochs):
            rollout = env.rollout(policy=policy, max_steps=max_steps)
            rollout["next", "state_value"] = value_operator(rollout["next", "observation"])
            rollout["sample_log_prob"] = rollout["sample_log_prob"].detach()
            
            advantage_module(rollout)
            loss_dict = loss_fn(rollout)
            total_loss = ( loss_dict["loss_objective"]
                        + loss_dict["loss_critic"]
                        + loss_dict["loss_entropy"] )
            
            print(f"Epoch: {epoch} Loss: {total_loss:.4f}, KL: {loss_dict['kl_approx']:.5f}, ClipFrac: {loss_dict['clip_fraction']:.5f}")
            file.write(f"{epoch},{total_loss.item()}\n")
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=0.5)
            optimizer.step()
            
if __name__ == "__main__":
    training_loop(epochs=10, optimization_steps=1)