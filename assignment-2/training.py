from torchrl.envs import GymEnv
import gym
from torchviz import make_dot
from policy import ChessPolicy

gym.register(
    id="Chess-v0",
    entry_point="env:ChessEnv",
    reward_threshold=100,
)

env = GymEnv("Chess-v0")

policy = ChessPolicy(env.action_space.n)
dot = make_dot(policy(env.reset()["observation"]), params=dict(policy.named_parameters()))
dot.render("policy_graph", format="png")

env.rollout(max_steps=1000, policy=policy)

