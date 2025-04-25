import gymnasium
import torch.nn as nn
from torch.optim import Adam
from torchrl.envs import GymEnv, TransformedEnv, FlattenObservation, StepCounter
from torchrl.modules import SafeModule, MLP
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator, ActorCriticWrapper
from torch.distributions import Categorical
from torchrl.objectives import PPOLoss
from torchrl.objectives.value import GAE
from actor import ChessActor
from critic import ChessCritic
from tensordict.nn import TensorDictModule
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule, Actor
from tensordict.nn import InteractionType
class TrainingLoop():
    def __init__(self, entropy_coef=0.1, critic_coef=0.1, normalize_advantage=False,
                 lr=2e-4, gamma=0.99, lmbda=0.95):
        # Register custom gym environment
        gymnasium.register(id="Chess-v-1", entry_point="env:ChessEnv")
        self.env = GymEnv("Chess-v-1")
        self.env = TransformedEnv(
            self.env, 
            FlattenObservation(-2,-1, in_keys=["observation"], out_keys=["observation"]),
            StepCounter(),
        )

        # Get observation and action sizes
        obs_shape = self.env.observation_space.shape  # (7, 8)
        obs_dim = obs_shape[-2] * obs_shape[0]  # 64
        n_actions = self.env.action_space.n          # 4095 for 64x64

        actor_net = ChessActor(n_obs=obs_dim, n_act=n_actions)

        actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
        self.actor = ProbabilisticActor(module=actor_module, in_keys=["logits"], 
                                        spec=self.env.action_spec, 
                                        distribution_class=Categorical, 
                                        return_log_prob=True,
                                        default_interaction_type=InteractionType.RANDOM
                                        )

        # Critic network
        critic_net = ChessCritic(n_obs=obs_dim)

        self.value_operator = ValueOperator(
            module=critic_net,
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        
        self.loss_fn = PPOLoss(
            actor_network=self.actor,
            critic_network=self.value_operator,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            normalize_advantage=normalize_advantage,
        )

        self.advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.value_operator
        )

        self.optimizer = Adam(self.loss_fn.parameters(), lr=lr)
        
        
    def train(self, rollouts=3, max_steps=50, epochs_per_rollout=1):
        epoch = 0
        with open("epochs.csv", "w") as file:
            file.write("rollout,epoch,loss,type")
            
            for r in range(rollouts):
                rollout = self.env.rollout(policy=self.actor, max_steps=max_steps)
                
                self.advantage_module(rollout)
                for _ in range(epochs_per_rollout):
                    loss_dict = self.loss_fn(rollout)
                    policy_loss = loss_dict["loss_objective"]
                    value_loss  = loss_dict["loss_critic"]
                    ent_loss    = loss_dict["loss_entropy"]
                    total_loss = policy_loss + value_loss + ent_loss
                    
                    print(f"Epoch {epoch}, Total Loss: {total_loss}, Policy Loss: {policy_loss}, Value Loss: {value_loss}, Entropy Loss: {ent_loss}")
                    file.write(f"\n{r},{epoch},{total_loss},total")
                    file.write(f"\n{r},{epoch},{policy_loss},policy")
                    file.write(f"\n{r},{epoch},{value_loss},value")
                    file.write(f"\n{r},{epoch},{ent_loss},entropy")
                               
                               
                    epoch += 1
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()