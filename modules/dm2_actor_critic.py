from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation

from modules.dm2 import GLOBAL_LEG_OBS_DIM
from modules.dm2_memory import SharedLegMemory


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(resolve_nn_activation(activation))
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DM2ActorCriticRecurrent(ActorCritic):
    """Parameter-shared per-leg actor with a centralized critic."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 128],
        critic_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_legs: int = 4,
        local_obs_dim: int = 46,
        local_action_dim: int = 3,
        **kwargs,
    ):
        # We intentionally initialize nn.Module directly because the standard
        # ActorCritic constructor creates a centralized actor MLP.
        nn.Module.__init__(self)

        if kwargs:
            print(
                "DM2ActorCriticRecurrent got unexpected arguments, "
                f"which will be ignored: {list(kwargs.keys())}"
            )

        if rnn_type.lower() != "lstm":
            raise ValueError("DM2 currently supports only LSTM memory.")

        expected_actor_obs = num_legs * local_obs_dim
        expected_actions = num_legs * local_action_dim

        if num_actor_obs != expected_actor_obs:
            raise ValueError(
                f"Expected num_actor_obs={expected_actor_obs}, "
                f"got {num_actor_obs}."
            )

        if num_actions != expected_actions:
            raise ValueError(
                f"Expected num_actions={expected_actions}, "
                f"got {num_actions}."
            )

        self.num_legs = num_legs
        self.local_obs_dim = local_obs_dim
        self.local_action_dim = local_action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # Four legs use the same LSTM weights but maintain independent states.
        self.memory_a = SharedLegMemory(
            input_size=local_obs_dim,
            num_legs=num_legs,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # The critic retains one centralized recurrent state per environment.
        self.memory_c = Memory(
            input_size=num_critic_obs,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # This actor head is applied independently to every leg.
        self.actor = _build_mlp(
            input_dim=rnn_hidden_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=local_action_dim,
            activation=activation,
        )

        self.critic = _build_mlp(
            input_dim=rnn_hidden_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation,
        )

        self.noise_std_type = noise_std_type

        if noise_std_type == "scalar":
            self.std = nn.Parameter(
                init_noise_std * torch.ones(num_actions)
            )
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown noise_std_type: {noise_std_type}."
            )

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"DM2 shared leg actor: {self.actor}")
        print(f"DM2 centralized critic: {self.critic}")
        print(f"DM2 actor memory: {self.memory_a.rnn}")
        print(f"DM2 critic memory: {self.memory_c.rnn}")

    def _compute_action_mean(
        self,
        packed_actor_features: torch.Tensor,
    ) -> torch.Tensor:
        """Map [..., 4H] features to CPG-ordered [..., 12] means."""

        expected_size = self.num_legs * self.rnn_hidden_dim

        if packed_actor_features.shape[-1] != expected_size:
            raise ValueError(
                f"Expected packed actor feature size {expected_size}, "
                f"got {packed_actor_features.shape[-1]}."
            )

        local_features = packed_actor_features.reshape(
            *packed_actor_features.shape[:-1],
            self.num_legs,
            self.rnn_hidden_dim,
        )

        local_action_means = self.actor(local_features)

        return torch.cat(
            (
                local_action_means[..., 0],
                local_action_means[..., 1],
                local_action_means[..., 2],
            ),
            dim=-1,
        )

    def update_distribution(
        self,
        packed_actor_features: torch.Tensor,
    ):
        mean = self._compute_action_mean(packed_actor_features)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)

        self.distribution = Normal(mean, std)

    def act(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states=None,
    ) -> torch.Tensor:
        actor_features = self.memory_a(
            observations,
            masks,
            hidden_states,
        )
        actor_features = actor_features.squeeze(0)

        self.update_distribution(actor_features)
        return self.distribution.sample()

    def act_inference(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        actor_features = self.memory_a(observations)
        actor_features = actor_features.squeeze(0)
        return self._compute_action_mean(actor_features)

    def evaluate(
        self,
        critic_observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states=None,
    ) -> torch.Tensor:
        critic_features = self.memory_c(
            critic_observations,
            masks,
            hidden_states,
        )
        return self.critic(critic_features.squeeze(0))

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def detach_hidden_states(self, dones=None):
        self.memory_a.detach_hidden_states(dones)
        self.memory_c.detach_hidden_states(dones)

    def get_hidden_states(self):
        return (
            self.memory_a.hidden_states,
            self.memory_c.hidden_states,
        )


class DM2GlobalActorCriticRecurrent(
    DM2ActorCriticRecurrent
):
    """Shared per-leg actor receiving global observations.

    Every leg receives the same 83-D global observation together
    with its unique 2-D leg identifier. Actor parameters remain
    shared and recurrent states remain separate.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        **kwargs,
    ):
        configured_local_dim = kwargs.pop(
            "local_obs_dim",
            GLOBAL_LEG_OBS_DIM,
        )

        if configured_local_dim != GLOBAL_LEG_OBS_DIM:
            raise ValueError(
                "DM2GlobalActorCriticRecurrent requires "
                f"local_obs_dim={GLOBAL_LEG_OBS_DIM}, "
                f"got {configured_local_dim}."
            )

        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            local_obs_dim=GLOBAL_LEG_OBS_DIM,
            **kwargs,
        )
