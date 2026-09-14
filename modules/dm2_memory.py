from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import unpad_trajectories


class SharedLegMemory(nn.Module):
    """Shared LSTM with an independent recurrent state for each leg."""

    def __init__(
        self,
        input_size: int,
        num_legs: int = 4,
        num_layers: int = 1,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_legs = num_legs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.packed_hidden_size = num_legs * hidden_size

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        # Stored externally as:
        # (h, c), each [num_layers, num_envs, num_legs * hidden_size]
        self.hidden_states = None

    def _unpack_hidden_states(self, hidden_states, batch_size: int):
        """Convert packed [L, N, 4H] states into [L, N*4, H]."""

        if hidden_states is None:
            return None

        if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) != 2:
            raise ValueError("SharedLegMemory expects LSTM (h, c) states.")

        unpacked = []

        for state in hidden_states:
            expected_shape = (
                self.num_layers,
                batch_size,
                self.packed_hidden_size,
            )

            if tuple(state.shape) != expected_shape:
                raise ValueError(
                    f"Expected packed hidden state {expected_shape}, "
                    f"got {tuple(state.shape)}."
                )

            state = state.reshape(
                self.num_layers,
                batch_size,
                self.num_legs,
                self.hidden_size,
            )
            state = state.reshape(
                self.num_layers,
                batch_size * self.num_legs,
                self.hidden_size,
            )
            unpacked.append(state)

        return tuple(unpacked)

    def _pack_hidden_states(self, hidden_states, batch_size: int):
        """Convert internal [L, N*4, H] states into [L, N, 4H]."""

        packed = []

        for state in hidden_states:
            state = state.reshape(
                self.num_layers,
                batch_size,
                self.num_legs,
                self.hidden_size,
            )
            state = state.reshape(
                self.num_layers,
                batch_size,
                self.packed_hidden_size,
            )
            packed.append(state)

        return tuple(packed)

    def forward(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states=None,
    ) -> torch.Tensor:
        """Process flattened [*, 4*D] local observations."""

        expected_input_size = self.num_legs * self.input_size

        if observations.shape[-1] != expected_input_size:
            raise ValueError(
                f"Expected observation size {expected_input_size}, "
                f"got {observations.shape[-1]}."
            )

        batch_mode = masks is not None

        if batch_mode:
            # PPO update:
            # observations [T, trajectories, 4D]
            if observations.ndim != 3:
                raise ValueError(
                    "Batch mode expects observations with shape [T, B, 4D]."
                )

            time_steps, batch_size, _ = observations.shape

            leg_observations = observations.reshape(
                time_steps,
                batch_size,
                self.num_legs,
                self.input_size,
            )
            leg_observations = leg_observations.reshape(
                time_steps,
                batch_size * self.num_legs,
                self.input_size,
            )

            unpacked_hidden = self._unpack_hidden_states(
                hidden_states,
                batch_size,
            )

            leg_output, _ = self.rnn(
                leg_observations,
                unpacked_hidden,
            )

            packed_output = leg_output.reshape(
                time_steps,
                batch_size,
                self.num_legs,
                self.hidden_size,
            )
            packed_output = packed_output.reshape(
                time_steps,
                batch_size,
                self.packed_hidden_size,
            )

            return unpad_trajectories(packed_output, masks)

        # Rollout/inference:
        # observations [N, 4D]
        if observations.ndim != 2:
            raise ValueError(
                "Inference mode expects observations with shape [N, 4D]."
            )

        batch_size = observations.shape[0]

        leg_observations = observations.reshape(
            batch_size,
            self.num_legs,
            self.input_size,
        )
        leg_observations = leg_observations.reshape(
            batch_size * self.num_legs,
            self.input_size,
        )

        unpacked_hidden = self._unpack_hidden_states(
            self.hidden_states,
            batch_size,
        )

        leg_output, new_hidden_states = self.rnn(
            leg_observations.unsqueeze(0),
            unpacked_hidden,
        )

        self.hidden_states = self._pack_hidden_states(
            new_hidden_states,
            batch_size,
        )

        return leg_output.reshape(
            1,
            batch_size,
            self.packed_hidden_size,
        )

    def reset(self, dones: torch.Tensor | None = None):
        if dones is None:
            self.hidden_states = None
            return

        if self.hidden_states is None:
            return

        done_mask = dones.bool()

        for state in self.hidden_states:
            state[:, done_mask, :] = 0.0

    def detach_hidden_states(self, dones: torch.Tensor | None = None):
        if self.hidden_states is None:
            return

        if dones is None:
            self.hidden_states = tuple(
                state.detach() for state in self.hidden_states
            )
            return

        done_mask = dones.bool()

        for state in self.hidden_states:
            state[:, done_mask, :] = state[:, done_mask, :].detach()
