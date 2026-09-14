from __future__ import annotations

import torch


NUM_LEGS = 4
GLOBAL_OBS_DIM = 80
LOCAL_OBS_DIM = 31
LOCAL_ACTION_DIM = 3

LEG_NAMES = ("FL", "FR", "RL", "RR")


def _build_local_source_indices() -> tuple[tuple[int, ...], ...]:
    """Build the 29 global-observation indices used by each leg."""

    rows = []

    for leg in range(NUM_LEGS):
        indices = (
            list(range(0, 3))                         # base angular velocity
            + list(range(3, 6))                       # projected gravity
            + list(range(34, 37))                     # velocity commands
            + [6 + leg, 10 + leg, 14 + leg]          # local joint positions
            + [18 + leg, 22 + leg, 26 + leg]         # local joint velocities
            + [30 + leg]                              # local foot contact
            + [37 + leg, 41 + leg, 45 + leg]         # previous local CPG action
            + [49 + 2 * leg, 50 + 2 * leg]           # local r_x, r_y
            + [57 + leg, 61 + leg]                   # sin(theta), cos(theta)
            + [65 + 2 * leg, 66 + 2 * leg]           # local r_dot_x, r_dot_y
            + [73 + leg]                             # local theta_dot
            + list(range(77, 80))                    # CPG design parameters
        )

        if len(indices) != LOCAL_OBS_DIM - 2:
            raise RuntimeError("Incorrect DM2 local-observation mapping.")

        rows.append(tuple(indices))

    return tuple(rows)


LOCAL_SOURCE_INDICES = _build_local_source_indices()

# First coordinate: front (+1) / rear (-1)
# Second coordinate: left (+1) / right (-1)
LEG_ID_VALUES = (
    (1.0, 1.0),    # FL
    (1.0, -1.0),   # FR
    (-1.0, 1.0),   # RL
    (-1.0, -1.0),  # RR
)


class DM2ObservationMapper:
    """Map centralized CPG observations/actions to the DM2 per-leg layout."""

    def __init__(self, device: str | torch.device = "cpu"):
        self.source_indices = torch.tensor(
            LOCAL_SOURCE_INDICES,
            dtype=torch.long,
            device=device,
        )
        self.leg_ids = torch.tensor(
            LEG_ID_VALUES,
            dtype=torch.float,
            device=device,
        )

    def build_local_observations(
        self,
        global_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Convert [N, 80] global observations into [N, 4, 31]."""

        if global_observations.ndim != 2:
            raise ValueError(
                "Expected global observations with shape [N, 80], "
                f"got {tuple(global_observations.shape)}."
            )

        if global_observations.shape[-1] != GLOBAL_OBS_DIM:
            raise ValueError(
                f"Expected {GLOBAL_OBS_DIM} global observation dimensions, "
                f"got {global_observations.shape[-1]}."
            )

        local_observations = global_observations[:, self.source_indices]

        leg_ids = self.leg_ids.to(dtype=global_observations.dtype)
        leg_ids = leg_ids.unsqueeze(0).expand(
            global_observations.shape[0],
            -1,
            -1,
        )

        return torch.cat((local_observations, leg_ids), dim=-1)

    def flatten_local_observations(
        self,
        local_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Convert [N, 4, 31] local observations into [N, 124]."""

        expected_shape = (NUM_LEGS, LOCAL_OBS_DIM)

        if (
            local_observations.ndim != 3
            or tuple(local_observations.shape[1:]) != expected_shape
        ):
            raise ValueError(
                f"Expected local observations with shape [N, 4, 31], "
                f"got {tuple(local_observations.shape)}."
            )

        return local_observations.reshape(
            local_observations.shape[0],
            NUM_LEGS * LOCAL_OBS_DIM,
        )

    @staticmethod
    def local_actions_to_cpg(
        local_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Convert [N, 4, 3] local actions to the CPG [N, 12] order."""

        expected_shape = (NUM_LEGS, LOCAL_ACTION_DIM)

        if (
            local_actions.ndim != 3
            or tuple(local_actions.shape[1:]) != expected_shape
        ):
            raise ValueError(
                f"Expected local actions with shape [N, 4, 3], "
                f"got {tuple(local_actions.shape)}."
            )

        return torch.cat(
            (
                local_actions[:, :, 0],
                local_actions[:, :, 1],
                local_actions[:, :, 2],
            ),
            dim=-1,
        )
