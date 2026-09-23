from __future__ import annotations

import torch


NUM_LEGS = 4
GLOBAL_OBS_DIM = 83
LOCAL_PRIVATE_OBS_DIM = 32
LEG_ID_DIM = 2
COORDINATION_OBS_DIM = 12
LOCAL_OBS_DIM = 46
GLOBAL_LEG_OBS_DIM = GLOBAL_OBS_DIM + LEG_ID_DIM
LOCAL_ACTION_DIM = 3

LEG_NAMES = ("FL", "FR", "RL", "RR")


def _build_local_source_indices() -> tuple[tuple[int, ...], ...]:
    """Build the 32 global-observation indices used by each leg."""

    rows = []

    for leg in range(NUM_LEGS):
        indices = (
            list(range(0, 3))                         # base linear velocity
            + list(range(3, 6))                       # base angular velocity
            + list(range(6, 9))                       # projected gravity
            + list(range(37, 40))                     # velocity commands
            + [9 + leg, 13 + leg, 17 + leg]          # local joint positions
            + [21 + leg, 25 + leg, 29 + leg]         # local joint velocities
            + [33 + leg]                              # local foot contact
            + [40 + leg, 44 + leg, 48 + leg]         # previous local CPG action
            + [52 + 2 * leg, 53 + 2 * leg]           # local r_x, r_y
            + [60 + leg, 64 + leg]                   # sin(theta), cos(theta)
            + [68 + 2 * leg, 69 + 2 * leg]           # local r_dot_x, r_dot_y
            + [76 + leg]                             # local theta_dot
            + list(range(80, 83))                    # CPG design parameters
        )

        if len(indices) != LOCAL_PRIVATE_OBS_DIM:
            raise RuntimeError("Incorrect DM2 local-observation mapping.")

        rows.append(tuple(indices))

    return tuple(rows)


LOCAL_SOURCE_INDICES = _build_local_source_indices()

# Low-bandwidth coordination information broadcast to every leg:
# four foot contacts followed by sin(theta) and cos(theta)
# for all four CPG oscillators.
COORDINATION_SOURCE_INDICES = (
    tuple(range(33, 37))
    + tuple(range(60, 68))
)

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

        self.coordination_indices = torch.tensor(
            COORDINATION_SOURCE_INDICES,
            dtype=torch.long,
            device=device,
        )

    def build_local_observations(
        self,
        global_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Convert [N, 83] global observations into [N, 4, 46]."""

        if global_observations.ndim != 2:
            raise ValueError(
                "Expected global observations with shape [N, 83], "
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

        coordination = global_observations[
            :,
            self.coordination_indices,
        ]
        coordination = coordination.unsqueeze(1).expand(
            global_observations.shape[0],
            NUM_LEGS,
            COORDINATION_OBS_DIM,
        )

        return torch.cat(
            (local_observations, leg_ids, coordination),
            dim=-1,
        )

    def build_global_leg_observations(
        self,
        global_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Give every leg the full global observation and its leg ID.

        This is an architectural diagnostic, not the final localized
        DM2 observation design.
        """

        if global_observations.ndim != 2:
            raise ValueError(
                "Expected global observations with shape [N, 83], "
                f"got {tuple(global_observations.shape)}."
            )

        if global_observations.shape[-1] != GLOBAL_OBS_DIM:
            raise ValueError(
                f"Expected {GLOBAL_OBS_DIM} global observation dimensions, "
                f"got {global_observations.shape[-1]}."
            )

        num_envs = global_observations.shape[0]

        shared_global_observations = (
            global_observations.unsqueeze(1).expand(
                num_envs,
                NUM_LEGS,
                GLOBAL_OBS_DIM,
            )
        )

        leg_ids = self.leg_ids.to(
            dtype=global_observations.dtype
        )
        leg_ids = leg_ids.unsqueeze(0).expand(
            num_envs,
            NUM_LEGS,
            LEG_ID_DIM,
        )

        return torch.cat(
            (
                shared_global_observations,
                leg_ids,
            ),
            dim=-1,
        )

    def flatten_global_leg_observations(
        self,
        global_leg_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Convert [N, 4, 85] observations into [N, 340]."""

        expected_shape = (
            NUM_LEGS,
            GLOBAL_LEG_OBS_DIM,
        )

        if (
            global_leg_observations.ndim != 3
            or tuple(
                global_leg_observations.shape[1:]
            ) != expected_shape
        ):
            raise ValueError(
                "Expected global leg observations with shape "
                f"[N, {NUM_LEGS}, {GLOBAL_LEG_OBS_DIM}], "
                f"got {tuple(global_leg_observations.shape)}."
            )

        return global_leg_observations.reshape(
            global_leg_observations.shape[0],
            NUM_LEGS * GLOBAL_LEG_OBS_DIM,
        )

    def flatten_local_observations(
        self,
        local_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Convert [N, 4, 46] local observations into [N, 184]."""

        expected_shape = (NUM_LEGS, LOCAL_OBS_DIM)

        if (
            local_observations.ndim != 3
            or tuple(local_observations.shape[1:]) != expected_shape
        ):
            raise ValueError(
                f"Expected local observations with shape [N, 4, 46], "
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
