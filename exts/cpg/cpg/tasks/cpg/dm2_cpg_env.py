# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from modules.dm2 import DM2ObservationMapper

from .cpg_env import CPGUnitreeA1Env


class DM2CPGUnitreeA1Env(CPGUnitreeA1Env):
    """CPG environment with decentralized per-leg policy observations."""

    def _get_observations(self):
        observations = super()._get_observations()

        # Create lazily because the base constructor may request observations
        # before initialization of subclass-specific members is complete.
        if not hasattr(self, "_dm2_observation_mapper"):
            self._dm2_observation_mapper = DM2ObservationMapper(
                device=self.device
            )

        # Actor: four 52-D local observations, flattened to 208-D.
        local_observations = (
            self._dm2_observation_mapper.build_local_observations(
                observations["policy"]
            )
        )
        observations["policy"] = (
            self._dm2_observation_mapper.flatten_local_observations(
                local_observations
            )
        )


        # Critic remains the original centralized 80-D observation.
        return observations
