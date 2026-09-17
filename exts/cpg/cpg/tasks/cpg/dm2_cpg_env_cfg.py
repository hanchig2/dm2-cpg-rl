# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from modules.cpg import CPGCfg
from modules.dm2 import GLOBAL_OBS_DIM, LOCAL_OBS_DIM, NUM_LEGS

from .cpg_env_cfg import (
    CPGCouplingK1Cfg,
    CPGCouplingK2Cfg,
    CPGCouplingK4Cfg,
    CPGUnitreeA1FlatEnvCfg,
    CPGUnitreeA1FlatEnvCfg_PLAY,
    CPGUnitreeA1RoughEnvCfg_EVAL,
)


@configclass
class DM2CPGUnitreeA1FlatEnvCfg(CPGUnitreeA1FlatEnvCfg):
    """Flat CPG environment configuration for DM2 training."""

    observation_space = NUM_LEGS * LOCAL_OBS_DIM
    state_space = GLOBAL_OBS_DIM


@configclass
class DM2CPGUnitreeA1FlatEnvCfg_PLAY(CPGUnitreeA1FlatEnvCfg_PLAY):
    """Flat CPG environment configuration for DM2 playback."""

    observation_space = NUM_LEGS * LOCAL_OBS_DIM
    state_space = GLOBAL_OBS_DIM


@configclass
class DM2CPGUnitreeA1FlatEnvCfg_EVAL(
    CPGUnitreeA1RoughEnvCfg_EVAL
):
    """Flat evaluation configuration for DM2-CPG-RL."""

    observation_space = NUM_LEGS * LOCAL_OBS_DIM
    state_space = GLOBAL_OBS_DIM


@configclass
class DM2CPGUnitreeA1FlatEnvCfg_K1(DM2CPGUnitreeA1FlatEnvCfg):
    """DM2 flat training configuration with CPG coupling K=1."""

    cpg_config: CPGCfg = CPGCouplingK1Cfg()


@configclass
class DM2CPGUnitreeA1FlatEnvCfg_K2(DM2CPGUnitreeA1FlatEnvCfg):
    """DM2 flat training configuration with CPG coupling K=2."""

    cpg_config: CPGCfg = CPGCouplingK2Cfg()


@configclass
class DM2CPGUnitreeA1FlatEnvCfg_K4(DM2CPGUnitreeA1FlatEnvCfg):
    """DM2 flat training configuration with CPG coupling K=4."""

    cpg_config: CPGCfg = CPGCouplingK4Cfg()


@configclass
class DM2CPGUnitreeA1FlatEnvCfg_EVAL_K1(DM2CPGUnitreeA1FlatEnvCfg_EVAL):
    """DM2 flat evaluation configuration with CPG coupling K=1."""

    cpg_config: CPGCfg = CPGCouplingK1Cfg()


@configclass
class DM2CPGUnitreeA1FlatEnvCfg_EVAL_K2(DM2CPGUnitreeA1FlatEnvCfg_EVAL):
    """DM2 flat evaluation configuration with CPG coupling K=2."""

    cpg_config: CPGCfg = CPGCouplingK2Cfg()


@configclass
class DM2CPGUnitreeA1FlatEnvCfg_EVAL_K4(DM2CPGUnitreeA1FlatEnvCfg_EVAL):
    """DM2 flat evaluation configuration with CPG coupling K=4."""

    cpg_config: CPGCfg = CPGCouplingK4Cfg()
