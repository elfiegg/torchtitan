"""Reduce pinned host memory for the compact GB300 presets."""

from dataclasses import replace


UNPOOLER = 'feed_forward.unpooler.*'


def fit_host_memory(config):
    policy = config.activation_remat
    if policy.save_regions.count(UNPOOLER) != 1 or policy.offload_regions.count(UNPOOLER) != 1:
        raise ValueError('The compact source must save and offload unpooler outputs exactly once')
    config.activation_remat = replace(
        policy,
        save_regions=[name for name in policy.save_regions if name != UNPOOLER],
        offload_regions=[name for name in policy.offload_regions if name != UNPOOLER],
    )
    return replace(config)
