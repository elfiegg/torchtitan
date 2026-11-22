# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``torchtitan.train`` with the env-gated Dist-MoE runtime patches imported first.

    ENTRYPOINT=torchtitan.experiments.dist_moe_patches.train ./run_train.sh ...

Each patch module applies itself on import when its env var is set and is a
no-op otherwise, so with no ``EXP_*`` variable set this is ``torchtitan.train``.
The import order is fixed: ``h3_offload`` re-patches the source that
``swiglu_recompute`` installs and must come after it. A module that is not in
this checkout is skipped, unless its env var asks for it.
"""

import importlib
import importlib.util
import os

# (module, env var) in the required import order.
_PATCHES = (
    ("swiglu_recompute", "EXP_SWIGLU_RECOMPUTE"),
    ("h3_offload", "EXP_MOE_H3_OFFLOAD"),
    ("moe_tiles", "EXP_MOE_TILES"),
)


def import_patches() -> None:
    """Import every patch module present, in order; each applies itself if enabled."""
    for name, env in _PATCHES:
        module = f"{__package__}.{name}"
        if importlib.util.find_spec(module) is None:
            if os.environ.get(env, "").strip() not in ("", "0"):
                raise ValueError(f"{env} is set but {module} is not in this checkout")
            continue
        importlib.import_module(module)


if __name__ == "__main__":
    import_patches()

    from torchtitan.train import main

    main()
