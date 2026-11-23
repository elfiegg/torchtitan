"""Opt-in source prototype for known fixed-capacity compact pinned host buffers."""

import inspect
import textwrap


TOKENS = 29_184
WIDTH = 8192
KV_WIDTH = 16 * 64
VOCAB_ROUTER = 256
ATTENTION_HEADS = 128
EXPERT_WIDTH = 4096
BYTES = {
    "block input": TOKENS * WIDTH * 4,
    "QKV projection": TOKENS * (WIDTH + 2 * KV_WIDTH) * 2,
    "attention output": TOKENS * WIDTH * 2,
    "attention LSE": TOKENS * ATTENTION_HEADS * 4,
    "attention output gate": TOKENS * WIDTH * 2,
    "attention output projection": TOKENS * WIDTH * 2,
    "router gate": TOKENS * VOCAB_ROUTER * 4,
    "pooler": TOKENS * EXPERT_WIDTH * 2,
    "unpooler": TOKENS * WIDTH * 2,
}
EXACT = frozenset(BYTES.values())


def capacity_for(nbytes):
    if nbytes in EXACT:
        return nbytes
    return 1 << max(nbytes - 1, 0).bit_length()


def exact_acquire(original, source=None):
    """Change only the bucket key/size in the pinned acquire implementation."""
    if source is None:
        source = inspect.getsource(original)
    source = textwrap.dedent(source)
    anchor = "    capacity = 1 << max(nbytes - 1, 0).bit_length()\n"
    if source.count(anchor) != 1:
        raise RuntimeError("Pinned TorchTitan host pool capacity expression changed")
    patched = source.replace(anchor, "    capacity = _muse_exact_capacity_for(nbytes)\n")
    namespace = {**original.__globals__, "_muse_exact_capacity_for": capacity_for}
    exec(compile(patched, "<compact-exact-pinned-pool>", "exec"), namespace)
    return namespace["acquire"]


def install():
    """Call only before constructing the pinned 29,184-token benchmark model."""
    from torchtitan.distributed.activation_offload import _PinnedBufferPool

    _PinnedBufferPool.acquire = exact_acquire(_PinnedBufferPool.acquire)
