"""Recipe-specific typed route kernels for offline scheduling and resource review."""

import triton
import triton.language as tl


PLANNER_ALIGNMENT = tl.constexpr(128)


@triton.jit
def route_forward(Raw, Gain, Scores, ExpertIds, Output, D: tl.constexpr, K: tl.constexpr,
                  EPS: tl.constexpr, GAIN_CENTER: tl.constexpr):
    """Normalize each route with learned input scale and reduce its BF16 result."""
    token = tl.program_id(0)
    routes = token * K + tl.arange(0, K)
    cols = tl.arange(0, D)
    valid = tl.load(ExpertIds + routes) >= 0
    raw = tl.load(Raw + routes[:, None] * D + cols[None, :], valid[:, None], 0).to(tl.float32)
    gain = tl.load(Gain + cols).to(tl.float32) + GAIN_CENTER
    scaled = raw * gain[None, :]
    rstd = tl.rsqrt(tl.sum(scaled * scaled, 1) * (1.0 / D) + EPS)
    normalized = (scaled * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    score = tl.load(Scores + routes, valid, 0).to(tl.float32)
    result = tl.sum(tl.where(valid[:, None], normalized * score[:, None], 0.0), 0)
    tl.store(Output + token * D + cols, result)


@triton.jit
def route_backward(Raw, SavedBytes, SavedOffset, NeedReplay, Upstream, Gain, Scores, ExpertIds,
                   PublishedGradient, ScoreGradient, GainPartials, PartialsOffset, ROUTES: tl.constexpr,
                   D: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr,
                   EPS: tl.constexpr, GAIN_CENTER: tl.constexpr):
    """Reverse normalized BF16 route weighting, with one owner per raw route row."""
    block = tl.program_id(0)
    rows = block * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    offsets = rows[:, None] * D + cols[None, :]
    present = rows < ROUTES
    valid = tl.load(ExpertIds + rows, present, -1) >= 0
    raw_base = Raw
    if tl.load(NeedReplay) == 0:
        offset = tl.multiple_of(tl.load(SavedOffset), PLANNER_ALIGNMENT)
        raw_base = (SavedBytes + offset).to(tl.pointer_type(tl.bfloat16))
    raw = tl.load(raw_base + offsets, valid[:, None], 0).to(tl.float32)
    dy = tl.load(Upstream + (rows[:, None] // K) * D + cols[None, :], valid[:, None], 0).to(tl.float32)
    score = tl.load(Scores + rows, valid, 0).to(tl.float32)
    gain = tl.load(Gain + cols).to(tl.float32) + GAIN_CENTER

    scaled = raw * gain[None, :]
    rstd = tl.rsqrt(tl.sum(scaled * scaled, 1) * (1.0 / D) + EPS)
    unit = scaled * rstd[:, None]
    rounded_unit = unit.to(tl.bfloat16).to(tl.float32)
    # Match the standalone score dot's four 1,024-channel groups at width 4,096.
    products = tl.reshape(rounded_unit * dy, (ROWS, 4, 1024))
    score_tiles = tl.sum(products, 2)
    dscore = tl.full((ROWS, 1), 0.0, tl.float32)
    for tile in tl.static_range(4):
        column = tl.full((ROWS, 1), tile, tl.int32)
        dscore += tl.gather(score_tiles, column, 1)
    dscore = tl.reshape(dscore, (ROWS,))

    dunit = (dy * score[:, None]).to(tl.bfloat16).to(tl.float32)
    projection = tl.sum(dunit * unit, 1)
    dscaled = (dunit - (unit * (1.0 / D)) * projection[:, None]) * rstd[:, None]
    draw = dscaled * gain[None, :]
    dgain_lane = dscaled.to(tl.bfloat16).to(tl.float32) * raw
    dgain = tl.sum(tl.where(valid[:, None], dgain_lane, 0.0), 0)

    partials = _gain_partials(GainPartials, PartialsOffset)
    tl.store(partials + block * D + cols, dgain)
    tl.store(ScoreGradient + rows, tl.where(valid, dscore, 0.0), present)
    # PublishedGradient may be the raw combine buffer. Finish all loads first.
    tl.debug_barrier()
    tl.store(PublishedGradient + offsets, tl.where(valid[:, None], draw, 0.0), present[:, None])


@triton.jit
def _gain_partials(Base, Offset):
    if Offset is None:
        return Base
    offset = tl.multiple_of(tl.load(Offset), PLANNER_ALIGNMENT)
    return (Base + offset).to(tl.pointer_type(tl.float32))


@triton.jit
def reduce_gain_tiles(Base, Offset, COUNT: tl.constexpr, D: tl.constexpr,
                      BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """Reduce gain partials into a smaller region immediately after the source region."""
    bank = _gain_partials(Base, Offset)
    block = tl.program_id(0)
    rows = block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    x = tl.load(bank + rows[:, None] * D + cols[None, :], rows[:, None] < COUNT, 0)
    tl.store(bank + (COUNT + block) * D + cols, tl.sum(x, 0))


@triton.jit
def finish_gain(Base, Offset, GainGradient, COUNT: tl.constexpr, TILES: tl.constexpr,
                D: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """Write the BF16 gradient before the next operator reuses the scratch bank."""
    bank = _gain_partials(Base, Offset)
    rows = tl.arange(0, BLOCK_R)
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    x = tl.load(bank + (COUNT + rows[:, None]) * D + cols[None, :], rows[:, None] < TILES, 0)
    tl.store(GainGradient + cols, tl.sum(x, 0))
