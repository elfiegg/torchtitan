"""Validate raw BF16 H3 words without allocating a second device snapshot."""


def compare_valid_words(torch, snapshot, replayed, expert_ids, width, *, chunk_rows=4096):
    if replayed.dtype != torch.bfloat16 or snapshot.dtype != torch.uint8 or chunk_rows < 1:
        raise ValueError("Host H3 replay probe requires BF16 values and their original byte snapshot")
    valid = (expert_ids.reshape(-1) >= 0).cpu()
    expected = snapshot.view(torch.int16).view(-1, width)
    observed = replayed.view(torch.int16).view(-1, width)
    if expected.shape != observed.shape or observed.shape[0] != valid.numel():
        raise ValueError("Host H3 replay probe requires matching compact BF16 route storage")
    mismatches = 0
    rows = int(valid.sum())
    if not rows:
        raise ValueError("Host H3 replay probe requires at least one valid route")
    for start in range(0, valid.numel(), chunk_rows):
        stop = min(start + chunk_rows, valid.numel())
        mask = valid[start:stop, None]
        different = observed[start:stop].cpu() != expected[start:stop]
        mismatches += int((different & mask).sum())
    if mismatches:
        raise RuntimeError(f"Host H3 replay differs in {mismatches} of {rows * width} valid BF16 words")
    return {"valid_routes": rows, "matched_words": rows * width}
