"""Read the Dist-MoE byte counters the runtime already transfers during warmup."""

from dist_moe._activation_buffer import ActivationBuffer


snapshots = []
original = ActivationBuffer._scratch_info_from_peaks


def record(self, peak_values):
    info = original(self, peak_values)
    if self.num_activation_slots != 1:
        return info
    snapshots.append({
        "physical_slot_bytes": self.activation_slot_bytes,
        "peak_used_slot_bytes": self.activation_slot_bytes - peak_values[0],
        "scratch_bytes": self.scratch_region_size,
        "peak_used_scratch_bytes": self.scratch_region_size - peak_values[1],
    })
    if len(snapshots) == 3:
        ActivationBuffer._scratch_info_from_peaks = original
    return info


def install():
    ActivationBuffer._scratch_info_from_peaks = record
