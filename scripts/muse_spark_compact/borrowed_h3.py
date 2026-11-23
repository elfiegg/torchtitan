"""Selective H3 copy ordering for one training slot on one expert-parallel rank."""

from dataclasses import dataclass


@dataclass
class ForwardH3:
    layer: int
    planner_address: int
    raw: object
    host: object | None = None
    d2h_done: object | None = None


class BorrowedH3:
    """Hold each forward's identity, including layers that keep stock replay."""

    def __init__(self, io, stages, host_buffers, layers, *, probe_layer=None):
        if len(stages) != 2 or any(layer < 0 or layer >= layers - 1 for layer in host_buffers):
            raise ValueError("Expected two private stages and an unselected last layer")
        if probe_layer is not None and probe_layer not in host_buffers:
            raise ValueError("The replay parity probe needs a retained host snapshot")
        self.io = io
        self.stages = stages
        self.host_buffers = host_buffers
        self.layers = layers
        self.stack = []
        self.combine_read_done = None
        self.stage_read_done = [None, None]
        self.next_stage = 0
        self.reversing = False
        self.backward_frame = None
        self.prefetched_frame = None
        self.prefetch_done = None
        self.completed_schedules = 0
        self.probe_layer = probe_layer
        self.probe_result = None

    def receipt(self):
        if (not self.completed_schedules or self.stack or self.reversing
                or self.backward_frame is not None or self.prefetched_frame is not None):
            raise RuntimeError("Host H3 receipt requires a complete drained schedule")
        result = {
            "constructed_schedules": self.completed_schedules,
            "layers": self.layers,
            "selected_zero_based_layers": sorted(self.host_buffers),
            "host_snapshots_per_schedule": len(self.host_buffers),
            "prefetched_restores_per_training_schedule": len(self.host_buffers),
        }
        if self.probe_layer is not None:
            result["warmup_replay_parity"] = {"layer": self.probe_layer, **self.probe_result}
        return result

    def probes(self, frame):
        return frame.layer == self.probe_layer and self.completed_schedules == 0

    def check_replay(self, frame, replayed, expert_ids):
        if frame is not self.backward_frame or not self.probes(frame) or self.probe_result is not None:
            raise RuntimeError("Host H3 parity check ran outside the selected warmup layer")
        self.probe_result = self.io.check_replay(frame.host, replayed, expert_ids, frame.d2h_done)

    def before_forward(self):
        if self.reversing:
            raise RuntimeError("Host H3 requires all forwards before their backwards")
        if len(self.stack) == self.layers:
            raise RuntimeError("Host H3 forward count exceeds the configured slot")
        # The next input publication barrier is after this stream wait. Every
        # peer's next W2 must wait for each owner's previous combine read.
        if self.combine_read_done is not None:
            self.io.wait_compute(self.combine_read_done)
            self.combine_read_done = None

    def saves_next_forward(self):
        return len(self.stack) in self.host_buffers

    def after_forward(self, planner_address, raw):
        layer = len(self.stack)
        frame = ForwardH3(layer, planner_address, raw, self.host_buffers.get(layer))
        if frame.host is not None:
            bank = self.next_stage
            self.next_stage = 1 - bank
            stage = self.stages[bank]
            self.combine_read_done = self.io.copy(
                "device_stage", stage, raw, self.io.compute_done(), self.stage_read_done[bank],
            )
            frame.d2h_done = self.io.copy("host", frame.host, stage, self.combine_read_done)
            self.stage_read_done[bank] = frame.d2h_done
        self.stack.append(frame)
        return frame

    def before_backward(self, frame, planner_address):
        if self.backward_frame is not None or not self.stack or frame is not self.stack[-1]:
            raise RuntimeError("Host H3 expected the most recent unmatched forward layer")
        if frame.planner_address != planner_address:
            raise RuntimeError("Host H3 planner identity changed between forward and backward")
        if not self.reversing:
            if len(self.stack) != self.layers:
                raise RuntimeError("Host H3 backward began before the forward slot was complete")
            # Backward scratch reuses both forward stages starting at the
            # first layer, even when that layer never offloaded H3.
            for mark in self.stage_read_done:
                if mark is not None:
                    self.io.wait_compute(mark)
            self.stage_read_done = [None, None]
            self.combine_read_done = None
            self.next_stage = 0
            self.reversing = True
        self.backward_frame = frame

    def restore(self, frame):
        if frame is not self.backward_frame or frame is not self.prefetched_frame:
            raise RuntimeError("Host H3 restore lacks an immediately preceding EP prefetch")
        self.io.wait_compute(self.prefetch_done)
        self.prefetched_frame = None
        self.prefetch_done = None
        return frame.raw

    def after_backward_barrier(self, frame):
        if frame is not self.backward_frame or self.prefetched_frame is not None:
            raise RuntimeError("Host H3 backward advanced before its prefetched data was consumed")
        if self.probes(frame) and self.probe_result is None:
            raise RuntimeError("Host H3 backward advanced before checking replay parity")
        self.stack.pop()
        self.backward_frame = None
        self.reversing = bool(self.stack)
        if self.stack and self.stack[-1].host is not None and not self.probes(self.stack[-1]):
            next_frame = self.stack[-1]
            # The final EP barrier makes every peer's current W2 DGRAD read
            # complete. An intervening stock backward would reuse combine.
            self.prefetch_done = self.io.copy(
                "restore", next_frame.raw, next_frame.host,
                self.io.compute_done(), next_frame.d2h_done,
            )
            self.prefetched_frame = next_frame
        elif not self.stack:
            self.completed_schedules += 1
