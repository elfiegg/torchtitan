"""Read host pin and NUMA accounting outside scored benchmark updates."""

import os
from pathlib import Path
import re
import resource


_USER_MEMORY_FIELDS = ("VmRSS", "VmHWM", "VmLck", "VmPin", "Mems_allowed_list")


def _read(path):
    try:
        return path.read_text()
    except OSError:
        return None


def _process_status(proc):
    status = _read(proc / "status")
    if status is None:
        return None
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    return {key: fields[key].strip() for key in _USER_MEMORY_FIELDS if key in fields}


def _process_pages(proc):
    mappings = _read(proc / "numa_maps")
    if mappings is None:
        return None
    resident = {}
    page_size_default = os.sysconf("SC_PAGE_SIZE")
    for line in mappings.splitlines():
        page_size = re.search(r"(?:^| )kernelpagesize_kB=(\d+)", line)
        page_bytes = int(page_size[1]) * 1024 if page_size else page_size_default
        for node, pages in re.findall(r"(?:^| )N(\d+)=(\d+)", line):
            resident[node] = resident.get(node, 0) + int(pages) * page_bytes
    return resident


def _mount_field(encoded):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), encoded)


def _cgroup_memory(proc):
    groups, mounts = _read(proc / "cgroup"), _read(proc / "mountinfo")
    if groups is None or mounts is None:
        return None
    group = next((Path(line[3:]) for line in groups.splitlines() if line.startswith("0::")), None)
    if group is None:
        return None

    for line in mounts.splitlines():
        mount, _, fs = line.partition(" - ")
        if not fs.startswith("cgroup2 "):
            continue
        fields = mount.split()
        root = Path(_mount_field(fields[3]))
        base = Path(_mount_field(fields[4]))
        if not group.is_relative_to(root):
            continue
        leaf = base / group.relative_to(root)
        reports = []
        for directory in (leaf, *leaf.parents):
            if not directory.is_relative_to(base):
                break
            counters = {}
            for field in ("current", "peak", "max", "high", "events", "numa_stat"):
                raw = _read(directory / f"memory.{field}")
                if raw is not None:
                    counters[field] = raw.strip()
            if counters:
                reports.append({"cgroup_visible_path": str(directory), **counters})
        if reports:
            return reports
    return None


def snapshot(torch, host_buffers=(), *, proc="/proc/self"):
    """Observe real host-resident storage without allocating or timing transfers."""
    proc = Path(proc)
    host = torch.cuda.memory.host_memory_stats()
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    return {
        "pinned_allocator_bytes": {
            key: value for key, value in host.items()
            if key.startswith(("allocated_bytes.", "active_bytes."))
        },
        "snapshot_count": len(host_buffers),
        "snapshot_payload_bytes": sum(t.numel() * t.element_size() for t in host_buffers),
        "snapshots_report_pinned": all(t.is_pinned() for t in host_buffers) if host_buffers else None,
        "process_memory": _process_status(proc),
        "process_resident_bytes_by_numa_node": _process_pages(proc),
        "cgroup_memory": _cgroup_memory(proc),
        "memlock_rlimit_bytes": {"soft": soft, "hard": hard},
    }
