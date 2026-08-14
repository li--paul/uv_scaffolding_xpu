#!/usr/bin/env python
"""Real-time utilization monitor for Intel Arc B-series GPUs (xe driver).

Works without xpu-smi / Sysman by reading the xe kernel driver's sysfs
counters:
  - busy %   : delta of gtidle/idle_residency_ms per GT
  - frequency: freq0/cur_freq, freq0/act_freq
  - power    : hwmon energy*_input delta (labels "card"/"pkg")
  - temp     : hwmon temp*_input (labels "pkg"/"vram")
"""

import argparse
import glob
import os
import sys
import time


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def find_card(device_id=None):
    for card in sorted(glob.glob("/sys/class/drm/card*")):
        if not os.path.isdir(os.path.join(card, "device")):
            continue
        if read(os.path.join(card, "device", "device")) == device_id:
            return card
    return None


class Gt:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.idle_path = os.path.join(path, "gtidle", "idle_residency_ms")
        self.cur_path = os.path.join(path, "freq0", "cur_freq")
        self.act_path = os.path.join(path, "freq0", "act_freq")
        self.last_idle = None
        self.last_t = None

    def sample(self):
        now = time.monotonic()
        idle = int(read(self.idle_path) or 0)
        cur = read(self.cur_path)
        act = read(self.act_path)
        busy = None
        if self.last_idle is not None:
            dt = now - self.last_t
            busy = max(0.0, min(100.0, 100 * (1 - (idle - self.last_idle) / (dt * 1000))))
        self.last_idle = idle
        self.last_t = now
        return busy, cur, act


class Power:
    def __init__(self, base):
        self.base = base
        self.energies = {}
        self.labels = {}
        for path in glob.glob(os.path.join(base, "energy*_input")):
            key = path.rsplit("/", 1)[1].replace("_input", "")
            label = read(os.path.join(base, f"{key}_label")) or key
            self.labels[key] = label
            self.energies[key] = int(read(path) or 0)
        self.last_t = time.monotonic()

    def sample(self):
        now = time.monotonic()
        dt = now - self.last_t
        watts = {}
        for key, label in self.labels.items():
            e = int(read(os.path.join(self.base, f"{key}_input")) or 0)
            watts[label] = (e - self.energies[key]) / (dt * 1e6) if dt > 0 else 0.0
            self.energies[key] = e
        self.last_t = now
        return watts

    def temps(self):
        out = {}
        for path in glob.glob(os.path.join(self.base, "temp*_input")):
            label = read(os.path.join(self.base, path.replace("_input", "_label"))) or ""
            if label in ("pkg", "vram"):
                v = read(path)
                if v:
                    out[label] = int(v) / 1000.0
        return out


def main():
    p = argparse.ArgumentParser(description="Intel Arc (xe) real-time GPU monitor")
    p.add_argument("--interval", type=float, default=2.0, help="sample interval in seconds")
    p.add_argument("--count", type=int, default=0, help="number of samples (0 = infinite)")
    p.add_argument("--device-id", default="0xe223", help="PCI device id to match")
    args = p.parse_args()

    card = find_card(args.device_id)
    if not card:
        sys.exit(f"no card found matching device id {args.device_id}")
    dev = os.path.join(card, "device")
    name = read(os.path.join(dev, "subsystem_device")) or read(os.path.join(dev, "device"))
    print(f"monitoring {card}  (pci {args.device_id})")

    gts = []
    for tile in sorted(glob.glob(os.path.join(dev, "tile*"))):
        for gt in sorted(glob.glob(os.path.join(tile, "gt*"))):
            gts.append(Gt(gt, os.path.basename(gt)))

    hwmon = glob.glob(os.path.join(dev, "hwmon", "hwmon*"))
    power = Power(hwmon[0]) if hwmon else None

    for gt in gts:
        gt.sample()
    if power:
        power.sample()

    header = "  ".join(
        [f"{gt.name:>6s}busy  freq(MHz)" for gt in gts]
        + [f"{lbl:>7s}W" for lbl in (power.labels.values() if power else [])]
        + ["pkg(C)", "vram(C)"]
    )
    print(header)
    print("-" * len(header))

    n = 0
    while args.count == 0 or n < args.count:
        time.sleep(args.interval)
        n += 1
        parts = []
        for gt in gts:
            busy, cur, act = gt.sample()
            b = f"{busy:5.1f}%" if busy is not None else "  n/a"
            freq = cur if cur is not None else "?"
            parts.append(f"{gt.name:>6s} {b:>6s} {freq:>8s}")
        if power:
            w = power.sample()
            parts.append("  ".join(f"{w.get(lbl, 0):7.1f}" for lbl in power.labels.values()))
            t = power.temps()
            parts.append(f"{t.get('pkg', float('nan')):7.1f} {t.get('vram', float('nan')):7.1f}")
        print("  ".join(parts), flush=True)


if __name__ == "__main__":
    main()
