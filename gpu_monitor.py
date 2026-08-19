#!/usr/bin/env python
"""Real-time utilization monitor for Intel Arc B-series GPUs (xe driver).

Works without xpu-smi / Sysman by reading the xe kernel driver's sysfs
counters:
  - busy %   : delta of gtidle/idle_residency_ms per GT; if that counter
               never advances (GT stuck in gt-c0, no RC6), falls back to a
               frequency-based estimate from freq0/cur_freq
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


def driver_of(card):
    path = os.path.join(card, "device", "driver")
    try:
        return os.path.basename(os.path.realpath(path))
    except OSError:
        return None


def find_cards(device_id=None):
    cards = []
    for card in sorted(glob.glob("/sys/class/drm/card*")):
        if not os.path.isdir(os.path.join(card, "device")):
            continue
        if device_id:
            if read(os.path.join(card, "device", "device")) == device_id:
                cards.append(card)
        elif driver_of(card) == "xe":
            cards.append(card)
    return cards


def card_bdf(card):
    path = os.path.realpath(os.path.join(card, "device"))
    name = os.path.basename(path)
    return name if name.startswith("0000:") else None


def xpu_index_map():
    """Map PCI BDF -> torch xpu index, via the Level-Zero UUID (embeds bus:device)."""
    try:
        import torch

        m = {}
        for i in range(torch.xpu.device_count()):
            bd = str(torch.xpu.get_device_properties(i).uuid).split("-")[3]
            m[f"0000:{bd[:2]}:{bd[2:]}.0"] = i
        return m
    except Exception:
        return {}


class Gt:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.idle_path = os.path.join(path, "gtidle", "idle_residency_ms")
        self.cur_path = os.path.join(path, "freq0", "cur_freq")
        self.act_path = os.path.join(path, "freq0", "act_freq")
        self.min_freq = int(read(os.path.join(path, "freq0", "min_freq")) or 0)
        self.max_freq = int(read(os.path.join(path, "freq0", "max_freq")) or 0)
        self.last_idle = None
        self.last_t = None
        self.samples = 0
        self.frozen = False

    def freq_at_floor(self, cur):
        try:
            return int(cur) <= self.min_freq
        except (TypeError, ValueError):
            return False

    def freq_busy(self, cur):
        if self.max_freq <= self.min_freq:
            return None
        try:
            c = int(cur)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(100.0, 100 * (c - self.min_freq) / (self.max_freq - self.min_freq)))

    def sample(self):
        now = time.monotonic()
        idle = int(read(self.idle_path) or 0)
        cur = read(self.cur_path)
        act = read(self.act_path)
        busy = None
        if self.last_idle is not None:
            dt = now - self.last_t
            idle_delta = idle - self.last_idle
            self.samples += 1
            if self.frozen:
                busy = self.freq_busy(cur)
            elif idle_delta > 0:
                busy = max(0.0, min(100.0, 100 * (1 - idle_delta / (dt * 1000))))
            else:
                if self.samples == 1 and self.freq_at_floor(cur):
                    self.frozen = True
                    busy = self.freq_busy(cur)
                else:
                    busy = 100.0
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
    known_series = {"60": "0xe211", "70": "0xe223"}
    p = argparse.ArgumentParser(description="Intel Arc (xe) real-time GPU monitor")
    p.add_argument("--interval", type=float, default=2.0, help="sample interval in seconds")
    p.add_argument("--count", type=int, default=0, help="number of samples (0 = infinite)")
    p.add_argument(
        "-B",
        "--series",
        choices=sorted(known_series),
        help="Arc B-series model (maps to its PCI device id): %(choices)s",
    )
    p.add_argument("--device-id", default=None, help="PCI device id to match (default: auto-detect all Intel Arc/xe cards)")
    args = p.parse_args()

    device_id = known_series[args.series] if args.series else args.device_id
    cards = find_cards(device_id)
    if not cards:
        if device_id:
            sys.exit(f"no card found matching device id {device_id}")
        sys.exit("no Intel Arc (xe driver) card found")

    xpu_map = xpu_index_map()
    monitors = []
    for card in cards:
        dev = os.path.join(card, "device")
        gts = []
        for tile in sorted(glob.glob(os.path.join(dev, "tile*"))):
            for gt in sorted(glob.glob(os.path.join(tile, "gt*"))):
                gts.append(Gt(gt, os.path.basename(gt)))
        hwmon = glob.glob(os.path.join(dev, "hwmon", "hwmon*"))
        power = Power(hwmon[0]) if hwmon else None
        monitors.append((os.path.basename(card), xpu_map.get(card_bdf(card)), gts, power))

    print(f"monitoring {len(monitors)} Intel Arc card(s)")
    for _, _, gts, power in monitors:
        for gt in gts:
            gt.sample()
        if power:
            power.sample()

    ncols = max((len(m[2]) for m in monitors), default=1)
    labels = []
    for _, _, _, power in monitors:
        if power:
            labels = list(power.labels.values())
            break

    def fmt_busy(b):
        return f"{b:5.1f}%" if b is not None else "  n/a"

    def fmt_freq(f):
        return f"{f:>5s}" if f is not None else "  n/a"

    header = ["card", "xpu"]
    for gi in range(ncols):
        header += [f"gt{gi}busy", "gt" + str(gi) + "MHz"]
    header += [lbl for lbl in labels] + ["pkg(C)", "vram(C)"]
    header = " ".join(f"{h:>6s}" for h in header)
    print(header)
    print("-" * len(header))

    n = 0
    warned = False
    while args.count == 0 or n < args.count:
        time.sleep(args.interval)
        n += 1
        if not warned and any(gt.frozen for _, _, gts, _ in monitors for gt in gts):
            warned = True
            print("note: gtidle idle_residency counter is not advancing; busy% is estimated from cur_freq", flush=True)
        for card, xidx, gts, power in monitors:
            cells = [f"{card:>6s}", f"{'xpu' + str(xidx) if xidx is not None else 'n/a':>6s}"]
            for gt in gts:
                busy, cur, act = gt.sample()
                cells.append(fmt_busy(busy))
                cells.append(fmt_freq(cur))
            if power:
                w = power.sample()
                cells += [f"{w.get(lbl, 0):6.1f}" for lbl in labels]
                t = power.temps()
                cells += [f"{t.get('pkg', float('nan')):6.1f}", f"{t.get('vram', float('nan')):6.1f}"]
            print(" ".join(cells), flush=True)


if __name__ == "__main__":
    main()
