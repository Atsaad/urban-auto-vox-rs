#!/usr/bin/env python3
"""Sample peak memory while the voxelisation pipeline runs.

The pipeline's work happens inside containers, so process RSS of the driving
script measures nothing. This polls Docker for per-container memory and
nvidia-smi for device memory, and writes one row per sample. Peak memory for
the throughput table is the maximum over the run; the stage it belongs to is
recovered by joining the timestamps against the run log.
"""
import csv, subprocess, sys, time

OUT = sys.argv[1] if len(sys.argv) > 1 else "memsample.csv"
EVERY = float(sys.argv[2]) if len(sys.argv) > 2 and OUT != "--summary" else 2.0


def docker_mem():
    """(name, mib) for each running container, or [] if docker is unavailable."""
    try:
        p = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    rows = []
    for line in p.stdout.splitlines():
        if "\t" not in line:
            continue
        name, usage = line.split("\t", 1)
        used = usage.split("/")[0].strip()          # e.g. "1.234GiB"
        for suf, mul in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024), ("B", 1 / 1048576)):
            if used.endswith(suf):
                try:
                    rows.append((name, float(used[:-len(suf)]) * mul))
                except ValueError:
                    pass
                break
    return rows


def gpu_mem():
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return float(p.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


def summarise(path):
    """Peaks over a sample file. Tolerates a short final row: the sampler is
    killed when the run ends, which can truncate the line it was writing."""
    rows = [r for r in csv.DictReader(open(path))]
    def peak(k):
        v = [float(r[k]) for r in rows if r.get(k) not in (None, "", "nan")]
        return max(v) if v else float("nan")
    print(f"samples        {len(rows)}")
    print(f"peak GPU       {peak('gpu_mib'):.0f} MiB")
    print(f"peak container {peak('containers_mib'):.0f} MiB (all containers)")
    print(f"peak single    {peak('top_mib'):.0f} MiB")


def main():
    if OUT == "--summary":
        return summarise(sys.argv[2])
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "gpu_mib", "containers_mib", "n_containers", "top_container", "top_mib"])
        while True:
            d = docker_mem()
            total = sum(m for _, m in d)
            top = max(d, key=lambda r: r[1]) if d else ("", 0.0)
            w.writerow([f"{time.time():.1f}", f"{gpu_mem():.0f}", f"{total:.1f}",
                        len(d), top[0], f"{top[1]:.1f}"])
            fh.flush()
            time.sleep(EVERY)


if __name__ == "__main__":
    main()
