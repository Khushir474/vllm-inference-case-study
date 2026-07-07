"""Snapshot vLLM's own Prometheus /metrics endpoint around a benchmark run.

This captures internals hosted APIs don't expose — KV cache utilization and
request queue depth straight from vLLM's PagedAttention scheduler — so we can
explain *why* throughput plateaus at a given concurrency level, not just that
it does.
"""

import argparse
import json
import re
import time
import urllib.request

METRIC_PATTERN = re.compile(r'^vllm:(\w+)\{[^}]*\}\s+([0-9.eE+-]+)\s*$', re.MULTILINE)


def fetch_metrics(base_url: str) -> dict[str, float]:
    url = base_url.rstrip("/v1").rstrip("/") + "/metrics"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode()
    values: dict[str, list[float]] = {}
    for name, value in METRIC_PATTERN.findall(text):
        values.setdefault(name, []).append(float(value))
    # Multiple lines can share a metric name if it's exposed per-label; sum them
    # (e.g. num_requests_waiting_by_reason) or just keep the single gauge value.
    return {name: sum(vals) for name, vals in values.items()}


def poll_loop(base_url: str, out_path: str, interval_s: float, stop_flag_path: str):
    with open(out_path, "w") as f:
        while True:
            try:
                m = fetch_metrics(base_url)
                m["_ts"] = time.time()
                f.write(json.dumps(m) + "\n")
                f.flush()
            except Exception as e:
                f.write(json.dumps({"_ts": time.time(), "_error": str(e)}) + "\n")
                f.flush()
            try:
                with open(stop_flag_path) as sf:
                    if sf.read().strip() == "stop":
                        break
            except FileNotFoundError:
                pass
            time.sleep(interval_s)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snapshot")
    snap.add_argument("base_url")

    poll = sub.add_parser("poll")
    poll.add_argument("base_url")
    poll.add_argument("--out", required=True)
    poll.add_argument("--interval", type=float, default=2.0)
    poll.add_argument("--stop-flag", required=True)

    args = parser.parse_args()
    if args.cmd == "snapshot":
        print(json.dumps(fetch_metrics(args.base_url)))
    elif args.cmd == "poll":
        poll_loop(args.base_url, args.out, args.interval, args.stop_flag)


if __name__ == "__main__":
    main()
