import sys

USAGE = """usage: python -m utils.switching_harness {verify|delta|full-model} [options]
  verify      correctness of delta switching vs. the SHARD model (CPU friendly)
  delta       level-to-level switching latency with delta files
  full-model  baseline: load an independent full model per level
Use `python -m utils.switching_harness <command> --help` for options."""


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("verify", "delta", "full-model"):
        sys.exit(USAGE)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "verify":
        from utils.switching_harness.verify import main as run
    elif cmd == "delta":
        from utils.switching_harness.delta_switching import main as run
    else:
        from utils.switching_harness.full_model_switching import main as run
    result = run(rest)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
