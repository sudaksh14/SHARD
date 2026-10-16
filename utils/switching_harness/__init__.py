"""
Switching harness: measurement and verification tools around the Delta Runtime Manager (DRM).

The DRM itself lives in `networks/level_delta_utils.py` (`InMemoryDeltaManager`, `FileDeltaManager`)
and in the `export_level_delta` / `apply_level_delta_*` methods of the SHARD modules.  This package
does not re-implement any of it; it drives the real classes.

    python -m utils.switching_harness verify            # CPU: delta-managed model == SHARD model, per level
    python -m utils.switching_harness delta      ...    # level-to-level switching latency with delta files
    python -m utils.switching_harness full-model ...    # baseline: load an independent full model per level
"""
