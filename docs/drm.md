# Delta Runtime Manager (DRM)

In the code base the DRM is the **delta manager** family in [`networks/level_delta_utils.py`](../networks/level_delta_utils.py)
(`BaseDeltaManager`, `InMemoryDeltaManager`, `FileDeltaManager`) together with the per-module
`export_level_delta` / `apply_level_delta_up` / `apply_level_delta_down` methods of the SHARD modules
([`shard_modules/`](../shard_modules), [`networks/shard_model.py`](../networks/shard_model.py)). There is no class literally called
"DRM"; the paper's term maps to these managers.

## The problem

Keeping one independently trained model per operating point on an edge device multiplies storage, and switching between
them means loading a complete model (hundreds of ms on a Jetson Orin).
Because SHARD levels are nested, going from level *i* to level *j* only needs the parameters in which they differ.

## Terminology

| Term | Meaning |
|---|---|
| level / sub-transformer | one nested operating point (embedding width `D_i`), see [architecture.md](architecture.md) |
| SHARD model | holds the full set of nested tensors and can run any level by slicing (`set_level_use`) - training/evaluation |
| regular (base) model | an ordinary `VisionTransformer` at *one* level - what is deployed / exported to ONNX |
| shared vs. level-specific parameters | the leading slice of every tensor is shared by all wider levels; the *additional* rows/columns/segments needed to reach level *i* from *i-1* are level-specific |
| `UpDelta` | the new tensor pieces (and bias segments, token/embedding columns) that must be added to a level *i-1* regular model |
| `DownDelta` | the metadata (target sizes) needed to cut a level *i* regular model back to level *i-1* - no tensor data |

## Delta generation and storage

For every module, `export_level_delta()` called at level *i* returns `(DownDelta, UpDelta)` (e.g. `shard_modules/linear.py`):
`UpDelta = (lower_part, right_part, bias_part)` - the new columns `[:out_i, in_{i-1}:in_i]`, the new rows `[out_{i-1}:out_i, :in_{i-1}]`
and the new bias segment; `DownDelta = (in_i, out_i)`. `ShardModel.export_level_delta` collects them module by module
(deltas of every submodule form one nested tuple, `networks/shard_deit_v3.py` additionally stores the hidden size).

`get_model_deltas(model)` loops over all levels and returns `{(level, up): delta}`.

* **`InMemoryDeltaManager(shard_model, starting_level)`** keeps all deltas as CPU tensors and a regular copy of the model at `starting_level`.
* **`FileDeltaManager.make_delta_file(file, shard_model, starting_level)`** serialises (with `torch.save`) one chunk for the regular model's
  state dict at `starting_level` followed by one chunk per delta: down deltas from `max-1` to `0`, then up deltas from `1` to `max`.
  A pickled list of byte offsets is written first (the "index"). Offline step; it takes a fraction of a second to a few seconds (not benchmarked here).
* **`FileDeltaManager(file, managed_config)`** reads the index, deserialises the base state dict into a regular model built from
  `managed_config` (`cfg.create_base_config(starting_level).no_prebuilt()`), and keeps the file handle. `get_level_delta(level, up)` seeks to the
  chunk and deserialises exactly that delta ("a single small random read"). `file_delta_manager(path, config)` is the context-manager form.

## Switching

```mermaid
sequenceDiagram
  participant App as application / policy
  participant M as FileDeltaManager
  participant F as delta file (disk)
  participant R as regular model (resident)
  App->>M: move_to(target)
  loop each intermediate level, one step at a time
    M->>F: seek + read chunk (level, up/down)
    F-->>M: delta (deserialised)
    M->>R: delta.apply(R)  (up: pad + copy new rows/cols; down: slice to smaller size)
  end
  M-->>App: R (now at target level)
```

`BaseDeltaManager.move_to(target)` applies the deltas of **every intermediate level** in sequence (up: `cur+1 .. target`;
down: `cur-1 .. target`), so a jump 0 -> 4 applies four up-deltas. Moving to the current level is a no-op. Any sequence of levels in
`[0, max_level]` is supported, in either direction; the managed model object stays the same (modified in place), so references held by the
application remain valid. Between switches no other state is kept besides `current_level` (and, for `FileDeltaManager`, the open file).

What happens to the active model state: parameter tensors of the regular model are replaced (`weight.data = ...`). *Up*: new, larger
tensors are allocated (`F.pad`/`torch.cat`) and filled with the old values plus the delta. *Down*: `weight.data = weight.data[:out, :in]` -
by code reading this is a **view** of the old tensor, so the old storage may stay allocated while it is referenced
(relevant for the memory interpretation of Fig. 5b). The managed model's gradients are zeroed on up-switches
(`model.zero_grad()`); inference mode is the intended use. Switching is not thread-safe and must not run concurrently with a forward pass.

## Correctness

`python -m utils.switching_harness verify` runs down-, up-, jump- and random-walk sequences and compares the regular model with the SHARD model
at every visited level (`rtol=1e-3, atol=1e-5`), for both managers. The same checks are in `tests/test_delta_manager.py`.

## Memory and latency: what is what

Three different costs must not be mixed up:

1. **Loading a complete independent checkpoint for another level** (baseline): `torch.load` of a whole model file -
   (mean 439.1 ms).
2. **Applying stored deltas** (DRM): read one or more chunks, deserialise, pad/copy tensors - `delta_switch_times_ms.csv`
   (mean 19.9 ms).
3. **Inference** at the new level: 20-36 ms ONNX Runtime on the Jetson (Table 3 / Fig. 4 data), independent of how the weights arrived.

Model *initialisation* (building the regular model, loading the base chunk) happens once when the manager is created and is not
part of the switching numbers. An ONNX graph is not rebuilt on a switch by the DRM; the Jetson benchmark exports each level to ONNX
separately (`utils/jetson_harness/benchmark_levels.py`).

Delta file size: for the released 5-level ViT-B (level-0 start) `make_delta_file` writes **364.5 MB**: base chunk 88.3 MB (22 M parameters, stored compactly),
up deltas 50.3 / 61.3 / 72.3 / 83.3 MB, down deltas 1.6-2.8 MB each; a plain level-4 state dict is 346 MB. The base state dict is cloned to tight storages before
serialisation (`_compact_state_dict`) because `make_base_copy()` hands out *views* of the full tensors and `torch.save` would otherwise write the entire storage behind each view.

## Using it (real API)

```python
import torch
from networks import level_delta_utils as delta
from utils.switching_harness import common

cfg = common.paper_config("shard_deit_v3")                         # ViTConfig_v3, 5 levels
shard = common.build_shard_model(cfg, "checkpoints/SHARD_deit_v3_5level.pt")  # strict state-dict load

# offline: write the delta file with a level-0 base model
with open("model.delta", "wb") as f:
    delta.FileDeltaManager.make_delta_file(f, shard, starting_level=0)

# on the device
with delta.file_delta_manager("model.delta", cfg.create_base_config(0).no_prebuilt()) as manager:
    model = manager.managed_model().eval()
    x = torch.randn(1, 3, 224, 224)
    for target in (4, 2, 0, 3):
        model = manager.move_to(target)          # applies the intermediate deltas, returns the same model object
        print(target, model(x).shape)
```

Runnable versions with the correctness check: [`examples/03_runtime_switching.py`](../examples/03_runtime_switching.py)
(`--tiny` needs no checkpoint). Timing: `python -m utils.switching_harness delta ...`.

## Three ways to change level (do not confuse them)

| | Switching between **pre-constructed levels** | **Applying deltas** (DRM) | **Loading an independent checkpoint** |
|---|---|---|---|
| what is resident | all levels' models/graphs | one regular model | one regular model |
| what moves | nothing (pick another object) | only the differing parameters | the complete model |
| storage | N full models (or SHARD model) | base + deltas (see sizes above) | N full models |
| implementation | `shard.set_level_use(i)` (SHARD model, slicing) / one ONNX session per level in `utils/jetson_harness` | `InMemoryDeltaManager`, `FileDeltaManager` | `python -m utils.switching_harness full-model` baseline |

## Limitations

* The deltas are specific to the SHARD module types registered in `LevelDeltas` (Linear, MultiheadAttention, LayerNorm, Conv2d, tokens, ...); new layer types need an `export_level_delta`/`apply_*` pair.
* The managed model is changed in place; the DRM does not decide *when* to switch (policy is out of scope, as in the paper).
* `FileDeltaManager` reads and deserialises the required chunks on every transition (no cache), which keeps the resident memory at one model.
* The "cold cache" variant (`utils/switching_harness/nocache.py`) is Linux-only.
* Dynamic batch sizes/resolutions are not handled by the DRM itself; the Jetson ONNX graphs are exported for batch size 1.
