# NVIDIA TAO DAFT — Converters

One concrete converter per `(source_format, target_format)` pair. Each
subclasses `BaseConverter` (in [`base.py`](base.py)), auto-registers itself,
and owns one nested CLI subcommand under `tao-daft convert`.

Directionality is a property of the pair, not of the registry. Most pairs are
uni-directional, data-centric (`metropolis-v3.0`) → training-centric
(`cosmos-reason-v1.0`, `tao-vl-reason-v1.0`). The `df-vlm-qa-v1.0` ↔
`tao-vl-reason-v1.0` pair runs both ways, because a correction batch is
produced *from* training data as well as consumed *into* it. The two
directions are not inverses — see the per-pair notes below.

See also: [CLI reference](../cli/README.md) · [validators](../validators/README.md) · [format specs](../formats/README.md)

---

## Registered pairs

| Source | Target | Module | Status |
|--------|--------|--------|--------|
| `metropolis-v3.0` | `cosmos-reason-v1.0` | [`pairs/metropolis_v3_0_to_cosmos_reason_v1_0.py`](pairs/metropolis_v3_0_to_cosmos_reason_v1_0.py) | 🚧 Active |
| `metropolis-v3.0` | `tao-vl-reason-v1.0` | [`pairs/metropolis_v3_0_to_tao_vl_reason_v1_0.py`](pairs/metropolis_v3_0_to_tao_vl_reason_v1_0.py) | 🚧 Active |
| `df-vlm-qa-v1.0` | `tao-vl-reason-v1.0` | [`pairs/df_vlm_qa_v1_0_to_tao_vl_reason_v1_0.py`](pairs/df_vlm_qa_v1_0_to_tao_vl_reason_v1_0.py) | 🚧 Active |
| `tao-vl-reason-v1.0` | `df-vlm-qa-v1.0` | [`pairs/tao_vl_reason_v1_0_to_df_vlm_qa_v1_0.py`](pairs/tao_vl_reason_v1_0_to_df_vlm_qa_v1_0.py) | 🚧 Active |

---

## `df-vlm-qa-v1.0` → `tao-vl-reason-v1.0`

Turns a corrected batch into training files: one output file per source
`task_type`, carrying `metadata.task`, with `media_root: null`.

Each sub-task becomes one item — `video_id`, `question`, `answer`, `reasoning`
when non-empty — plus an `item_index` of `<clip-stem>:<sub_task index>`, which
routes a training item back to the exact sub-task of the exact clip.
`video_url` carries through when present.

**Lossy by construction.** `tracking` is dropped, because
`tao-vl-reason-v1.0` has nowhere to put box geometry; the converter warns and
the geometry stays in the source batch. `<SKIP>`ped sub-tasks are dropped and
counted in `samples_skipped` — they were never human-corrected, so they are
not training data.

| Flag | Effect |
|------|--------|
| `--markers {keep,strip,drop}` | `<track>` handling in text. `keep` (default) leaves markers intact; `strip` unwraps to the bare `track_id`; `drop` removes marker and contents. |
| `--exclude-task-type [T ...]` | Leave task types out. Every type converts by default; `tracking_description` is annotation scaffolding rather than a QA task, so it is the common choice here. |
| `--description STR` | Suffixed with the task name into each output file's `metadata.description`. |

A non-uniform `metadata` block across the batch is an error: the per-task
output files have exactly one metadata block, so two licenses or two dates
cannot be represented honestly.

---

## `tao-vl-reason-v1.0` → `df-vlm-qa-v1.0`

Seeds a correction batch from an existing training set, so QA items that never
went through human review can. A **directory-level** operation: it reads every
annotation file under `--path` and emits one document per distinct `video_id`,
inverting the forward direction's per-`task_type` partitioning.

`item_index` is the join key — its `<clip-stem>:<sub_task index>` form recovers
both the grouping and the original `sub_tasks` ordering. When absent, items fall
back to file-then-array order and the converter warns. `metadata.task` becomes
`task_type`; a source file without one needs `--default-task-type`.

### What cannot be reconstructed

| Field | Result | Why |
|---|---|---|
| `tracking` | `[]` | No geometry survives the forward direction |
| `tracking_meta` | omitted | The grid is a property of a tracker run, not of QA text |
| `video_sha256` | recomputed from media when reachable, else omitted | Guessing it would be worse than omitting |
| `<SKIP>`ped sub-tasks | not restored | `item_index` reveals the gap but cannot fill it |
| `metadata` beyond `date` / `license` | dropped | The forward direction carries only those two |

| Flag | Effect |
|------|--------|
| `--default-task-type T` | `task_type` for items whose source file has no `metadata.task`. |
| `--geometry-from <batch>/` | Re-attaches `tracking`, `tracking_meta` and `video_sha256` by `video_id` from an existing batch, making the round trip lossless apart from the `metadata` extras above. |

### Seeded batches and `<track>` markers

A batch seeded without `--geometry-from` has empty `tracking`. If the source QA
text still carries `<track>` markers — which it does when the forward direction
ran with `--markers keep` — every marker is then an unresolvable reference, and
the validator reports each as an **error**. Three ways to get a batch that
validates: seed with `--geometry-from`, run the forward direction with
`--markers strip` or `drop`, or supply a tracker run before shipping.

---

## Architecture

`BaseConverter` ([`base.py`](base.py)) is an ABC with:

- class attributes `source_format`, `target_format` (both must be registered validator formats)
- class method `register_subparser(target_subparsers)` for CLI wiring
- instance method `run() -> int` invoked by the CLI

Subclasses populate `BaseConverter.converters` via `__init_subclass__`. The
CLI groups pairs by `source_format` to build nested subparsers and dispatches
`tao-daft convert <source> <target>` to the matching class.
`BaseConverter.validate_registry()` runs once at CLI startup to assert that
every pair's `source_format` / `target_format` is a known validator format.

---

## `ConversionResult`

Returned by every converter — `samples_written`, `samples_skipped`,
`warnings`, `errors`, plus `is_success()` and `summary()` helpers. Full
dataclass definition in [`base.py`](base.py).

---

## Adding a new converter pair

Use one of the existing pair modules (linked in the Registered pairs table
above) as a template. Required: subclass `BaseConverter`, set `source_format`
and `target_format` (both must be known validator formats), implement
`register_subparser` and `run`. Re-export from `converters/__init__.py` to
trigger auto-registration.
