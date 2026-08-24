# df-vlm-qa-v1.0 — Directory Structure

## Overview

A `df-vlm-qa-v1.0` **batch** is a directory of per-clip JSON documents plus
the media those documents reference. One document describes exactly one video
clip.

```
{batch}/
├── <clip-stem>.json         # one document per clip
├── <clip-stem>.json
└── <clip-stem>.json
```

There is no fixed filename. The validator finds documents by reading every
`*.json` under the batch root and accepting any whose top-level `format`
field equals `"df-vlm-qa-v1.0"`. Anything else is skipped, so unrelated JSON
can sit alongside.

Every document has `metadata.type: "annotation"` (the schema discriminator).

## Filename convention

The conventional filename is the clip stem, matching the media it describes.

Keeping the two equal is what lets an individual sub-task be addressed as
`<clip-stem>:<sub_task index>` — a stable reference to the exact sub-task of
the exact clip, which any consumer regrouping or flattening a batch needs in
order to find its way back.

## One document per clip

Unlike `tao-vl-reason-v1.0`, where one file holds many items across many
clips, a `df-vlm-qa-v1.0` file is scoped to a single clip. That scoping is
load-bearing:

- `video_id` is a top-level field, not a per-item one;
- `track_id` is unique **within a clip**, so tracks cannot be pooled across
  files;
- the annotation grid in `tracking_meta` describes one clip's timeline;
- the correction UI opens one clip at a time, and a returned file is matched
  back to the sent one on `video_id`.

## Media root

`video_id` is a path relative to a **batch media root that the file itself
never names**.

This is the one structural difference from `tao-vl-reason-v1.0`, which
carries a `media_root` field per annotation file and supports three layouts
(self-contained, absolute, portable — see
[tao-vl-reason-v1.0 directory structure](../../tao-vl-reason-v1.0/specs/directory-structure.md#media-layouts)).
`df-vlm-qa-v1.0` has no such field: the media root is always supplied out of
band by the consumer, which is `tao-vl-reason-v1.0`'s "portable" shape and
only that shape.

| Consumer | How it resolves `video_id` |
|---|---|
| Validator | Against the batch root — `--path <batch>/` joined to `video_id`. |
| Annotation UI | Not at all; it uses `video_url`, presigned client-side. |
| Training pipeline | Against an out-of-band media root of its own choosing. |

The validator reports a `video_id` that does not resolve under the media root
as a **warning in permissive mode and an error in strict mode**, matching the
existing media check on the other formats.

## `video_id` vs `video_url`

The two fields answer different questions, and conflating them breaks the
hand-back match.

| Field | Question | Rewritable |
|---|---|---|
| `video_id` | *Which clip is this?* | **Never.** A returned file is matched back on it. |
| `video_url` | *Where do I fetch it?* | Freely — a bundle can be re-hosted without changing any clip's identity. |

Because `video_id` is never rewritten, the same clip keeps the same identity
whether it is read from a local checkout, a shared store, or S3.

## `video_url` is optional

The annotation UI is a browser component that fetches media client-side, so
it needs a per-item URI it can presign. But the destination is not knowable
at inference time — it is stamped at the upload step.

- Producers that know the destination at upload time stamp it in.
- Producers that do not omit it, and let the consumer join `video_id` to an
  out-of-band batch root.

When present it must be an `s3://` URI. A malformed one fails at presign
time, deep in the UI, so the validator treats it as an error rather than a
warning.

## `video_sha256`

Pins the exact bytes both sides annotated. A re-encode or re-cut of the
source invalidates the file loudly instead of silently shifting every box.

It is omitted when the media was not reachable at conversion time — guessing
would be worse than omitting. The consuming browser cannot hash the file, so
DAFT is the ingest point that can, and the check runs at validation.

Checks that need the media file — `video_sha256`, and media duration against
`frame_count`/`source_fps` or `duration_s` — are skipped with a warning when
the media is not reachable, consistent with the other validators.

## Linking model

```
{batch}/<clip-stem>.json
  ├── video_id       →  clip identity; resolved against the batch media root
  ├── video_url      →  where to fetch that same clip (optional, s3://)
  ├── tracking_meta  →  declares the annotation grid for this clip
  ├── tracking[]
  │     └── track_id  ←──────┐  clip-unique
  └── sub_tasks[]            │
        ├── question ────────┤  <track>{track_id}</track>
        ├── answer ──────────┤
        └── reasoning ───────┘
```

Tracks are referenced from QA text **only** through `<track>` markers. There
is no separate id list, so there is nothing to keep in sync — and an
unresolvable marker is a validator error.
