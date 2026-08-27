# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Converter: ``tao-vl-reason-v1.0`` training format → ``df-vlm-qa-v1.0`` correction batch.

Seeds a correction batch from an existing training set, so QA items that never
went through human review can. This is a **directory-level** operation, not a
file-level one: the forward direction partitions by ``task_type``, so inverting
it means reading every annotation file under ``--path`` and regrouping the items
by clip.

The two directions are not inverses. Forward discards geometry; this direction
cannot invent it back, so ``tracking`` comes out empty and ``tracking_meta`` is
omitted unless ``--geometry-from`` points at a batch to recover them from.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from nvidia_tao_daft.converters.base import BaseConverter, ConversionResult
from nvidia_tao_daft.utils.df_vlm_qa_v1_0 import is_document
from nvidia_tao_daft.utils.df_vlm_qa_v1_0 import find_datasets as find_df_batches
from nvidia_tao_daft.utils.tao_vl_reason_v1_0 import FORMAT as SOURCE_FORMAT
from nvidia_tao_daft.utils.tao_vl_reason_v1_0 import find_datasets, resolve_media_path
from nvidia_tao_daft.utils.utils import FormatError, read_json_object

#: ``metadata`` keys carried from the source dataset onto every emitted document.
_CARRIED_METADATA = ("date", "license")

#: Fields recovered from a ``--geometry-from`` batch, keyed by ``video_id``.
_GEOMETRY_FIELDS = ("tracking_meta", "tracking", "video_sha256")


class _Item:
    """One source item plus the ordering key recovered from ``item_index``."""

    def __init__(self, item: dict, task_type: str, file_order: int, array_order: int) -> None:
        self.item = item
        self.task_type = task_type
        self.file_order = file_order
        self.array_order = array_order
        self.clip_stem: Optional[str] = None
        self.sub_task_index: Optional[int] = None
        raw = item.get("item_index")
        if isinstance(raw, str) and ":" in raw:
            stem, _, index = raw.rpartition(":")
            if stem and index.isdigit():
                self.clip_stem = stem
                self.sub_task_index = int(index)

    @property
    def sort_key(self) -> Tuple[int, int, int]:
        """Order by the recovered sub-task index, else by file-then-array order."""
        if self.sub_task_index is not None:
            return (0, self.sub_task_index, 0)
        return (1, self.file_order, self.array_order)


class TaoVlReasonV1_0ToDfVlmQaV1_0Converter(BaseConverter):
    """Converts a tao-vl-reason-v1.0 dataset into a df-vlm-qa-v1.0 correction batch.

    Output layout::

        {output}/
        ├── <clip-stem>.json      (one document per distinct video_id)
        └── <clip-stem>.json

    A seeded batch is schema-valid but not ready to ship: it carries no
    annotation grid, so it trips the validator's "``tracking_meta`` present when
    boxes were sought" warning until a tracker run supplies one.
    """

    source_format: ClassVar[str] = SOURCE_FORMAT
    target_format: ClassVar[str] = "df-vlm-qa-v1.0"

    # ------------------------------------------------------------------
    # CLI plumbing
    # ------------------------------------------------------------------
    @classmethod
    def register_subparser(cls, target_subparsers: "argparse._SubParsersAction") -> None:
        """Register this pair's argparse subparser under its target format."""
        parser = target_subparsers.add_parser(
            cls.target_format,
            help=f"Convert {cls.source_format} → {cls.target_format}",
        )
        cls._add_shared_arguments(parser)
        parser.add_argument(
            "--default-task-type",
            type=str,
            default=None,
            help="task_type for items whose source file has no metadata.task. "
            "Required only when some source file omits it.",
        )
        parser.add_argument(
            "--geometry-from",
            type=Path,
            default=None,
            help="Path to an existing df-vlm-qa-v1.0 batch. Re-attaches tracking, "
            "tracking_meta and video_sha256 by video_id, turning this direction "
            "into a true round-trip.",
        )

    # ------------------------------------------------------------------
    # CLI execution
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Convert the dataset at ``--path`` into a batch at ``--output``."""
        from nvidia_tao_daft import __version__

        assert self.args is not None, "run() requires args; pass them at construction"
        args = self.args

        print(f"🔄 NVIDIA TAO DAFT Converter v{__version__}")
        print(f"   {self.source_format} → {self.target_format}")
        print(f"   Source : {args.path}")
        print(f"   Output : {args.output}\n")

        result = self.convert_dataset(
            dataset_path=Path(args.path),
            output_path=Path(args.output),
            default_task_type=args.default_task_type,
            geometry_from=args.geometry_from,
        )

        print("=" * 60)
        print("CONVERSION RESULTS")
        print("=" * 60)
        print(result.summary())

        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")
        if result.errors:
            print(f"\n❌ Errors ({len(result.errors)}):")
            for e in result.errors:
                print(f"  - {e}")
        print("\n" + "=" * 60)

        if result.is_success():
            print("✅ CONVERSION COMPLETE")
            return 0
        if result.errors:
            print("❌ CONVERSION FAILED")
        else:
            print("⚠️  CONVERSION INCOMPLETE (some samples skipped — see warnings)")
        return 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert_dataset(
        self,
        dataset_path: Path,
        output_path: Path,
        *,
        default_task_type: Optional[str] = None,
        geometry_from: Optional[Path] = None,
    ) -> ConversionResult:
        """Regroup every tao-vl-reason-v1.0 item under *dataset_path* by clip.

        One document is emitted per distinct ``video_id``. Sub-task order is
        recovered from ``item_index`` where present; otherwise items fall back
        to file-then-array order and the loss of the original ordering is
        reported.
        """
        dataset_path = Path(dataset_path).resolve()
        output_path = Path(output_path)
        result = ConversionResult()

        by_clip, metadata = self._collect_items(
            dataset_path, default_task_type, result
        )
        if result.errors:
            return result
        if not by_clip:
            result.errors.append(
                f"No {self.source_format} items found under {dataset_path}"
            )
            return result

        geometry = self._load_geometry(geometry_from, result) if geometry_from else {}

        output_path.mkdir(parents=True, exist_ok=True)
        used_names: Dict[str, str] = {}
        for video_id, items in sorted(by_clip.items()):
            doc = self._build_document(video_id, items, metadata, geometry, result)
            name = self._document_name(video_id, items)
            if name in used_names and used_names[name] != video_id:
                result.errors.append(
                    f"two clips map to the same document name '{name}.json': "
                    f"{used_names[name]!r} and {video_id!r}"
                )
                continue
            used_names[name] = video_id
            with open(output_path / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
                f.write("\n")
            result.samples_written += len(doc["sub_tasks"])

        return result

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------
    def _collect_items(
        self,
        dataset_path: Path,
        default_task_type: Optional[str],
        result: ConversionResult,
    ) -> Tuple[Dict[str, List[_Item]], Dict[str, Any]]:
        """Read every annotation file under *dataset_path* and group items by clip.

        Also resolves each item's media path against its own file's
        ``media_root``, since two annotation files in one dataset may use
        different roots.
        """
        by_clip: Dict[str, List[_Item]] = defaultdict(list)
        seen_metadata: Dict[str, List[str]] = defaultdict(list)
        missing_index = 0
        file_order = 0

        for ds_root in find_datasets(dataset_path):
            for ann_path in sorted(ds_root.glob("*.json")):
                try:
                    data = read_json_object(ann_path)
                except FormatError:
                    continue
                if data.get("format") != SOURCE_FORMAT:
                    continue

                meta = data.get("metadata", {})
                task_type = meta.get("task") or default_task_type
                if not task_type:
                    result.errors.append(
                        f"{ann_path.name}: no metadata.task and no --default-task-type; "
                        "cannot decide task_type for its items"
                    )
                    continue

                key = json.dumps({k: meta.get(k) for k in _CARRIED_METADATA}, sort_keys=True)
                seen_metadata[key].append(ann_path.name)

                media_root = data.get("media_root")
                for array_order, item in enumerate(data.get("items", [])):
                    video_id = item.get("video_id")
                    if not video_id:
                        result.warnings.append(
                            f"{ann_path.name}: items[{array_order}] has no video_id "
                            "(image-only items have no place in df-vlm-qa-v1.0), skipping"
                        )
                        result.samples_skipped += 1
                        continue
                    wrapped = _Item(item, task_type, file_order, array_order)
                    wrapped.item = dict(item)
                    wrapped.item["_media"] = resolve_media_path(ds_root, media_root, video_id)
                    if wrapped.sub_task_index is None:
                        missing_index += 1
                    by_clip[video_id].append(wrapped)
                file_order += 1

        if len(seen_metadata) > 1:
            variants = "; ".join(
                f"{names[0]} has {key}" for key, names in sorted(seen_metadata.items())
            )
            result.errors.append(
                f"source metadata is not uniform across the dataset: {variants}"
            )
            return {}, {}
        if missing_index:
            result.warnings.append(
                f"{missing_index} item(s) have no item_index; their sub_task order falls "
                "back to file-then-array order, which may not match the original clip"
            )

        metadata: Dict[str, Any] = {}
        if seen_metadata:
            source = json.loads(next(iter(seen_metadata)))
            metadata = {k: v for k, v in source.items() if v is not None}
        return by_clip, metadata

    @staticmethod
    def _load_geometry(
        geometry_from: Path,
        result: ConversionResult,
    ) -> Dict[str, Dict[str, Any]]:
        """Index an existing df-vlm-qa-v1.0 batch by ``video_id``.

        Only the fields the forward direction destroys are recovered; the
        sub-tasks in the donor batch are ignored, since the whole point is to
        carry the *current* QA text forward onto the *original* geometry.
        """
        index: Dict[str, Dict[str, Any]] = {}
        for batch_root in find_df_batches(Path(geometry_from)):
            for doc_path in sorted(batch_root.glob("*.json")):
                try:
                    data = read_json_object(doc_path)
                except FormatError:
                    continue
                if not is_document(data):
                    continue
                index[data["video_id"]] = {
                    k: data[k] for k in _GEOMETRY_FIELDS if k in data
                }
        if not index:
            result.warnings.append(
                f"--geometry-from {geometry_from}: no {SOURCE_FORMAT} donor documents "
                "found, so no geometry was re-attached"
            )
        return index

    # ------------------------------------------------------------------
    # Document assembly
    # ------------------------------------------------------------------
    def _build_document(
        self,
        video_id: str,
        items: List[_Item],
        metadata: Dict[str, Any],
        geometry: Dict[str, Dict[str, Any]],
        result: ConversionResult,
    ) -> dict:
        """Assemble one df-vlm-qa-v1.0 document for a single clip.

        Field order follows the schema's own ordering so a diff against a
        donor batch stays readable.
        """
        ordered = sorted(items, key=lambda i: i.sort_key)

        doc: Dict[str, Any] = {
            "format": self.target_format,
            "metadata": dict({"type": "annotation"}, **metadata),
            "video_id": video_id,
        }

        video_url = next(
            (i.item["video_url"] for i in ordered if i.item.get("video_url")), None
        )
        if video_url:
            doc["video_url"] = video_url

        recovered = geometry.get(video_id)
        if recovered is None and geometry:
            result.warnings.append(
                f"{video_id}: no donor document in --geometry-from, so tracking is empty"
            )
        recovered = recovered or {}

        sha = recovered.get("video_sha256") or self._sha256(ordered[0].item.get("_media"))
        if sha:
            doc["video_sha256"] = sha

        if "tracking_meta" in recovered:
            doc["tracking_meta"] = recovered["tracking_meta"]
        doc["tracking"] = recovered.get("tracking", [])

        doc["sub_tasks"] = [self._build_sub_task(i) for i in ordered]
        return doc

    @staticmethod
    def _build_sub_task(wrapped: _Item) -> dict:
        """Map one tao-vl-reason item back into a ``sub_tasks`` entry."""
        item = wrapped.item
        sub_task = {
            "task_type": wrapped.task_type,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        }
        if item.get("reasoning"):
            sub_task["reasoning"] = item["reasoning"]
        return sub_task

    @staticmethod
    def _document_name(video_id: str, items: List[_Item]) -> str:
        """Return the output filename stem for a clip.

        Prefers the stem recorded in ``item_index``, which is what the forward
        direction wrote and therefore round-trips exactly. Falls back to the
        clip's own filename stem.
        """
        for wrapped in items:
            if wrapped.clip_stem:
                return wrapped.clip_stem
        return Path(video_id).stem

    @staticmethod
    def _sha256(media: Optional[Path]) -> Optional[str]:
        """Return the media's SHA-256, or None when it is not reachable.

        Omitting is deliberate: a hash that cannot be computed is better left
        out than guessed, because the field's whole purpose is to let both
        sides prove they annotated the same bytes.
        """
        if media is None or not Path(media).is_file():
            return None
        digest = hashlib.sha256()
        with open(media, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
