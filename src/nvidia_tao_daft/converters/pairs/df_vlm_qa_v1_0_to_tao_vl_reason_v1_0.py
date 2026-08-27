# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Converter: ``df-vlm-qa-v1.0`` correction batch → ``tao-vl-reason-v1.0`` training format.

Turns a corrected batch into training files. Sub-tasks are aggregated by
``task_type``, one output file per type, matching how ``tao-vl-reason-v1.0``
datasets are already partitioned.

The conversion is lossy by construction: ``tracking`` has nowhere to go in
``tao-vl-reason-v1.0``, so the geometry stays behind in the source batch. Every
emitted item carries an ``item_index`` of ``<clip-stem>:<sub_task index>``, which
is the route back to the exact sub-task of the exact clip — and the join key the
reverse converter uses.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from nvidia_tao_daft.converters.base import BaseConverter, ConversionResult
from nvidia_tao_daft.utils.df_vlm_qa_v1_0 import TRACK_REF, find_datasets, is_document, is_skipped
from nvidia_tao_daft.utils.utils import FormatError, read_json_object

#: ``metadata`` keys carried from the source batch onto every output file.
_CARRIED_METADATA = ("date", "license")

#: Collapses the whitespace a dropped marker leaves behind.
_MULTISPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def _describe_variant(metadata_json: str, names: List[str]) -> str:
    """Render one distinct metadata block and the files carrying it."""
    more = f" (+{len(names) - 1} more)" if len(names) > 1 else ""
    return f"{names[0]}{more} has {metadata_json}"


class DfVlmQaV1_0ToTaoVlReasonV1_0Converter(BaseConverter):
    """Converts a df-vlm-qa-v1.0 batch to a tao-vl-reason-v1.0 training dataset.

    Output layout::

        {output}/
        ├── open_qa.json              (one annotation file per source task_type)
        ├── scene_description.json
        └── ...

    Every file uses ``media_root: null``; items keep the source ``video_id``
    verbatim, so the consumer points ``media_root`` at whichever store holds
    the clips.
    """

    source_format: ClassVar[str] = "df-vlm-qa-v1.0"
    target_format: ClassVar[str] = "tao-vl-reason-v1.0"

    #: ``<track>`` handling modes for ``--markers``.
    MARKER_MODES = ("keep", "strip", "drop")

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
            "--markers",
            choices=cls.MARKER_MODES,
            default="keep",
            help="How to handle <track> markers in question / answer / reasoning: "
            "'keep' leaves them intact (default), 'strip' unwraps them to the bare "
            "track_id, 'drop' removes them and their contents entirely.",
        )
        parser.add_argument(
            "--exclude-task-type",
            type=str,
            nargs="*",
            default=None,
            help="Task types to leave out of the output. Every type converts by "
            "default; 'tracking_description' is annotation scaffolding rather than "
            "a QA task, so --exclude-task-type tracking_description is a common "
            "choice.",
        )
        parser.add_argument(
            "--description",
            type=str,
            help="Human-readable description added to each annotation file's metadata block",
        )

    # ------------------------------------------------------------------
    # CLI execution
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Convert the batch at ``--path`` into ``--output``.

        Returns 0 when every sub-task was written, 1 when anything errored or
        was skipped — including ``<SKIP>``ped sub-tasks, which are a deliberate
        omission but still mean the output is smaller than the input.
        """
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
            markers=args.markers,
            exclude_task_types=args.exclude_task_type or None,
            description=args.description,
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
        markers: str = "keep",
        exclude_task_types: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> ConversionResult:
        """Convert every df-vlm-qa-v1.0 document under *dataset_path*.

        Sub-tasks from all clips are aggregated by ``task_type`` into one
        annotation file per type at the output root. A non-uniform ``metadata``
        block across the batch is an error: the output files carry a single
        merged metadata block, so there is no honest way to represent two
        different licenses or dates in one file.
        """
        dataset_path = Path(dataset_path).resolve()
        output_path = Path(output_path)
        result = ConversionResult()

        documents = self._load_documents(dataset_path, result)
        if result.errors:
            return result
        if not documents:
            result.errors.append(
                f"No {self.source_format} documents found under {dataset_path}"
            )
            return result

        source_metadata = self._uniform_metadata(documents, result)
        if result.errors:
            return result

        excluded = set(exclude_task_types or ())
        items_by_type: Dict[str, List[dict]] = defaultdict(list)
        dropped_tracks = 0

        for doc_path, doc in documents:
            dropped_tracks += len(doc.get("tracking", []))
            self._collect_items(doc_path, doc, markers, excluded, items_by_type, result)

        if dropped_tracks:
            result.warnings.append(
                f"{dropped_tracks} track(s) across {len(documents)} clip(s) were dropped — "
                f"{self.target_format} has nowhere to put box geometry. The geometry stays "
                f"in the {self.source_format} batch."
            )

        output_path.mkdir(parents=True, exist_ok=True)
        for task_type, items in sorted(items_by_type.items()):
            self._write_annotation_file(
                output_path, task_type, items, source_metadata, description
            )
            result.samples_written += len(items)

        return result

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------
    def _load_documents(
        self,
        dataset_path: Path,
        result: ConversionResult,
    ) -> List[Tuple[Path, dict]]:
        """Return ``[(path, doc), ...]`` for every document in the batch.

        Discovery reuses the validator's rule — both the top-level ``format``
        and the ``metadata.type`` discriminator must match — so a batch's build
        artefacts are not mistaken for documents.
        """
        documents: List[Tuple[Path, dict]] = []
        for batch_root in find_datasets(dataset_path):
            for doc_path in sorted(batch_root.glob("*.json")):
                try:
                    data = read_json_object(doc_path)
                except FormatError as e:
                    result.errors.append(str(e))
                    continue
                if is_document(data):
                    documents.append((doc_path, data))
        return documents

    @staticmethod
    def _uniform_metadata(
        documents: List[Tuple[Path, dict]],
        result: ConversionResult,
    ) -> Dict[str, Any]:
        """Return the batch's single ``metadata`` block, or record an error.

        The whole block is compared, not just the keys carried onto the output.
        A batch is one delivery; two clips disagreeing on license, date, or
        provenance means it was assembled from more than one source, and the
        per-task output files have exactly one metadata block to put it in.
        """
        seen: Dict[str, List[str]] = defaultdict(list)
        for doc_path, doc in documents:
            key = json.dumps(doc.get("metadata", {}), sort_keys=True)
            seen[key].append(doc_path.name)
        if len(seen) > 1:
            variants = "; ".join(
                _describe_variant(key, names) for key, names in sorted(seen.items())
            )
            result.errors.append(
                f"source metadata is not uniform across the batch: {variants}"
            )
            return {}
        meta = documents[0][1].get("metadata", {})
        # Only keys that are present: a real batch was found carrying `license`
        # but no `date`, and inventing one would be worse than omitting it.
        return {k: meta[k] for k in _CARRIED_METADATA if k in meta}

    # ------------------------------------------------------------------
    # Core conversion logic
    # ------------------------------------------------------------------
    def _collect_items(
        self,
        doc_path: Path,
        doc: dict,
        markers: str,
        excluded: set,
        items_by_type: Dict[str, List[dict]],
        result: ConversionResult,
    ) -> None:
        """Turn one clip's sub-tasks into tao-vl-reason items.

        ``<SKIP>``ped sub-tasks are dropped and counted as skipped — they were
        never human-corrected, so they are not training data. Excluded task
        types are dropped silently; that is a deliberate filter, not a loss.
        """
        stem = doc_path.stem
        video_id = doc["video_id"]
        video_url = doc.get("video_url")

        for index, sub_task in enumerate(doc["sub_tasks"]):
            task_type = sub_task["task_type"]
            if task_type in excluded:
                continue
            if is_skipped(sub_task):
                result.samples_skipped += 1
                result.warnings.append(
                    f"{doc_path.name}: sub_tasks[{index}] is <SKIP>ped and was dropped — "
                    "it was never human-corrected"
                )
                continue

            item: Dict[str, Any] = {
                "video_id": video_id,
                "question": self._apply_markers(sub_task["question"], markers),
                "answer": self._apply_markers(sub_task["answer"], markers),
                "item_index": f"{stem}:{index}",
            }
            reasoning = sub_task.get("reasoning")
            if reasoning:
                item["reasoning"] = self._apply_markers(reasoning, markers)
            if video_url:
                item["video_url"] = video_url

            items_by_type[task_type].append(item)

    @staticmethod
    def _apply_markers(text: str, mode: str) -> str:
        """Rewrite ``<track>`` markers in *text* according to *mode*.

        ``keep`` returns the text unchanged. ``strip`` unwraps each marker to
        its bare ``track_id``, keeping the reference readable while removing the
        tags. ``drop`` removes marker and contents, then tidies the whitespace
        and punctuation spacing the removal leaves behind.
        """
        if mode == "keep":
            return text
        if mode == "strip":
            return TRACK_REF.sub(lambda m: m.group(1), text)
        without = TRACK_REF.sub("", text)
        without = _MULTISPACE.sub(" ", without)
        without = _SPACE_BEFORE_PUNCT.sub(r"\1", without)
        return without.strip()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _write_annotation_file(
        self,
        output_path: Path,
        task_type: str,
        items: List[dict],
        source_metadata: Dict[str, Any],
        description: Optional[str],
    ) -> None:
        """Emit ``output_path/<task_type>.json`` with the aggregated items.

        ``metadata.type`` is fixed to ``"annotation"`` (the schema
        discriminator) and the source task type is recorded in ``metadata.task``,
        so a multi-file dataset stays partitionable without reading items.
        """
        meta_block: Dict[str, Any] = {"type": "annotation", "task": task_type}
        meta_block.update(source_metadata)
        if description:
            meta_block["description"] = f"{description} — {task_type} items."
        annotation = {
            "format": self.target_format,
            "metadata": meta_block,
            "media_root": None,
            "items": items,
        }
        with open(output_path / f"{task_type}.json", "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
            f.write("\n")
