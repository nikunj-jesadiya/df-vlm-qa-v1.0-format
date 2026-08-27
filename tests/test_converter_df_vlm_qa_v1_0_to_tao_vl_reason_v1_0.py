# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the df-vlm-qa-v1.0 → tao-vl-reason-v1.0 converter."""

import argparse
import copy
import json
from pathlib import Path

import pytest

from nvidia_tao_daft.converters import BaseConverter
from nvidia_tao_daft.converters.pairs.df_vlm_qa_v1_0_to_tao_vl_reason_v1_0 import (
    DfVlmQaV1_0ToTaoVlReasonV1_0Converter,
)

_META = {"type": "annotation", "date": "2026-08-04", "license": "CC-BY-4.0"}

_DOC = {
    "format": "df-vlm-qa-v1.0",
    "metadata": _META,
    "video_id": "clips/a.mp4",
    "tracking_meta": {
        "time_unit": "frame_index",
        "box_order": "t_x1_y1_x2_y2",
        "coordinate_space": "pixel",
        "width": 100,
        "height": 100,
        "source_fps": 30.0,
        "frame_count": 300,
        "sample_fps": 7.5,
    },
    "tracking": [{"track_id": "person_T001", "track": [[0, 10, 10, 20, 20]]}],
    "sub_tasks": [
        {
            "task_type": "tracking_description",
            "question": "Describe the tracking <track>person_T001</track>.",
            "answer": "a person <track>person_T001</track>",
        },
        {
            "task_type": "open_qa",
            "question": "What happens?",
            "answer": "Nothing.",
            "reasoning": "Because nothing moves.",
        },
    ],
}


def _batch(tmp_path: Path, docs=None) -> Path:
    """Write a batch of documents, keyed by clip stem."""
    batch = tmp_path / "batch"
    batch.mkdir(exist_ok=True)
    for stem, doc in (docs or {"a": _DOC}).items():
        with open(batch / f"{stem}.json", "w") as f:
            json.dump(doc, f)
    return batch


def _convert(tmp_path: Path, docs=None, **kwargs):
    """Run the converter and return ``(result, output_path)``."""
    out = tmp_path / "out"
    result = DfVlmQaV1_0ToTaoVlReasonV1_0Converter().convert_dataset(
        _batch(tmp_path, docs), out, **kwargs
    )
    return result, out


def _load(out: Path, task_type: str) -> dict:
    with open(out / f"{task_type}.json") as f:
        return json.load(f)


class TestRegistration:
    """The pair is discoverable by the CLI."""

    def test_pair_registered(self):
        """Importing the package registers the pair on BaseConverter.converters."""
        pairs = {(c.source_format, c.target_format) for c in BaseConverter.converters}
        assert ("df-vlm-qa-v1.0", "tao-vl-reason-v1.0") in pairs

    def test_registry_endpoints_are_known_formats(self):
        """Both endpoints resolve to registered validator formats."""
        BaseConverter.validate_registry()

    def test_subparser_wired(self):
        """register_subparser exposes --path, --output, --markers."""
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="target")
        DfVlmQaV1_0ToTaoVlReasonV1_0Converter.register_subparser(subs)
        args = parser.parse_args(
            ["tao-vl-reason-v1.0", "--path", "a", "--output", "b", "--markers", "strip"]
        )
        assert args.markers == "strip"

    def test_markers_default_is_keep(self):
        """The issue's suggested default."""
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="target")
        DfVlmQaV1_0ToTaoVlReasonV1_0Converter.register_subparser(subs)
        args = parser.parse_args(["tao-vl-reason-v1.0", "--path", "a", "--output", "b"])
        assert args.markers == "keep"


class TestPartitioning:
    """One output file per task_type."""

    def test_one_file_per_task_type(self, tmp_path):
        """Each source task_type becomes its own annotation file."""
        result, out = _convert(tmp_path)
        assert result.samples_written == 2
        assert sorted(p.name for p in out.glob("*.json")) == [
            "open_qa.json",
            "tracking_description.json",
        ]

    def test_target_format_and_media_root(self, tmp_path):
        """Output declares the target format with a null media_root."""
        _, out = _convert(tmp_path)
        doc = _load(out, "open_qa")
        assert doc["format"] == "tao-vl-reason-v1.0"
        assert doc["media_root"] is None

    def test_metadata_task_and_carried_fields(self, tmp_path):
        """metadata.task names the source type; date and license carry through."""
        _, out = _convert(tmp_path)
        meta = _load(out, "open_qa")["metadata"]
        assert meta["type"] == "annotation"
        assert meta["task"] == "open_qa"
        assert meta["date"] == "2026-08-04"
        assert meta["license"] == "CC-BY-4.0"

    def test_aggregates_across_clips(self, tmp_path):
        """Sub-tasks from several clips land in the same per-type file."""
        b = copy.deepcopy(_DOC)
        b["video_id"] = "clips/b.mp4"
        _, out = _convert(tmp_path, {"a": _DOC, "b": b})
        assert len(_load(out, "open_qa")["items"]) == 2


class TestItemShape:
    """Each sub-task becomes one tao-vl-reason item."""

    def test_item_fields(self, tmp_path):
        """video_id, question, answer and item_index are always present."""
        _, out = _convert(tmp_path)
        item = _load(out, "open_qa")["items"][0]
        assert item["video_id"] == "clips/a.mp4"
        assert item["question"] == "What happens?"
        assert item["answer"] == "Nothing."
        assert item["item_index"] == "a:1"

    def test_item_index_is_clip_stem_and_source_index(self, tmp_path):
        """The index is the sub-task's position in the source, not the output."""
        _, out = _convert(tmp_path)
        assert _load(out, "tracking_description")["items"][0]["item_index"] == "a:0"
        assert _load(out, "open_qa")["items"][0]["item_index"] == "a:1"

    def test_reasoning_only_when_non_empty(self, tmp_path):
        """An empty reasoning field is omitted rather than written blank."""
        doc = copy.deepcopy(_DOC)
        doc["sub_tasks"][1]["reasoning"] = ""
        _, out = _convert(tmp_path, {"a": doc})
        assert "reasoning" not in _load(out, "open_qa")["items"][0]

    def test_reasoning_carried(self, tmp_path):
        """A non-empty reasoning field rides along."""
        _, out = _convert(tmp_path)
        assert _load(out, "open_qa")["items"][0]["reasoning"] == "Because nothing moves."

    def test_video_url_carried_when_present(self, tmp_path):
        """tao-vl-reason items are additionalProperties: true, so it fits."""
        doc = dict(copy.deepcopy(_DOC), video_url="s3://bucket/clips/a.mp4")
        _, out = _convert(tmp_path, {"a": doc})
        assert _load(out, "open_qa")["items"][0]["video_url"] == "s3://bucket/clips/a.mp4"

    def test_video_url_absent_when_source_has_none(self, tmp_path):
        """Nothing is invented when the source omits it."""
        _, out = _convert(tmp_path)
        assert "video_url" not in _load(out, "open_qa")["items"][0]


class TestMarkers:
    """--markers keep | strip | drop."""

    def test_keep(self, tmp_path):
        """Default leaves markers intact."""
        _, out = _convert(tmp_path, markers="keep")
        q = _load(out, "tracking_description")["items"][0]["question"]
        assert q == "Describe the tracking <track>person_T001</track>."

    def test_strip(self, tmp_path):
        """Unwraps to the bare track_id."""
        _, out = _convert(tmp_path, markers="strip")
        q = _load(out, "tracking_description")["items"][0]["question"]
        assert q == "Describe the tracking person_T001."

    def test_drop(self, tmp_path):
        """Removes marker and contents, tidying the leftover spacing."""
        _, out = _convert(tmp_path, markers="drop")
        q = _load(out, "tracking_description")["items"][0]["question"]
        assert q == "Describe the tracking."

    def test_markers_applied_to_all_text_fields(self, tmp_path):
        """question, answer and reasoning are treated alike."""
        doc = copy.deepcopy(_DOC)
        doc["sub_tasks"][1]["reasoning"] = "because <track>person_T001</track> moved"
        _, out = _convert(tmp_path, {"a": doc}, markers="strip")
        assert _load(out, "open_qa")["items"][0]["reasoning"] == "because person_T001 moved"


class TestSkipAndExclusion:
    """<SKIP>ped sub-tasks and --exclude-task-type."""

    def test_skipped_sub_tasks_dropped_and_counted(self, tmp_path):
        """They were never human-corrected, so they are not training data."""
        doc = copy.deepcopy(_DOC)
        doc["sub_tasks"][1]["answer"] = "<SKIP>not legible"
        result, out = _convert(tmp_path, {"a": doc})
        assert result.samples_skipped == 1
        assert result.samples_written == 1
        assert not (out / "open_qa.json").exists()

    def test_skip_makes_run_unsuccessful(self, tmp_path):
        """The output is smaller than the input, so is_success() is False."""
        doc = copy.deepcopy(_DOC)
        doc["sub_tasks"][1]["question"] = "<SKIP>q"
        result, _ = _convert(tmp_path, {"a": doc})
        assert not result.is_success()

    def test_exclude_task_type(self, tmp_path):
        """Excluded types are dropped silently — a filter, not a loss."""
        result, out = _convert(tmp_path, exclude_task_types=["tracking_description"])
        assert result.samples_written == 1
        assert result.samples_skipped == 0
        assert not (out / "tracking_description.json").exists()
        assert (out / "open_qa.json").exists()


class TestTrackingAndMetadata:
    """Geometry loss and batch consistency."""

    def test_tracking_dropped_with_warning(self, tmp_path):
        """Lossy by construction, and the converter says so."""
        result, _ = _convert(tmp_path)
        assert any("nowhere to put box geometry" in w for w in result.warnings)

    def test_no_tracking_no_warning(self, tmp_path):
        """A QA-only batch loses nothing, so nothing is reported."""
        doc = copy.deepcopy(_DOC)
        doc.pop("tracking")
        doc.pop("tracking_meta")
        doc["sub_tasks"] = [{"task_type": "open_qa", "question": "q?", "answer": "a."}]
        result, _ = _convert(tmp_path, {"a": doc})
        assert not any("box geometry" in w for w in result.warnings)

    def test_non_uniform_metadata_is_error(self, tmp_path):
        """A batch is one delivery; two licenses means two sources."""
        b = copy.deepcopy(_DOC)
        b["metadata"] = dict(_META, license="OTHER")
        result, _ = _convert(tmp_path, {"a": _DOC, "b": b})
        assert not result.is_success()
        assert any("not uniform" in e for e in result.errors)

    def test_missing_date_is_omitted_not_invented(self, tmp_path):
        """A real batch was found carrying license but no date."""
        doc = copy.deepcopy(_DOC)
        doc["metadata"] = {"type": "annotation", "license": ""}
        _, out = _convert(tmp_path, {"a": doc})
        meta = _load(out, "open_qa")["metadata"]
        assert "date" not in meta
        assert meta["license"] == ""

    def test_empty_batch_is_error(self, tmp_path):
        """Pointing at a directory with no documents fails clearly."""
        empty = tmp_path / "empty"
        empty.mkdir()
        result = DfVlmQaV1_0ToTaoVlReasonV1_0Converter().convert_dataset(
            empty, tmp_path / "out"
        )
        assert result.errors


class TestRun:
    """Exit codes from run()."""

    def test_run_success(self, tmp_path):
        """A clean batch converts and exits 0."""
        args = argparse.Namespace(
            path=_batch(tmp_path), output=tmp_path / "out", markers="keep",
            exclude_task_type=None, description=None,
        )
        assert DfVlmQaV1_0ToTaoVlReasonV1_0Converter(args).run() == 0

    def test_run_incomplete_on_skip(self, tmp_path):
        """A <SKIP>ped sub-task means the output is smaller than asked for."""
        doc = copy.deepcopy(_DOC)
        doc["sub_tasks"][1]["answer"] = "<SKIP>n/a"
        args = argparse.Namespace(
            path=_batch(tmp_path, {"a": doc}), output=tmp_path / "out", markers="keep",
            exclude_task_type=None, description=None,
        )
        assert DfVlmQaV1_0ToTaoVlReasonV1_0Converter(args).run() == 1

    def test_description_lands_in_metadata(self, tmp_path):
        """--description is suffixed with the task name, per the local script."""
        args = argparse.Namespace(
            path=_batch(tmp_path), output=tmp_path / "out", markers="keep",
            exclude_task_type=None, description="Batch 1",
        )
        DfVlmQaV1_0ToTaoVlReasonV1_0Converter(args).run()
        meta = _load(tmp_path / "out", "open_qa")["metadata"]
        assert meta["description"] == "Batch 1 — open_qa items."
