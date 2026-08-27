# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tao-vl-reason-v1.0 → df-vlm-qa-v1.0 converter, plus the round trip."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest

from nvidia_tao_daft.converters import BaseConverter
from nvidia_tao_daft.converters.pairs.df_vlm_qa_v1_0_to_tao_vl_reason_v1_0 import (
    DfVlmQaV1_0ToTaoVlReasonV1_0Converter,
)
from nvidia_tao_daft.converters.pairs.tao_vl_reason_v1_0_to_df_vlm_qa_v1_0 import (
    TaoVlReasonV1_0ToDfVlmQaV1_0Converter,
)

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _annotation(task, items, media_root=None, **meta):
    """Build one tao-vl-reason-v1.0 annotation file body."""
    metadata = {"type": "annotation", "license": "CC-BY-4.0"}
    if task is not None:
        metadata["task"] = task
    metadata.update(meta)
    return {
        "format": "tao-vl-reason-v1.0",
        "metadata": metadata,
        "media_root": media_root,
        "items": items,
    }


def _item(video_id="clips/a.mp4", q="q?", a="a.", index=None, **extra):
    item = {"video_id": video_id, "question": q, "answer": a}
    if index is not None:
        item["item_index"] = index
    item.update(extra)
    return item


def _dataset(tmp_path: Path, files: dict, *, media=()) -> Path:
    """Write annotation files into a dataset root, optionally with media."""
    root = tmp_path / "ds"
    root.mkdir(exist_ok=True)
    for rel in media:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"")
    for name, body in files.items():
        with open(root / f"{name}.json", "w") as f:
            json.dump(body, f)
    return root


def _convert(tmp_path: Path, files: dict, *, media=(), **kwargs):
    out = tmp_path / "batch"
    result = TaoVlReasonV1_0ToDfVlmQaV1_0Converter().convert_dataset(
        _dataset(tmp_path, files, media=media), out, **kwargs
    )
    return result, out


def _doc(out: Path, stem: str) -> dict:
    with open(out / f"{stem}.json") as f:
        return json.load(f)


class TestRegistration:
    """The reverse pair is discoverable and its endpoints are known formats."""

    def test_pair_registered(self):
        """Both directions of the pair are in the registry."""
        pairs = {(c.source_format, c.target_format) for c in BaseConverter.converters}
        assert ("tao-vl-reason-v1.0", "df-vlm-qa-v1.0") in pairs
        assert ("df-vlm-qa-v1.0", "tao-vl-reason-v1.0") in pairs

    def test_registry_valid(self):
        """validate_registry accepts the reverse pair."""
        BaseConverter.validate_registry()

    def test_subparser_wired(self):
        """--default-task-type and --geometry-from are exposed."""
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="target")
        TaoVlReasonV1_0ToDfVlmQaV1_0Converter.register_subparser(subs)
        args = parser.parse_args(
            ["df-vlm-qa-v1.0", "--path", "a", "--output", "b",
             "--default-task-type", "open_qa", "--geometry-from", "g"]
        )
        assert args.default_task_type == "open_qa"
        assert args.geometry_from == Path("g")


class TestRegrouping:
    """One document per distinct video_id, inverting the per-task partitioning."""

    def test_one_document_per_clip(self, tmp_path):
        """Items from many task files regroup by clip."""
        files = {
            "open_qa": _annotation("open_qa", [
                _item("clips/a.mp4", index="a:0"), _item("clips/b.mp4", index="b:0")]),
            "bcq": _annotation("bcq", [_item("clips/a.mp4", index="a:1")]),
        }
        result, out = _convert(tmp_path, files)
        assert sorted(p.stem for p in out.glob("*.json")) == ["a", "b"]
        assert result.samples_written == 3

    def test_sub_task_order_from_item_index(self, tmp_path):
        """The index restores the original ordering across files."""
        files = {
            "z_last": _annotation("open_qa", [_item(q="second", index="a:1")]),
            "a_first": _annotation("bcq", [_item(q="first", index="a:0")]),
        }
        _, out = _convert(tmp_path, files)
        assert [s["question"] for s in _doc(out, "a")["sub_tasks"]] == ["first", "second"]

    def test_task_type_from_metadata_task(self, tmp_path):
        """metadata.task becomes the sub-task's task_type."""
        _, out = _convert(tmp_path, {"f": _annotation("scene_description", [_item(index="a:0")])})
        assert _doc(out, "a")["sub_tasks"][0]["task_type"] == "scene_description"

    def test_missing_item_index_warns_and_falls_back(self, tmp_path):
        """File-then-array order is a fallback, and the loss is reported."""
        files = {"f": _annotation("open_qa", [_item(q="one"), _item(q="two")])}
        result, out = _convert(tmp_path, files)
        assert any("no item_index" in w for w in result.warnings)
        assert [s["question"] for s in _doc(out, "a")["sub_tasks"]] == ["one", "two"]

    def test_document_named_from_item_index_stem(self, tmp_path):
        """The stem the forward direction wrote round-trips exactly."""
        files = {"f": _annotation("open_qa", [_item("x/y/clip.mp4", index="original_stem:0")])}
        _, out = _convert(tmp_path, files)
        assert (out / "original_stem.json").exists()


class TestTaskTypeResolution:
    """metadata.task, else --default-task-type, else an error."""

    def test_default_task_type_used(self, tmp_path):
        """A file with no metadata.task falls back to the flag."""
        files = {"f": _annotation(None, [_item(index="a:0")])}
        _, out = _convert(tmp_path, files, default_task_type="open_qa")
        assert _doc(out, "a")["sub_tasks"][0]["task_type"] == "open_qa"

    def test_missing_task_type_is_error(self, tmp_path):
        """Neither source nor flag means there is nothing to write."""
        result, _ = _convert(tmp_path, {"f": _annotation(None, [_item(index="a:0")])})
        assert any("no metadata.task" in e for e in result.errors)

    def test_metadata_task_wins_over_default(self, tmp_path):
        """The flag is a fallback, not an override."""
        files = {"f": _annotation("bcq", [_item(index="a:0")])}
        _, out = _convert(tmp_path, files, default_task_type="open_qa")
        assert _doc(out, "a")["sub_tasks"][0]["task_type"] == "bcq"


class TestCannotReconstruct:
    """The documented losses, and nothing silently beyond them."""

    def test_tracking_and_meta_both_omitted(self, tmp_path):
        """No geometry survives the forward direction.

        Both keys are left off entirely, not written empty/blank — tao-vl-
        reason-v1.0 has nowhere to hold geometry, so there is nothing to
        report finding-and-losing; an empty tracking: [] would misrepresent
        "boxes were sought and none found" instead of "never sought."
        """
        _, out = _convert(tmp_path, {"f": _annotation("open_qa", [_item(index="a:0")])})
        doc = _doc(out, "a")
        assert "tracking" not in doc
        assert "tracking_meta" not in doc

    def test_sha256_omitted_when_media_unreachable(self, tmp_path):
        """Guessing it would be worse than omitting it."""
        _, out = _convert(tmp_path, {"f": _annotation("open_qa", [_item(index="a:0")])})
        assert "video_sha256" not in _doc(out, "a")

    def test_sha256_recomputed_when_media_reachable(self, tmp_path):
        """DAFT is the ingest point that can hash the file."""
        files = {"f": _annotation("open_qa", [_item(index="a:0")])}
        _, out = _convert(tmp_path, files, media=["clips/a.mp4"])
        assert _doc(out, "a")["video_sha256"] == _EMPTY_SHA256

    def test_video_url_picked_back_up(self, tmp_path):
        """It rode along on the item, so it comes back."""
        files = {"f": _annotation("open_qa", [_item(index="a:0", video_url="s3://b/k.mp4")])}
        _, out = _convert(tmp_path, files)
        assert _doc(out, "a")["video_url"] == "s3://b/k.mp4"

    def test_metadata_carried(self, tmp_path):
        """date and license carry through onto every document."""
        files = {"f": _annotation("open_qa", [_item(index="a:0")], date="2026-08-04")}
        _, out = _convert(tmp_path, files)
        meta = _doc(out, "a")["metadata"]
        assert meta == {"type": "annotation", "date": "2026-08-04", "license": "CC-BY-4.0"}


class TestGeometryFrom:
    """--geometry-from turns the reverse direction into a true round trip."""

    def _donor(self, tmp_path: Path) -> Path:
        donor = tmp_path / "donor"
        donor.mkdir()
        doc = {
            "format": "df-vlm-qa-v1.0",
            "metadata": {"type": "annotation", "license": "CC-BY-4.0"},
            "video_id": "clips/a.mp4",
            "video_sha256": "b" * 64,
            "tracking_meta": {
                "time_unit": "frame_index", "box_order": "t_x1_y1_x2_y2",
                "coordinate_space": "pixel", "width": 100, "height": 100,
                "source_fps": 30.0, "frame_count": 300, "sample_fps": 7.5,
            },
            "tracking": [{"track_id": "person_T001", "track": [[0, 1, 1, 2, 2]]}],
            "sub_tasks": [{"task_type": "open_qa", "question": "ignored", "answer": "ignored"}],
        }
        with open(donor / "a.json", "w") as f:
            json.dump(doc, f)
        return donor

    def test_reattaches_geometry(self, tmp_path):
        """tracking, tracking_meta and video_sha256 come back by video_id."""
        files = {"f": _annotation("open_qa", [_item(index="a:0")])}
        _, out = _convert(tmp_path, files, geometry_from=self._donor(tmp_path))
        doc = _doc(out, "a")
        assert len(doc["tracking"]) == 1
        assert doc["tracking_meta"]["sample_fps"] == 7.5
        assert doc["video_sha256"] == "b" * 64

    def test_donor_sub_tasks_ignored(self, tmp_path):
        """The point is current QA text on original geometry."""
        files = {"f": _annotation("open_qa", [_item(q="current", index="a:0")])}
        _, out = _convert(tmp_path, files, geometry_from=self._donor(tmp_path))
        assert _doc(out, "a")["sub_tasks"][0]["question"] == "current"

    def test_missing_donor_clip_warns(self, tmp_path):
        """A clip with no donor gets no tracking key at all, and says so."""
        files = {"f": _annotation("open_qa", [_item("clips/zz.mp4", index="zz:0")])}
        result, out = _convert(tmp_path, files, geometry_from=self._donor(tmp_path))
        assert "tracking" not in _doc(out, "zz")
        assert any("no donor document" in w for w in result.warnings)

    def test_empty_donor_batch_warns(self, tmp_path):
        """Pointing at a directory with no documents is reported, not silent."""
        empty = tmp_path / "nothing"
        empty.mkdir()
        files = {"f": _annotation("open_qa", [_item(index="a:0")])}
        result, _ = _convert(tmp_path, files, geometry_from=empty)
        assert any("no" in w and "donor" in w for w in result.warnings)


class TestPlaceVideos:
    """--place-videos switches to the jsons/+videos/ bundle layout."""

    def test_default_stays_flat_and_untouched(self, tmp_path):
        """Without the flag, documents stay flat and no videos/ appears."""
        files = {"f": _annotation("open_qa", [_item(index="a:0")])}
        _, out = _convert(tmp_path, files, media=("clips/a.mp4",))
        assert (out / "a.json").is_file()
        assert not (out / "jsons").exists()
        assert not (out / "videos").exists()

    def test_places_media_and_nests_documents(self, tmp_path):
        """The document moves under jsons/ and the media lands under videos/."""
        files = {"f": _annotation("open_qa", [_item(index="a:0")])}
        result, out = _convert(
            tmp_path, files, media=("clips/a.mp4",), place_videos="copy"
        )
        assert (out / "jsons" / "a.json").is_file()
        assert (out / "videos" / "clips" / "a.mp4").is_file()
        assert result.is_success()

    def test_placed_media_matches_source_bytes(self, tmp_path):
        """copy mode places the real bytes, not a stub."""
        root = tmp_path / "ds"
        root.mkdir()
        (root / "clips").mkdir()
        (root / "clips" / "a.mp4").write_bytes(b"real video bytes")
        with open(root / "f.json", "w") as f:
            json.dump(_annotation("open_qa", [_item(index="a:0")]), f)
        out = tmp_path / "batch"
        TaoVlReasonV1_0ToDfVlmQaV1_0Converter().convert_dataset(
            root, out, place_videos="copy"
        )
        assert (out / "videos" / "clips" / "a.mp4").read_bytes() == b"real video bytes"

    def test_video_id_needs_no_rewrite(self, tmp_path):
        """video_id already names the path under videos/, unchanged."""
        files = {"f": _annotation("open_qa", [_item(index="a:0")])}
        _, out = _convert(
            tmp_path, files, media=("clips/a.mp4",), place_videos="copy"
        )
        doc = json.load(open(out / "jsons" / "a.json"))
        assert doc["video_id"] == "clips/a.mp4"
        assert (out / "videos" / doc["video_id"]).is_file()

    def test_unreachable_media_warns_but_still_writes_document(self, tmp_path):
        """A missing source file doesn't stop the batch from being written."""
        files = {"f": _annotation("open_qa", [_item(index="a:0")])}
        result, out = _convert(tmp_path, files, place_videos="copy")
        assert (out / "jsons" / "a.json").is_file()
        assert not (out / "videos" / "clips" / "a.mp4").exists()
        assert any("not reachable" in w for w in result.warnings)

    def test_symlink_mode(self, tmp_path):
        """symlink mode links rather than copies."""
        files = {"f": _annotation("open_qa", [_item(index="a:0")])}
        _, out = _convert(
            tmp_path, files, media=("clips/a.mp4",), place_videos="symlink"
        )
        placed = out / "videos" / "clips" / "a.mp4"
        assert placed.is_symlink()

    def test_hardlink_mode(self, tmp_path):
        """hardlink mode links rather than copies."""
        files = {"f": _annotation("open_qa", [_item(index="a:0")])}
        _, out = _convert(
            tmp_path, files, media=("clips/a.mp4",), place_videos="hardlink"
        )
        src = tmp_path / "ds" / "clips" / "a.mp4"
        placed = out / "videos" / "clips" / "a.mp4"
        assert placed.stat().st_ino == src.stat().st_ino


class TestEdgeCases:
    """Inputs that have no honest representation in the target."""

    def test_image_only_item_skipped(self, tmp_path):
        """df-vlm-qa-v1.0 is video-only; an image item has nowhere to go."""
        files = {"f": _annotation("open_qa", [{"image_id": "i.jpg", "question": "q", "answer": "a"}])}
        result, _ = _convert(tmp_path, files)
        assert result.samples_skipped == 1
        assert any("no video_id" in w for w in result.warnings)

    def test_non_uniform_metadata_is_error(self, tmp_path):
        """Two licenses in one dataset cannot both land on a document."""
        files = {
            "a": _annotation("open_qa", [_item(index="a:0")]),
            "b": _annotation("bcq", [_item(index="a:1")], license="OTHER"),
        }
        result, _ = _convert(tmp_path, files)
        assert any("not uniform" in e for e in result.errors)

    def test_empty_dataset_is_error(self, tmp_path):
        """Nothing to convert fails clearly."""
        empty = tmp_path / "empty"
        empty.mkdir()
        result = TaoVlReasonV1_0ToDfVlmQaV1_0Converter().convert_dataset(empty, tmp_path / "o")
        assert result.errors

    def test_run_exit_code(self, tmp_path):
        """A clean dataset converts and exits 0."""
        root = _dataset(tmp_path, {"f": _annotation("open_qa", [_item(index="a:0")])})
        args = argparse.Namespace(
            path=root,
            output=tmp_path / "batch",
            default_task_type=None,
            geometry_from=None,
            place_videos=None,
        )
        assert TaoVlReasonV1_0ToDfVlmQaV1_0Converter(args).run() == 0


class TestRoundTrip:
    """forward → reverse loses exactly what the format documents, and no more."""

    SOURCE = {
        "format": "df-vlm-qa-v1.0",
        "metadata": {"type": "annotation", "date": "2026-08-04", "license": "CC-BY-4.0"},
        "video_id": "clips/a.mp4",
        "video_url": "s3://bucket/clips/a.mp4",
        "video_sha256": "c" * 64,
        "tracking_meta": {
            "time_unit": "frame_index", "box_order": "t_x1_y1_x2_y2",
            "coordinate_space": "pixel", "width": 100, "height": 100,
            "source_fps": 30.0, "frame_count": 300, "sample_fps": 7.5,
        },
        "tracking": [{"track_id": "person_T001", "track": [[0, 1, 1, 2, 2]]}],
        "sub_tasks": [
            {"task_type": "open_qa", "question": "first?", "answer": "1."},
            {"task_type": "bcq", "question": "second?", "answer": "Yes", "reasoning": "because"},
            {"task_type": "open_qa", "question": "third?", "answer": "3."},
        ],
    }

    def _round_trip(self, tmp_path, source=None, **reverse_kwargs):
        """Forward then reverse *source* (default: SOURCE); return the result."""
        batch = tmp_path / "src"
        batch.mkdir(exist_ok=True)
        with open(batch / "a.json", "w") as f:
            json.dump(self.SOURCE if source is None else source, f)
        mid = tmp_path / "mid"
        DfVlmQaV1_0ToTaoVlReasonV1_0Converter().convert_dataset(batch, mid)
        out = tmp_path / "back"
        TaoVlReasonV1_0ToDfVlmQaV1_0Converter().convert_dataset(mid, out, **reverse_kwargs)
        with open(out / "a.json") as f:
            return json.load(f)

    def test_sub_tasks_survive_exactly(self, tmp_path):
        """Content and ordering both come back intact."""
        back = self._round_trip(tmp_path)
        assert back["sub_tasks"] == self.SOURCE["sub_tasks"]

    def test_identity_fields_survive(self, tmp_path):
        """The clip's identity and location are preserved."""
        back = self._round_trip(tmp_path)
        assert back["video_id"] == self.SOURCE["video_id"]
        assert back["video_url"] == self.SOURCE["video_url"]
        assert back["format"] == "df-vlm-qa-v1.0"

    def test_documented_losses_and_no_others(self, tmp_path):
        """Exactly tracking, tracking_meta, video_sha256 and metadata extras."""
        back = self._round_trip(tmp_path)
        lost = {k for k in self.SOURCE if self.SOURCE.get(k) != back.get(k)}
        assert lost == {"tracking", "tracking_meta", "video_sha256"}
        assert "tracking" not in back
        assert "tracking_meta" not in back
        assert "video_sha256" not in back

    def test_geometry_from_restores_the_losses(self, tmp_path):
        """With the donor batch, only metadata extras differ."""
        batch = tmp_path / "donor"
        batch.mkdir()
        with open(batch / "a.json", "w") as f:
            json.dump(self.SOURCE, f)
        back = self._round_trip(tmp_path, geometry_from=batch)
        assert back == self.SOURCE

    def test_metadata_extras_are_lost_undocumented(self, tmp_path):
        """description and tags do not survive, and the format does not say so.

        The forward direction carries only ``date`` and ``license``, so any
        other ``metadata`` key is dropped. That loss is real — it shows on every
        clip of the production batches — but it is absent from the documented
        list of what cannot be reconstructed. Pinned here so the gap is visible
        rather than surprising.
        """
        source = copy.deepcopy(self.SOURCE)
        source["metadata"]["description"] = "batch notes"
        source["metadata"]["tags"] = ["a", "b"]
        back = self._round_trip(tmp_path, source=source)
        assert "description" not in back["metadata"]
        assert "tags" not in back["metadata"]
        assert back["metadata"] == {
            "type": "annotation", "date": "2026-08-04", "license": "CC-BY-4.0"
        }

    def test_empty_reasoning_survives_the_round_trip(self, tmp_path):
        """An explicit reasoning: "" is not indistinguishable from absent.

        A truthy check on either side of the round trip (``if reasoning:``)
        treats "" the same as a missing key, silently turning an explicit
        empty reasoning into no reasoning key at all.
        """
        source = copy.deepcopy(self.SOURCE)
        source["sub_tasks"][1]["reasoning"] = ""
        back = self._round_trip(tmp_path, source=source)
        assert back["sub_tasks"][1]["reasoning"] == ""

    def test_skipped_sub_tasks_not_restored(self, tmp_path):
        """item_index reveals the gap but cannot fill it."""
        source = copy.deepcopy(self.SOURCE)
        source["sub_tasks"][1]["answer"] = "<SKIP>n/a"
        batch = tmp_path / "src"
        batch.mkdir()
        with open(batch / "a.json", "w") as f:
            json.dump(source, f)
        mid = tmp_path / "mid"
        DfVlmQaV1_0ToTaoVlReasonV1_0Converter().convert_dataset(batch, mid)
        out = tmp_path / "back"
        TaoVlReasonV1_0ToDfVlmQaV1_0Converter().convert_dataset(mid, out)
        with open(out / "a.json") as f:
            back = json.load(f)
        assert len(back["sub_tasks"]) == 2
        assert [s["question"] for s in back["sub_tasks"]] == ["first?", "third?"]
