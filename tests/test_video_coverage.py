"""Tests for VideoProcessor coverage gaps — extract_audio and fallback path.

Covers the previously-uncovered ``extract_audio``, ``_run_fallback`` with
frame analysis + audio analysis fusion, and multi-video fallback.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pinocchio.models.enums import AgentRole
from pinocchio.multimodal.video_processor import VideoProcessor
from pinocchio.utils.logger import PinocchioLogger
from pinocchio.memory.memory_manager import MemoryManager


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_video_proc(mock_llm=None):
    llm = mock_llm or MagicMock()
    mem = MagicMock(spec=MemoryManager)
    logger = PinocchioLogger()
    return VideoProcessor(llm, mem, logger)


# ── extract_audio ────────────────────────────────────────────────────────

class TestExtractAudio:
    """Cover the extract_audio method."""

    @patch("pinocchio.multimodal.video_processor.subprocess.run")
    def test_extract_audio_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        vp = _make_video_proc()
        result = vp.extract_audio("/tmp/test.mp4")
        assert result is not None
        assert result.endswith(".wav")
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["check"] is True

    @patch("pinocchio.multimodal.video_processor.subprocess.run",
           side_effect=subprocess.CalledProcessError(1, "ffmpeg"))
    def test_extract_audio_ffmpeg_error(self, mock_run):
        vp = _make_video_proc()
        result = vp.extract_audio("/tmp/test.mp4")
        assert result is None

    @patch("pinocchio.multimodal.video_processor.subprocess.run",
           side_effect=FileNotFoundError("ffmpeg not found"))
    def test_extract_audio_ffmpeg_not_installed(self, mock_run):
        vp = _make_video_proc()
        result = vp.extract_audio("/tmp/test.mp4")
        assert result is None


# ── _run_fallback ────────────────────────────────────────────────────────

class TestRunFallback:
    """Cover the _run_fallback code path with frame + audio analysis."""

    def test_fallback_with_frames_and_audio(self):
        """Full fallback: frames extracted, audio extracted, both analysed."""
        llm = MagicMock()
        llm.ask.return_value = "Fused video analysis: action scene with dialogue"
        vp = _make_video_proc(llm)

        vision_proc = MagicMock()
        vision_proc.run.return_value = "A person running in a park"

        audio_proc = MagicMock()
        audio_proc.run.return_value = "Speech: 'Let's go!'"

        with patch.object(vp, "extract_frames", return_value=["/tmp/frame_0001.jpg"]):
            with patch.object(vp, "extract_audio", return_value="/tmp/audio.wav"):
                result = vp.run(
                    task="Describe the video",
                    video_paths=["/tmp/test.mp4"],
                    vision_processor=vision_proc,
                    audio_processor=audio_proc,
                    native_video=False,
                )

        assert "Fused video analysis" in result
        vision_proc.run.assert_called_once()
        audio_proc.run.assert_called_once()
        llm.ask.assert_called_once()

    def test_fallback_multiple_frames_parallel(self):
        """Fallback with multiple frames triggers parallel analysis."""
        llm = MagicMock()
        llm.ask.return_value = "Multi-frame analysis result"
        vp = _make_video_proc(llm)

        vision_proc = MagicMock()
        vision_proc.run.side_effect = lambda **kw: f"Description for {kw.get('task', '')}"

        with patch.object(vp, "extract_frames",
                          return_value=["/tmp/f1.jpg", "/tmp/f2.jpg", "/tmp/f3.jpg"]):
            with patch.object(vp, "extract_audio", return_value=None):
                result = vp.run(
                    task="Analyse video",
                    video_paths=["/tmp/test.mp4"],
                    vision_processor=vision_proc,
                    audio_processor=None,
                    native_video=False,
                )

        assert result == "Multi-frame analysis result"
        assert vision_proc.run.call_count == 3

    def test_fallback_no_frames_no_audio(self):
        """Fallback with no frames and no audio still produces fusion output."""
        llm = MagicMock()
        llm.ask.return_value = "Minimal analysis"
        vp = _make_video_proc(llm)

        with patch.object(vp, "extract_frames", return_value=[]):
            with patch.object(vp, "extract_audio", return_value=None):
                result = vp.run(
                    task="Analyse video",
                    video_paths=["/tmp/test.mp4"],
                    vision_processor=None,
                    audio_processor=None,
                    native_video=False,
                )

        assert result == "Minimal analysis"
        # Fusion prompt should contain "(no frames extracted)" and "(no audio)"
        call_user = llm.ask.call_args[1]["user"]
        assert "(no frames extracted)" in call_user
        assert "(no audio)" in call_user

    def test_fallback_multi_video(self):
        """Fallback processes multiple videos and joins results."""
        llm = MagicMock()
        llm.ask.side_effect = ["Video 1 result", "Video 2 result"]
        vp = _make_video_proc(llm)

        with patch.object(vp, "extract_frames", return_value=[]):
            with patch.object(vp, "extract_audio", return_value=None):
                result = vp.run(
                    task="Compare videos",
                    video_paths=["/tmp/a.mp4", "/tmp/b.mp4"],
                    native_video=False,
                )

        assert "Video 1 result" in result
        assert "Video 2 result" in result
        assert "---" in result  # separator


# ── vision_processor _encode_image with real file ────────────────────────

class TestVisionEncodeCoverage:
    """Cover the local-file branch in VisionProcessor.run()."""

    @patch("pinocchio.multimodal.vision_processor.Path")
    def test_encode_image_png(self, mock_path_cls):
        from pinocchio.multimodal.vision_processor import VisionProcessor as VP
        mock_path = MagicMock()
        mock_path.suffix = ".png"
        mock_path.read_bytes.return_value = b"\x89PNG\r\n"
        mock_path_cls.return_value = mock_path
        result = VP._encode_image("/tmp/test.png")
        assert result.startswith("data:image/png;base64,")

    @patch("pinocchio.multimodal.vision_processor.Path")
    def test_encode_image_jpg(self, mock_path_cls):
        from pinocchio.multimodal.vision_processor import VisionProcessor as VP
        mock_path = MagicMock()
        mock_path.suffix = ".jpg"
        mock_path.read_bytes.return_value = b"\xff\xd8\xff\xe0"
        mock_path_cls.return_value = mock_path
        result = VP._encode_image("/tmp/test.jpg")
        assert result.startswith("data:image/jpeg;base64,")
