"""Tests for the multimodal processors."""

from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch
from pathlib import Path

import pytest

from pinocchio.multimodal.text_processor import TextProcessor
from pinocchio.multimodal.vision_processor import VisionProcessor
from pinocchio.multimodal.audio_processor import AudioProcessor
from pinocchio.multimodal.video_processor import VideoProcessor
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import AgentRole
from pinocchio.utils.logger import PinocchioLogger


@pytest.fixture
def logger():
    return PinocchioLogger()


@pytest.fixture
def memory(tmp_data_dir):
    return MemoryManager(data_dir=tmp_data_dir)


# ------------------------------------------------------------------
# TextProcessor
# ------------------------------------------------------------------


class TestTextProcessor:
    def test_role(self, mock_llm, memory, logger):
        proc = TextProcessor(mock_llm, memory, logger)
        assert proc.role == AgentRole.TEXT_PROCESSOR

    def test_run_sends_task_and_text(self, mock_llm, memory, logger):
        mock_llm.ask.return_value = "This is a summary of the input."
        proc = TextProcessor(mock_llm, memory, logger)
        result = proc.run(task="summarise", text="Long document content here...")
        assert result == "This is a summary of the input."
        mock_llm.ask.assert_called_once()
        call_args = mock_llm.ask.call_args
        user_arg = call_args[1].get("user", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "summarise" in user_arg
        assert "Long document content" in user_arg


# ------------------------------------------------------------------
# VisionProcessor
# ------------------------------------------------------------------


class TestVisionProcessor:
    def test_role(self, mock_llm, memory, logger):
        proc = VisionProcessor(mock_llm, memory, logger)
        assert proc.role == AgentRole.VISION_PROCESSOR

    def test_run_with_url_images(self, mock_llm, memory, logger):
        mock_llm.chat.return_value = "A beautiful sunset over the ocean."
        mock_llm.build_vision_message.return_value = {"role": "user", "content": []}
        proc = VisionProcessor(mock_llm, memory, logger)
        result = proc.run(
            task="describe this image",
            image_paths=["https://example.com/sunset.jpg"],
        )
        assert "sunset" in result
        mock_llm.build_vision_message.assert_called_once()
        call_args = mock_llm.build_vision_message.call_args
        urls = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("image_urls", [])
        assert "https://example.com/sunset.jpg" in urls

    def test_encode_image_static_method(self):
        """Test that _encode_image produces a data URL."""
        fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        with patch.object(Path, "read_bytes", return_value=fake_bytes):
            with patch.object(Path, "suffix", new_callable=lambda: property(lambda self: ".png")):
                result = VisionProcessor._encode_image("test.png")
                assert result.startswith("data:image/png;base64,")


# ------------------------------------------------------------------
# AudioProcessor (Qwen3-VL native audio)
# ------------------------------------------------------------------


class TestAudioProcessor:
    def test_role(self, mock_llm, memory, logger):
        proc = AudioProcessor(mock_llm, memory, logger)
        assert proc.role == AgentRole.AUDIO_PROCESSOR

    def test_no_separate_openai_client(self, mock_llm, memory, logger):
        """AudioProcessor should NOT have its own openai client anymore."""
        proc = AudioProcessor(mock_llm, memory, logger)
        assert not hasattr(proc, "_oai")

    def test_run_sends_audio_via_build_audio_message(self, mock_llm, memory, logger):
        mock_llm.build_audio_message.return_value = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Task: summarise this audio"},
                {"type": "input_audio", "input_audio": {"data": "base64data", "format": "wav"}},
            ],
        }
        mock_llm.chat.return_value = "The speaker discusses AI advances in 2025."

        proc = AudioProcessor(mock_llm, memory, logger)
        result = proc.run(task="summarise this audio", audio_paths=["test.wav"])

        assert "AI" in result
        mock_llm.build_audio_message.assert_called_once()
        mock_llm.chat.assert_called_once()
        call_args = mock_llm.build_audio_message.call_args
        assert call_args[1]["audio_urls"] == ["test.wav"]

    def test_run_multiple_audio_files(self, mock_llm, memory, logger):
        mock_llm.build_audio_message.return_value = {"role": "user", "content": []}
        mock_llm.chat.return_value = "Combined audio analysis result."

        proc = AudioProcessor(mock_llm, memory, logger)
        result = proc.run(
            task="compare these recordings",
            audio_paths=["file1.wav", "file2.mp3"],
        )
        assert result == "Combined audio analysis result."
        call_args = mock_llm.build_audio_message.call_args
        assert call_args[1]["audio_urls"] == ["file1.wav", "file2.mp3"]


# ------------------------------------------------------------------
# VideoProcessor
# ------------------------------------------------------------------


class TestVideoProcessor:
    def test_role(self, mock_llm, memory, logger):
        proc = VideoProcessor(mock_llm, memory, logger)
        assert proc.role == AgentRole.VIDEO_PROCESSOR

    def test_extract_frames_calls_ffmpeg(self, mock_llm, memory, logger):
        proc = VideoProcessor(mock_llm, memory, logger)
        with patch("pinocchio.multimodal.video_processor.subprocess.run") as mock_run:
            with patch("pinocchio.multimodal.video_processor.Path.glob", return_value=[
                    Path("/tmp/frame_0001.jpg"),
                    Path("/tmp/frame_0002.jpg"),
            ]):
                mock_run.return_value = MagicMock(returncode=0)
                frames = proc.extract_frames("test.mp4", interval_sec=5.0, max_frames=2)
                assert len(frames) == 2
                mock_run.assert_called_once()

    def test_extract_frames_handles_ffmpeg_failure(self, mock_llm, memory, logger):
        proc = VideoProcessor(mock_llm, memory, logger)
        with patch("pinocchio.multimodal.video_processor.subprocess.run",
                    side_effect=FileNotFoundError("ffmpeg not found")):
            frames = proc.extract_frames("test.mp4")
            assert frames == []

    def test_run_native_mode_uses_build_video_message(self, mock_llm, memory, logger):
        """Default native_video=True sends video directly to Qwen3-VL."""
        mock_llm.build_video_message.return_value = {"role": "user", "content": []}
        mock_llm.chat.return_value = "Video shows a person walking in a park."

        proc = VideoProcessor(mock_llm, memory, logger)
        result = proc.run(task="describe this video", video_paths=["test.mp4"])

        assert "walking" in result
        mock_llm.build_video_message.assert_called_once()
        mock_llm.chat.assert_called_once()

    def test_run_fallback_mode(self, mock_llm, memory, logger):
        """native_video=False should use ffmpeg extraction path."""
        proc = VideoProcessor(mock_llm, memory, logger)
        mock_llm.ask.return_value = "Fallback analysis result."

        with patch.object(proc, "extract_frames", return_value=[]):
            with patch.object(proc, "extract_audio", return_value=None):
                result = proc.run(
                    task="describe this video",
                    video_paths=["test.mp4"],
                    native_video=False,
                )

        assert "Fallback" in result
        mock_llm.ask.assert_called_once()

    def test_run_native_multiple_videos(self, mock_llm, memory, logger):
        mock_llm.build_video_message.return_value = {"role": "user", "content": []}
        mock_llm.chat.return_value = "Scene description."

        proc = VideoProcessor(mock_llm, memory, logger)
        result = proc.run(
            task="compare these videos",
            video_paths=["vid1.mp4", "vid2.mp4"],
        )
        assert mock_llm.build_video_message.call_count == 2
        assert mock_llm.chat.call_count == 2
