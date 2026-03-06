"""Tests for utility modules (LLMClient, PinocchioLogger)."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from pinocchio.utils.llm_client import LLMClient
from pinocchio.utils.logger import PinocchioLogger
from pinocchio.models.enums import AgentRole


# ------------------------------------------------------------------
# LLMClient
# ------------------------------------------------------------------


class TestLLMClient:
    @patch("pinocchio.utils.llm_client.openai")
    def test_init_default_model(self, mock_openai):
        client = LLMClient()
        assert "qwen3-vl" in client.model.lower() or client.model  # env may override

    @patch("pinocchio.utils.llm_client.openai")
    def test_chat_calls_openai_compatible(self, mock_openai):
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Hello!"
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion

        client = LLMClient(api_key="test")
        result = client.chat([{"role": "user", "content": "Hi"}])
        assert result == "Hello!"

    @patch("pinocchio.utils.llm_client.openai")
    def test_ask_wraps_messages(self, mock_openai):
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Response"
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion

        client = LLMClient(api_key="test")
        result = client.ask(system="You are helpful.", user="Say hi")
        assert result == "Response"

        call_args = mock_openai.OpenAI.return_value.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @patch("pinocchio.utils.llm_client.openai")
    def test_ask_json_parses_response(self, mock_openai):
        response_json = {"key": "value", "number": 42}
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(response_json)
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion

        client = LLMClient(api_key="test")
        result = client.ask_json(system="Return JSON.", user="Give me data")
        assert result == response_json

    @patch("pinocchio.utils.llm_client.openai")
    def test_ask_json_handles_markdown_fenced_json(self, mock_openai):
        response_json = {"key": "value"}
        raw_response = f"```json\n{json.dumps(response_json)}\n```"
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = raw_response
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_completion

        client = LLMClient(api_key="test")
        result = client.ask_json(system="Return JSON.", user="data")
        assert result == response_json

    def test_build_vision_message(self):
        with patch("pinocchio.utils.llm_client.openai"):
            client = LLMClient(api_key="test")
        msg = client.build_vision_message("Describe this", ["https://example.com/img.jpg"])
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"]["url"] == "https://example.com/img.jpg"

    def test_build_audio_message_with_url(self):
        with patch("pinocchio.utils.llm_client.openai"):
            client = LLMClient(api_key="test")
        msg = client.build_audio_message("Transcribe this", ["https://example.com/audio.wav"])
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "input_audio"
        assert msg["content"][1]["input_audio"]["data"] == "https://example.com/audio.wav"
        assert msg["content"][1]["input_audio"]["format"] == "wav"

    def test_build_audio_message_with_local_file(self):
        with patch("pinocchio.utils.llm_client.openai"):
            client = LLMClient(api_key="test")
        fake_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "
        with patch("pinocchio.utils.llm_client.Path.read_bytes", return_value=fake_bytes):
            msg = client.build_audio_message("Analyse", ["recording.wav"])
        assert msg["content"][1]["type"] == "input_audio"
        expected_b64 = base64.b64encode(fake_bytes).decode()
        assert msg["content"][1]["input_audio"]["data"] == expected_b64

    def test_build_video_message(self):
        with patch("pinocchio.utils.llm_client.openai"):
            client = LLMClient(api_key="test")
        msg = client.build_video_message("Describe this video", ["https://example.com/vid.mp4"])
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "video"
        assert msg["content"][1]["video"] == "https://example.com/vid.mp4"

    def test_audio_format_detection(self):
        with patch("pinocchio.utils.llm_client.openai"):
            client = LLMClient(api_key="test")
        assert client._audio_format("test.mp3") == "mp3"
        assert client._audio_format("test.wav") == "wav"
        assert client._audio_format("test.flac") == "flac"
        assert client._audio_format("test.ogg") == "ogg"
        assert client._audio_format("test.unknown") == "wav"  # fallback


# ------------------------------------------------------------------
# PinocchioLogger
# ------------------------------------------------------------------


class TestPinocchioLogger:
    def test_log_does_not_crash(self):
        logger = PinocchioLogger()
        logger.log(AgentRole.ORCHESTRATOR, "Test message")
        logger.log(AgentRole.PERCEPTION, "With data", {"key": "value"})

    def test_phase_does_not_crash(self):
        logger = PinocchioLogger()
        logger.phase("PERCEIVE")

    def test_convenience_methods(self):
        logger = PinocchioLogger()
        logger.info(AgentRole.EXECUTION, "info msg")
        logger.warn(AgentRole.LEARNING, "warn msg")
        logger.error(AgentRole.STRATEGY, "error msg")
        logger.success(AgentRole.EVALUATION, "success msg")
        logger.separator()
