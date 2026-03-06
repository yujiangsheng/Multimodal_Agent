"""AudioProcessor -- sub-agent for audio modality processing.

Handles audio understanding, transcription, and cross-modal reasoning
between audio and text.

Backend: **Qwen3-VL** -- a native multimodal model that accepts
audio input directly alongside text, eliminating the need for a
separate transcription step.

Skills / Capabilities
---------------------
1. **Native Audio Understanding**
   Send raw audio directly to Qwen3-VL for end-to-end understanding
   without a separate transcription model.

2. **Speech-to-Text Transcription**
   Extract text from spoken audio, supporting multiple languages via the
   omni-modal model.

3. **Tone & Emotion Analysis**
   Infer the speaker's tone, emotion, and emphasis from the raw audio
   signal -- richer than text-only analysis.

4. **Audio Summarisation**
   Produce a concise summary of long audio recordings (podcasts,
   meetings, lectures).

5. **Speaker Identification Hints**
   When multiple speakers are present, attempt to segment by speaker
   turns.

6. **Audio to Text Bridging**
   Generate structured textual representations of audio content suitable
   for downstream reasoning by other agents.

7. **Music / Sound Event Description**
   For non-speech audio, describe the sounds, instruments, rhythms, and
   mood conveyed.

8. **Multilingual Support**
   Handle audio in any language supported by Qwen3-VL and auto-detect
   the language spoken.
"""

from __future__ import annotations

from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.models.enums import AgentRole

_SYSTEM_PROMPT = (
    "You are the Audio Processor of Pinocchio, a self-evolving multimodal AI.\n"
    "You specialise in audio understanding.  Analyse the provided audio\n"
    "file(s) carefully -- including speech content, tone, emotion, background\n"
    "sounds, and any other audible information -- and complete the requested task."
)


class AudioProcessor(BaseAgent):
    """Processes audio-modality tasks natively via Qwen3-VL.

    Audio is sent directly to the omni-modal LLM as ``input_audio``
    content parts, enabling richer understanding (tone, emotion,
    overlapping speakers, music).
    """

    role = AgentRole.AUDIO_PROCESSOR

    # No extra openai client needed -- we reuse self.llm (LLMClient)

    def run(self, *, task: str, audio_paths: list[str], **kwargs: Any) -> str:  # type: ignore[override]
        """Execute an audio task by sending audio directly to Qwen3-VL.

        Parameters
        ----------
        task : Description of the audio task (e.g., "summarise this meeting",
               "what is the speaker's tone?").
        audio_paths : Paths or URLs to audio files to process.
        """
        self._log(f"Audio task: {task} -- {len(audio_paths)} file(s)")

        # Build a multimodal message with native audio content
        audio_msg = self.llm.build_audio_message(
            text=f"Task: {task}",
            audio_urls=audio_paths,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            audio_msg,
        ]
        result = self.llm.chat(messages)
        self._log(f"Audio analysis complete -- {len(result)} chars")
        return result
