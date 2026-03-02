"""Pinocchio multimodal processor modules.

Each processor handles one input modality and delegates to the LLM
(Qwen2.5-Omni) for understanding:

* :class:`TextProcessor`   — text understanding, generation, translation
* :class:`VisionProcessor` — image understanding, VQA, OCR, captioning
* :class:`AudioProcessor`  — speech-to-text, tone analysis, audio summary
* :class:`VideoProcessor`  — video understanding (native or ffmpeg fallback)
"""

from pinocchio.multimodal.text_processor import TextProcessor
from pinocchio.multimodal.vision_processor import VisionProcessor
from pinocchio.multimodal.audio_processor import AudioProcessor
from pinocchio.multimodal.video_processor import VideoProcessor

__all__ = [
    "TextProcessor",
    "VisionProcessor",
    "AudioProcessor",
    "VideoProcessor",
]
