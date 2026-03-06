"""VideoProcessor -- sub-agent for video modality processing.

Handles video understanding via **Qwen3-VL** which natively
accepts video input -- no manual ffmpeg frame extraction or separate
audio transcription step is required.

A fallback path using ffmpeg + VisionProcessor + AudioProcessor is
retained for scenarios where the video is too large or the model
endpoint does not support direct video ingestion.

Skills / Capabilities
---------------------
1. **Native Video Understanding**
   Send video directly to Qwen3-VL for end-to-end analysis of
   both visual and audio streams in a single call.

2. **Key-Frame Extraction (fallback)**
   Sample representative frames via ffmpeg when native video input
   is not available or the file exceeds size limits.

3. **Frame-by-Frame Analysis (fallback)**
   Analyse extracted key frames using the VisionProcessor.

4. **Temporal Reasoning**
   Reason about the *sequence* and *timing* of events in a video.

5. **Video Summarisation**
   Produce comprehensive summaries combining visual and audio.

6. **Action / Event Detection**
   Identify and describe significant actions or events.

7. **Cross-Modal Video Understanding**
   Fuse visual and audio analysis for unified understanding.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.models.enums import AgentRole

_SYSTEM_PROMPT = (
    "You are the Video Processor of Pinocchio, a self-evolving multimodal AI.\n"
    "You specialise in video understanding through temporal visual and audio analysis.\n"
    "Given the video content (or extracted key-frame descriptions and audio),\n"
    "complete the task."
)


class VideoProcessor(BaseAgent):
    """Processes video-modality tasks, preferring native Qwen3-VL video input.

    Primary path:  send video directly to Qwen3-VL via ``build_video_message``.
    Fallback path: extract frames + audio via ffmpeg and delegate to
                   VisionProcessor / AudioProcessor.
    """

    role = AgentRole.VIDEO_PROCESSOR

    def extract_frames(
        self,
        video_path: str,
        interval_sec: float = 5.0,
        max_frames: int = 10,
    ) -> list[str]:
        """Extract key frames from a video using ffmpeg.

        Returns a list of file paths to the extracted frame images.
        """
        self._log(f"Extracting frames from {video_path} every {interval_sec}s...")
        out_dir = Path(tempfile.mkdtemp(prefix="pinocchio_frames_"))
        pattern = str(out_dir / "frame_%04d.jpg")

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=1/{interval_sec}",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            pattern,
            "-y", "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self._warn(f"Frame extraction failed: {e}")
            return []

        frames = sorted(out_dir.glob("frame_*.jpg"))
        self._log(f"Extracted {len(frames)} frames")
        return [str(f) for f in frames[:max_frames]]

    def extract_audio(self, video_path: str) -> str | None:
        """Extract the audio track from a video file.

        Returns path to the extracted .wav file, or None on failure.
        """
        out_path = Path(tempfile.mktemp(suffix=".wav", prefix="pinocchio_audio_"))
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            str(out_path),
            "-y", "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(out_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self._warn(f"Audio extraction failed: {e}")
            return None

    def run(  # type: ignore[override]
        self,
        *,
        task: str,
        video_paths: list[str],
        vision_processor: Any = None,
        audio_processor: Any = None,
        native_video: bool = True,
        **kwargs: Any,
    ) -> str:
        """Execute a video task.

        Parameters
        ----------
        task : Description of the video task.
        video_paths : Paths to video files.
        vision_processor : A VisionProcessor instance for frame analysis (fallback).
        audio_processor : An AudioProcessor instance for audio analysis (fallback).
        native_video : If True (default), send video directly to Qwen3-VL.
                       Set to False to force the ffmpeg fallback path.
        """
        self._log(f"Video task: {task} -- {len(video_paths)} file(s)")

        if native_video:
            return self._run_native(task, video_paths)
        return self._run_fallback(task, video_paths, vision_processor, audio_processor)

    # ------------------------------------------------------------------
    # Primary path -- native Qwen3-VL video understanding
    # ------------------------------------------------------------------

    def _run_native(self, task: str, video_paths: list[str]) -> str:
        """Send video directly to Qwen3-VL via ``build_video_message``.

        Multiple videos are processed in parallel threads.
        """
        self._log("Using native video input (Qwen3-VL)")

        def _process_one(vp: str) -> str:
            video_msg = self.llm.build_video_message(text=f"Task: {task}", video_urls=[vp])
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                video_msg,
            ]
            return self.llm.chat(messages)

        if len(video_paths) == 1:
            all_descriptions = [_process_one(video_paths[0])]
        else:
            self._log(f"Parallel processing {len(video_paths)} videos")
            with ThreadPoolExecutor(max_workers=min(4, len(video_paths))) as pool:
                futures = {pool.submit(_process_one, vp): i for i, vp in enumerate(video_paths)}
                all_descriptions = [""] * len(video_paths)
                for fut in as_completed(futures):
                    all_descriptions[futures[fut]] = fut.result()

        final = "\n\n---\n\n".join(all_descriptions)
        self._log(f"Video analysis complete -- {len(final)} chars")
        return final

    # ------------------------------------------------------------------
    # Fallback path -- ffmpeg frame/audio extraction
    # ------------------------------------------------------------------

    def _run_fallback(
        self,
        task: str,
        video_paths: list[str],
        vision_processor: Any,
        audio_processor: Any,
    ) -> str:
        """Extract frames + audio via ffmpeg and delegate to sub-processors.

        Frame analysis and audio analysis run in parallel threads.
        """
        self._log("Using fallback path (ffmpeg + sub-processors)")
        all_descriptions: list[str] = []
        temp_paths: list[Path] = []  # track temp files/dirs for cleanup

        for vp in video_paths:
            # --- Launch frame extraction and audio extraction concurrently ---
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="video-extract") as pool:
                frame_future = pool.submit(self.extract_frames, vp)
                audio_future = pool.submit(self.extract_audio, vp)
                frames = frame_future.result()
                audio_path = audio_future.result()

            # Track temp paths for cleanup
            if frames:
                temp_paths.append(Path(frames[0]).parent)
            if audio_path:
                temp_paths.append(Path(audio_path))

            # --- Analyse frames in parallel ---
            frame_descs: list[str] = []
            if frames and vision_processor:
                def _analyse_frame(idx_frame: tuple[int, str]) -> str:
                    i, frame = idx_frame
                    desc = vision_processor.run(
                        task=f"Describe frame {i+1} of the video in detail",
                        image_paths=[frame],
                    )
                    return f"[Frame {i+1}] {desc}"

                if len(frames) == 1:
                    frame_descs = [_analyse_frame((0, frames[0]))]
                else:
                    self._log(f"Parallel analysis of {len(frames)} frames")
                    with ThreadPoolExecutor(
                        max_workers=min(4, len(frames)),
                        thread_name_prefix="frame-analysis",
                    ) as pool:
                        futures = {
                            pool.submit(_analyse_frame, (i, f)): i
                            for i, f in enumerate(frames)
                        }
                        frame_descs = [""] * len(frames)
                        for fut in as_completed(futures):
                            frame_descs[futures[fut]] = fut.result()

            # --- Analyse audio (may already be done if few frames) ---
            audio_desc = ""
            if audio_path and audio_processor:
                audio_desc = audio_processor.run(
                    task="Transcribe and describe the audio content",
                    audio_paths=[audio_path],
                )

            # --- Fuse visual + audio for temporal reasoning ---
            frame_text = "\n".join(frame_descs if frame_descs else ["(no frames extracted)"])
            audio_text = audio_desc or "(no audio)"
            fusion_prompt = (
                f"Task: {task}\n\n"
                f"=== KEY FRAMES ===\n{frame_text}"
                f"\n\n=== AUDIO ===\n{audio_text}"
            )

            result = self.llm.ask(system=_SYSTEM_PROMPT, user=fusion_prompt)
            all_descriptions.append(result)

        final = "\n\n---\n\n".join(all_descriptions)
        self._log(f"Video analysis complete -- {len(final)} chars")

        # Cleanup temporary files
        for p in temp_paths:
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink(missing_ok=True)
            except Exception:
                pass

        return final
