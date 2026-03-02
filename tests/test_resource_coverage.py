"""Tests for ResourceMonitor coverage gaps — GPU detection, RAM detection, Ollama.

Covers the previously-uncovered platform-specific code paths in
``resource_monitor.py`` by mocking subprocess calls and platform detection.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch, mock_open

import pytest

from pinocchio.utils.resource_monitor import GPUInfo, ResourceMonitor, ResourceSnapshot


# ── Helper ──────────────────────────────────────────────────────────────

def _fresh_monitor() -> ResourceMonitor:
    """Return a monitor with no cached snapshot."""
    return ResourceMonitor()


# ── _detect_gpu: CUDA path ─────────────────────────────────────────────

class TestDetectGPU:
    """Cover _detect_gpu for CUDA, MPS, and ROCm branches."""

    @patch("pinocchio.utils.resource_monitor.shutil.which", side_effect=lambda cmd: cmd == "nvidia-smi")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output")
    def test_cuda_gpu_detected(self, mock_subproc, mock_which):
        mock_subproc.return_value = "NVIDIA RTX 4090, 24576, 20000\n"
        snap = ResourceSnapshot(cpu_count=8, cpu_count_physical=8,
                                ram_total_mb=32000, ram_available_mb=16000)
        ResourceMonitor._detect_gpu(snap)
        assert len(snap.gpus) == 1
        assert snap.gpus[0].backend == "cuda"
        assert snap.gpus[0].name == "NVIDIA RTX 4090"
        assert snap.gpus[0].vram_total_mb == 24576
        assert snap.gpus[0].vram_free_mb == 20000

    @patch("pinocchio.utils.resource_monitor.shutil.which", side_effect=lambda cmd: cmd == "nvidia-smi")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output")
    def test_cuda_multi_gpu(self, mock_subproc, mock_which):
        mock_subproc.return_value = (
            "NVIDIA RTX 4090, 24576, 20000\n"
            "NVIDIA RTX 4090, 24576, 18000\n"
        )
        snap = ResourceSnapshot(cpu_count=16, cpu_count_physical=16)
        ResourceMonitor._detect_gpu(snap)
        assert len(snap.gpus) == 2
        assert snap.gpus[0].backend == "cuda"
        assert snap.gpus[1].vram_free_mb == 18000

    @patch("pinocchio.utils.resource_monitor.platform.machine", return_value="x86_64")
    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Linux")
    @patch("pinocchio.utils.resource_monitor.shutil.which", side_effect=lambda cmd: cmd == "nvidia-smi")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output",
           side_effect=subprocess.CalledProcessError(1, "nvidia-smi"))
    def test_cuda_nvidia_smi_fails_gracefully(self, mock_subproc, mock_which, mock_sys, mock_mach):
        snap = ResourceSnapshot()
        ResourceMonitor._detect_gpu(snap)
        # Should not crash, no GPUs added (Linux x86_64 — no MPS fallback)
        assert len(snap.gpus) == 0

    @patch("pinocchio.utils.resource_monitor.shutil.which", return_value=None)
    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Darwin")
    @patch("pinocchio.utils.resource_monitor.platform.machine", return_value="arm64")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output")
    def test_apple_mps_detected(self, mock_subproc, mock_machine, mock_system, mock_which):
        mock_subproc.return_value = "Apple M2 Max"
        snap = ResourceSnapshot(ram_total_mb=32000, ram_available_mb=20000)
        ResourceMonitor._detect_gpu(snap)
        assert len(snap.gpus) == 1
        assert snap.gpus[0].backend == "mps"
        assert "Apple M2 Max" in snap.gpus[0].name
        assert snap.gpus[0].vram_total_mb == 32000  # unified memory

    @patch("pinocchio.utils.resource_monitor.shutil.which", return_value=None)
    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Darwin")
    @patch("pinocchio.utils.resource_monitor.platform.machine", return_value="arm64")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output",
           side_effect=FileNotFoundError("sysctl not found"))
    def test_apple_mps_fallback_name(self, mock_subproc, mock_machine, mock_system, mock_which):
        snap = ResourceSnapshot(ram_total_mb=16000, ram_available_mb=8000)
        ResourceMonitor._detect_gpu(snap)
        assert len(snap.gpus) == 1
        assert snap.gpus[0].name == "Apple Silicon (MPS)"
        assert snap.gpus[0].backend == "mps"

    @patch("pinocchio.utils.resource_monitor.shutil.which",
           side_effect=lambda cmd: cmd == "rocm-smi")
    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Linux")
    @patch("pinocchio.utils.resource_monitor.platform.machine", return_value="x86_64")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output")
    def test_rocm_gpu_detected(self, mock_subproc, mock_machine, mock_system, mock_which):
        mock_subproc.return_value = "GPU[0]: vram Total: 16384 MB\n"
        snap = ResourceSnapshot()
        ResourceMonitor._detect_gpu(snap)
        assert len(snap.gpus) == 1
        assert snap.gpus[0].backend == "rocm"

    @patch("pinocchio.utils.resource_monitor.shutil.which",
           side_effect=lambda cmd: cmd == "rocm-smi")
    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Linux")
    @patch("pinocchio.utils.resource_monitor.platform.machine", return_value="x86_64")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output",
           side_effect=subprocess.CalledProcessError(1, "rocm-smi"))
    def test_rocm_fails_gracefully(self, mock_subproc, mock_machine, mock_system, mock_which):
        snap = ResourceSnapshot()
        ResourceMonitor._detect_gpu(snap)
        assert len(snap.gpus) == 0

    @patch("pinocchio.utils.resource_monitor.shutil.which", return_value=None)
    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Linux")
    @patch("pinocchio.utils.resource_monitor.platform.machine", return_value="x86_64")
    def test_no_gpu_at_all(self, mock_machine, mock_system, mock_which):
        snap = ResourceSnapshot()
        ResourceMonitor._detect_gpu(snap)
        assert len(snap.gpus) == 0
        assert not snap.has_gpu


# ── _detect_ram ──────────────────────────────────────────────────────────

class TestDetectRAM:
    """Cover _detect_ram for Darwin, Linux, and fallback paths."""

    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Darwin")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output")
    def test_darwin_ram_detection(self, mock_subproc, mock_system):
        def _side(cmd, text=True):
            if "hw.memsize" in cmd:
                return str(32 * 1024 * 1024 * 1024)  # 32 GB
            if cmd == ["vm_stat"]:
                return (
                    "Mach Virtual Memory Statistics:\n"
                    "Pages free:                    500000.\n"
                    "Pages inactive:                300000.\n"
                    "Pages active:                  200000.\n"
                )
            return ""
        mock_subproc.side_effect = _side
        snap = ResourceSnapshot()
        ResourceMonitor._detect_ram(snap)
        assert snap.ram_total_mb == 32768
        # free_pages = 500000 + 300000 = 800000
        expected_avail = (800000 * 4096) // (1024 * 1024)
        assert snap.ram_available_mb == expected_avail

    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Linux")
    def test_linux_ram_detection(self, mock_system):
        meminfo = (
            "MemTotal:       16384000 kB\n"
            "MemFree:         2000000 kB\n"
            "MemAvailable:    8000000 kB\n"
        )
        with patch("builtins.open", mock_open(read_data=meminfo)):
            snap = ResourceSnapshot()
            ResourceMonitor._detect_ram(snap)
        assert snap.ram_total_mb == 16384000 // 1024
        assert snap.ram_available_mb == 8000000 // 1024

    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Darwin")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output",
           side_effect=FileNotFoundError("no sysctl"))
    def test_ram_detection_fallback(self, mock_subproc, mock_system):
        snap = ResourceSnapshot()
        ResourceMonitor._detect_ram(snap)
        assert snap.ram_total_mb == 8_000
        assert snap.ram_available_mb == 4_000


# ── _detect_ollama ───────────────────────────────────────────────────────

class TestDetectOllama:
    """Cover _detect_ollama for present, absent, error paths."""

    @patch("pinocchio.utils.resource_monitor.shutil.which", return_value="/usr/bin/ollama")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output")
    def test_ollama_running(self, mock_subproc, mock_which):
        mock_subproc.return_value = "NAME         ID       SIZE\nqwen2.5:7b   abc123   4.1 GB\n"
        snap = ResourceSnapshot()
        ResourceMonitor._detect_ollama(snap)
        assert snap.ollama_running is True

    @patch("pinocchio.utils.resource_monitor.shutil.which", return_value="/usr/bin/ollama")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output", return_value="")
    def test_ollama_not_running(self, mock_subproc, mock_which):
        snap = ResourceSnapshot()
        ResourceMonitor._detect_ollama(snap)
        assert snap.ollama_running is False

    @patch("pinocchio.utils.resource_monitor.shutil.which", return_value=None)
    def test_ollama_not_installed(self, mock_which):
        snap = ResourceSnapshot()
        ResourceMonitor._detect_ollama(snap)
        assert snap.ollama_running is False

    @patch("pinocchio.utils.resource_monitor.shutil.which", return_value="/usr/bin/ollama")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output",
           side_effect=subprocess.TimeoutExpired("ollama ps", 5))
    def test_ollama_timeout(self, mock_subproc, mock_which):
        snap = ResourceSnapshot()
        ResourceMonitor._detect_ollama(snap)
        assert snap.ollama_running is False


# ── _physical_cores ──────────────────────────────────────────────────────

class TestPhysicalCores:
    """Cover _physical_cores for Darwin, Linux, fallback."""

    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Darwin")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output")
    def test_darwin_physical_cores(self, mock_subproc, mock_system):
        mock_subproc.return_value = "10"
        assert ResourceMonitor._physical_cores() == 10

    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Linux")
    @patch("pinocchio.utils.resource_monitor.subprocess.check_output")
    def test_linux_physical_cores(self, mock_subproc, mock_system):
        mock_subproc.return_value = "16"
        assert ResourceMonitor._physical_cores() == 16

    @patch("pinocchio.utils.resource_monitor.platform.system", return_value="Windows")
    def test_fallback_uses_os_cpu_count(self, mock_system):
        result = ResourceMonitor._physical_cores()
        assert result >= 1
