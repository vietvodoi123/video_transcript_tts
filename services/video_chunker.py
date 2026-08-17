import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class VideoChunk:

    id: str
    index: int

    start: float
    end: float

    duration: float


@dataclass
class Silence:

    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start


class VideoChunker:

    def __init__(
        self,
        target_duration=5 * 60,
        min_duration=4 * 60,
        max_duration=6 * 60,
        silence_duration=0.8,
    ):
        self.target_duration = target_duration
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_duration = silence_duration

    # =====================================================
    # Duration
    # =====================================================

    def get_duration(self, video_path):

        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        return float(result.stdout.strip())

    # =====================================================
    # Detect silence
    # =====================================================

    def detect_silences(self, video_path):

        command = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-af",
            (
                f"silencedetect="
                f"noise=-35dB:"
                f"d={self.silence_duration}"
            ),
            "-f",
            "null",
            "-",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        output = result.stderr

        silences = []

        current_start = None

        for line in output.splitlines():

            start_match = re.search(
                r"silence_start:\s*([\d.]+)",
                line,
            )

            if start_match:
                current_start = float(
                    start_match.group(1)
                )
                continue

            end_match = re.search(
                r"silence_end:\s*([\d.]+)",
                line,
            )

            if (
                end_match
                and current_start is not None
            ):

                end = float(
                    end_match.group(1)
                )

                silences.append(
                    Silence(
                        start=current_start,
                        end=end,
                    )
                )

                current_start = None

        return silences

    # =====================================================
    # Find best cut
    # =====================================================

    def find_cut_point(
        self,
        current_start,
        duration,
        silences,
    ):

        target = (
            current_start
            + self.target_duration
        )

        min_cut = (
            current_start
            + self.min_duration
        )

        max_cut = (
            current_start
            + self.max_duration
        )

        candidates = []

        for silence in silences:

            # Chỉ xét silence nằm trong
            # vùng cho phép.

            if (
                silence.end < min_cut
                or silence.start > max_cut
            ):
                continue

            # Điểm cắt nằm giữa silence.

            cut_point = (
                silence.start
                + silence.duration / 2
            )

            if (
                cut_point >= min_cut
                and cut_point <= max_cut
            ):
                candidates.append(
                    cut_point
                )

        if not candidates:

            return min(
                max_cut,
                duration,
            )

        return min(
            candidates,
            key=lambda x: abs(
                x - target
            ),
        )

    # =====================================================
    # Create chunks
    # =====================================================

    def create_chunks(
        self,
        video_path,
        manifest_path,
    ):

        video_path = Path(video_path)
        manifest_path = Path(manifest_path)

        if not video_path.exists():
            raise FileNotFoundError(
                video_path
            )

        duration = self.get_duration(
            video_path
        )

        print(
            f"Video duration: {duration:.2f}s"
        )

        print(
            "Detecting silence..."
        )

        silences = self.detect_silences(
            video_path
        )

        print(
            f"Detected {len(silences)} "
            f"silence regions."
        )

        chunks = []

        current_start = 0.0
        index = 1

        while current_start < duration:

            if (
                duration - current_start
                <= self.max_duration
            ):
                end = duration

            else:
                end = self.find_cut_point(
                    current_start,
                    duration,
                    silences,
                )

            chunk = VideoChunk(
                id=f"chunk_{index:04d}",
                index=index,
                start=current_start,
                end=end,
                duration=end - current_start,
            )

            chunks.append(chunk)

            current_start = end
            index += 1

        manifest = {
            "version": 2,
            "source": str(video_path),
            "duration": duration,

            "chunking": {
                "strategy": "silence",
                "target_duration": (
                    self.target_duration
                ),
                "min_duration": (
                    self.min_duration
                ),
                "max_duration": (
                    self.max_duration
                ),
                "silence_duration": (
                    self.silence_duration
                ),
            },

            "chunks": [
                asdict(chunk)
                for chunk in chunks
            ],
        }

        manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                manifest,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return chunks