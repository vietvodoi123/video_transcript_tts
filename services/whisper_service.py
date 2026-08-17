from dataclasses import asdict, dataclass
import json

from pathlib import Path
from runtime.cuda_runtime import setup_cuda_runtime

# Phải setup DLL trước khi import ctranslate2
setup_cuda_runtime()

import ctranslate2
from faster_whisper import WhisperModel

@dataclass
class TranscriptWord:
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord]

class WhisperService:

    def __init__(self, model_size="large-v3"):

        self.model_size = model_size
        self.device = self._detect_device()

        print(f"Model: {model_size}")
        print(f"Device: {self.device}")

        if self.device == "cuda":

            self.model = WhisperModel(
                model_size,
                device="cuda",
                compute_type="float16",
            )

        else:

            self.model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
            )

    @staticmethod
    def _detect_device():

        try:

            supported = ctranslate2.get_supported_compute_types(
                "cuda"
            )

            if "float16" in supported:

                print("CUDA backend detected.")
                print(
                    f"Supported CUDA compute types: {supported}"
                )

                return "cuda"

        except Exception as e:

            print(f"CUDA unavailable: {e}")

        print("Falling back to CPU.")

        return "cpu"

    def transcribe(
            self,
            audio_path,
    ):
        segments, info = self.model.transcribe(
            str(audio_path),
            language="zh",
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        results = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            words = []

            for word in segment.words or []:

                if (
                        word.start is None
                        or word.end is None
                ):
                    continue

                word_text = word.word.strip()

                if not word_text:
                    continue

                words.append(
                    TranscriptWord(
                        word=word_text,
                        start=word.start,
                        end=word.end,
                    )
                )

            if not words:
                continue

            results.append(
                TranscriptSegment(
                    index=len(results) + 1,
                    start=words[0].start,
                    end=words[-1].end,
                    text=text,
                    words=words,
                )
            )

        return results, info

    def transcribe_to_json(
            self,
            audio_path,
            output_path,
            chunk_id,
    ):
        segments, info = self.transcribe(
            audio_path
        )

        data = {
            "version": 2,
            "chunk_id": chunk_id,
            "language": info.language,
            "language_probability": (
                info.language_probability
            ),
            "segments": [
                asdict(segment)
                for segment in segments
            ],
        }

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
                output_path,
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return output_path