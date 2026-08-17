import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    PROJECT_ROOT / ".env",
    override=True,
)


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


class OpenAIWhisperService:

    def __init__(
        self,
        model="whisper-1",
    ):
        self.model = model

        api_key = __import__("os").getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in .env"
            )

        self.client = OpenAI(
            api_key=api_key
        )

        print(
            f"OpenAI transcription model: "
            f"{self.model}"
        )

    def transcribe(
        self,
        audio_path,
    ):

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                audio_path
            )

        with open(
            audio_path,
            "rb",
        ) as audio_file:

            response = (
                self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language="zh",
                    response_format="verbose_json",
                    timestamp_granularities=[
                        "segment",
                        "word",
                    ],
                )
            )

        print(
            f"Detected language: "
            f"{getattr(response, 'language', None)}"
        )

        # -------------------------------------------------
        # OpenAI trả word timestamps ở response.words
        # -------------------------------------------------

        response_words = (
            getattr(
                response,
                "words",
                None,
            )
            or []
        )

        words = []

        for word in response_words:

            if (
                word.start is None
                or word.end is None
            ):
                continue

            text = word.word.strip()

            if not text:
                continue

            words.append(
                TranscriptWord(
                    word=text,
                    start=float(word.start),
                    end=float(word.end),
                )
            )

        # -------------------------------------------------
        # Segment text nằm ở response.segments
        # -------------------------------------------------

        response_segments = (
            getattr(
                response,
                "segments",
                None,
            )
            or []
        )

        segments = []

        for index, segment in enumerate(
            response_segments,
            start=1,
        ):

            text = segment.text.strip()

            if not text:
                continue

            segment_start = float(
                segment.start
            )

            segment_end = float(
                segment.end
            )

            # -------------------------------------------------
            # Gắn các word nằm trong segment
            # -------------------------------------------------

            segment_words = [
                word
                for word in words
                if (
                    word.start >= segment_start
                    and word.end <= segment_end
                )
            ]

            # -------------------------------------------------
            # Nếu có word timestamp,
            # ưu tiên first/last word.
            # -------------------------------------------------

            if segment_words:

                start = (
                    segment_words[0].start
                )

                end = (
                    segment_words[-1].end
                )

            else:

                start = segment_start
                end = segment_end

            segments.append(
                TranscriptSegment(
                    index=index,
                    start=start,
                    end=end,
                    text=text,
                    words=segment_words,
                )
            )

        return segments, response

    def transcribe_to_json(
        self,
        audio_path,
        output_path,
        chunk_id,
    ):

        segments, response = self.transcribe(
            audio_path
        )

        data = {
            "version": 3,
            "provider": "openai",
            "model": self.model,
            "chunk_id": chunk_id,
            "language": getattr(
                response,
                "language",
                "zh",
            ),
            "segments": [
                asdict(segment)
                for segment in segments
            ],
        }

        output_path = Path(
            output_path
        )

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