from pathlib import Path

from services.piper_tts_service import PiperTTSService
from services.text_splitter import TextSplitter


class TextPipeline:

    def __init__(
        self,
        *,
        model_dir: str | Path,
        voice: str,
    ):
        self._tts = PiperTTSService(
            model_dir=model_dir,
            voice=voice,
        )

        self._splitter = TextSplitter()

    def run(
        self,
        *,
        text: str,
        output_file: str | Path,
    ):

        text = text.strip()

        if not text:
            raise ValueError("Text is empty.")

        texts = self._splitter.split(text)

        self._tts.synthesize_many(
            texts=texts,
            output_file=output_file,
        )

    def run_file(
        self,
        *,
        input_file: str | Path,
        output_file: str | Path,
    ):

        text = Path(input_file).read_text(
            encoding="utf-8",
        )

        self.run(
            text=text,
            output_file=output_file,
        )