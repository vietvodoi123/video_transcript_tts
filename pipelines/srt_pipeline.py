from pathlib import Path
import shutil

from services.piper_tts_service import PiperTTSService
from services.srt_parser import SrtParser
from services.timeline_merger import TimelineMerger


class SrtPipeline:

    def __init__(
        self,
        *,
        model_dir: str | Path,
        voice: str,
        temp_dir: str | Path,
    ):
        self._tts = PiperTTSService(
            model_dir=model_dir,
            voice=voice,
        )

        self._parser = SrtParser()

        self._merger = TimelineMerger()

        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(
            self,
            *,
            input_file: str | Path,
            output_file: str | Path,
    ):

        self._clean_temp()

        try:

            subtitles = self._parser.parse(
                input_file
            )

            audio_files = []

            for subtitle in subtitles:
                wav_file = (
                        self._temp_dir
                        / f"{subtitle.index:06d}.wav"
                )

                self._tts.synthesize_text(
                    text=subtitle.text,
                    output_file=wav_file,
                )

                audio_files.append(wav_file)

            self._merger.merge(
                subtitles=subtitles,
                audio_files=audio_files,
                output_file=output_file,
            )

        finally:
            self._clean_temp()

    def _clean_temp(self):
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)

        self._temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )