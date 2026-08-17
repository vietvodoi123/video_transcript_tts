from pathlib import Path
import wave

from piper.voice import PiperVoice


class PiperTTSService:

    def __init__(
        self,
        *,
        model_dir: str | Path,
        voice: str,
    ):
        model_dir = Path(model_dir)

        self._voice = PiperVoice.load(
            model_path=model_dir / f"{voice}.onnx",
            config_path=model_dir / f"{voice}.onnx.json",
        )

        self.sample_rate = self._voice.config.sample_rate

    def synthesize_text(
        self,
        *,
        text: str,
        output_file: str | Path,
    ):
        """
        Sinh audio cho một đoạn text.
        """

        text = text.strip()

        if not text:
            return

        output_file = Path(output_file)

        with wave.open(str(output_file), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)

            for chunk in self._voice.synthesize(text):
                wav.writeframes(chunk.audio_int16_bytes)

    def synthesize_many(
        self,
        *,
        texts: list[str],
        output_file: str | Path,
    ):
        """
        Ghép nhiều đoạn text thành một file wav.
        """

        output_file = Path(output_file)

        with wave.open(str(output_file), "wb") as wav:

            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)

            for index, text in enumerate(texts):

                text = text.strip()

                if not text:
                    continue

                print(
                    f"TTS {index + 1}/{len(texts)}"
                )

                for chunk in self._voice.synthesize(text):
                    wav.writeframes(chunk.audio_int16_bytes)

    # Backward compatibility
    def synthesize_to_wav(
        self,
        *,
        texts: list[str],
        output_file: str | Path,
    ):
        self.synthesize_many(
            texts=texts,
            output_file=output_file,
        )