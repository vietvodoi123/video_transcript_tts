from pathlib import Path
import wave

from models.subtitle import Subtitle


class TimelineMerger:

    def merge(
        self,
        *,
        subtitles: list[Subtitle],
        audio_files: list[str | Path],
        output_file: str | Path,
    ):

        if len(subtitles) != len(audio_files):
            raise ValueError(
                "subtitle count != audio count"
            )

        output_file = Path(output_file)

        sample_rate = None
        sample_width = None
        channels = None

        with wave.open(
            str(output_file),
            "wb",
        ) as out:

            current_frame = 0

            for subtitle, audio_file in zip(
                subtitles,
                audio_files,
            ):

                audio_file = Path(audio_file)

                with wave.open(
                    str(audio_file),
                    "rb",
                ) as wav:

                    if sample_rate is None:

                        sample_rate = wav.getframerate()
                        sample_width = wav.getsampwidth()
                        channels = wav.getnchannels()

                        out.setnchannels(channels)
                        out.setsampwidth(sample_width)
                        out.setframerate(sample_rate)

                    target_frame = int(
                        subtitle.start * sample_rate
                    )

                    if target_frame > current_frame:

                        silence_frames = (
                            target_frame
                            - current_frame
                        )

                        out.writeframes(
                            b"\x00"
                            * silence_frames
                            * sample_width
                            * channels
                        )

                        current_frame = target_frame

                    frames = wav.readframes(
                        wav.getnframes()
                    )

                    out.writeframes(frames)

                    current_frame += wav.getnframes()