import json
from pathlib import Path


class SRTBuilder:

    @staticmethod
    def _format_timestamp(seconds):
        milliseconds = int(
            round(seconds * 1000)
        )

        hours = milliseconds // 3_600_000
        milliseconds %= 3_600_000

        minutes = milliseconds // 60_000
        milliseconds %= 60_000

        seconds = milliseconds // 1_000
        milliseconds %= 1_000

        return (
            f"{hours:02}:"
            f"{minutes:02}:"
            f"{seconds:02},"
            f"{milliseconds:03}"
        )

    def build(
        self,
        transcript_path,
        output_path,
    ):
        transcript_path = Path(
            transcript_path
        )

        output_path = Path(
            output_path
        )

        with open(
            transcript_path,
            "r",
            encoding="utf-8",
        ) as f:
            transcript = json.load(f)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        subtitle_index = 1

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            for segment in transcript["segments"]:

                text = segment["text"].strip()

                if not text:
                    continue

                words = segment.get("words", [])

                if not words:
                    continue

                # -------------------------------------------------
                # Timestamp được lấy trực tiếp từ word timestamp
                # -------------------------------------------------

                start = words[0]["start"]
                end = words[-1]["end"]

                # -------------------------------------------------
                # Không chỉnh sửa text
                # Không chia / gom segment
                # -------------------------------------------------

                f.write(
                    f"{subtitle_index}\n"
                )

                f.write(
                    f"{self._format_timestamp(start)}"
                    f" --> "
                    f"{self._format_timestamp(end)}\n"
                )

                f.write(
                    f"{text}\n\n"
                )

                subtitle_index += 1

        return output_path