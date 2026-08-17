from pathlib import Path

from models.subtitle import Subtitle


class SrtParser:

    @staticmethod
    def parse(
        file: str | Path,
    ) -> list[Subtitle]:

        file = Path(file)

        content = file.read_text(
            encoding="utf-8",
        )

        content = content.replace(
            "\r\n",
            "\n",
        )

        subtitles: list[Subtitle] = []

        blocks = content.strip().split("\n\n")

        for block in blocks:

            lines = [
                line.strip()
                for line in block.split("\n")
                if line.strip()
            ]

            if len(lines) < 3:
                continue

            try:

                index = int(lines[0])

                start_str, end_str = lines[1].split("-->")

                start = SrtParser._parse_time(
                    start_str.strip()
                )

                end = SrtParser._parse_time(
                    end_str.strip()
                )

                text = "\n".join(
                    lines[2:]
                )

                subtitles.append(
                    Subtitle(
                        index=index,
                        start=start,
                        end=end,
                        text=text,
                    )
                )

            except Exception:
                continue

        return subtitles

    @staticmethod
    def _parse_time(
        value: str,
    ) -> float:

        hour, minute, second = value.split(":")

        second, ms = second.split(",")

        return (
            int(hour) * 3600
            + int(minute) * 60
            + int(second)
            + int(ms) / 1000
        )