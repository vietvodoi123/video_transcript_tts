import json
from pathlib import Path


class TranscriptMerger:

    def merge(
        self,
        manifest_path,
        transcript_dir,
        output_path,
    ):
        manifest_path = Path(manifest_path)
        transcript_dir = Path(transcript_dir)
        output_path = Path(output_path)

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        all_segments = []

        for chunk in manifest["chunks"]:

            chunk_id = chunk["id"]

            transcript_path = (
                transcript_dir
                / f"{chunk_id}.json"
            )

            if not transcript_path.exists():
                raise FileNotFoundError(
                    f"Missing transcript: "
                    f"{transcript_path}"
                )

            with open(
                transcript_path,
                "r",
                encoding="utf-8",
            ) as f:
                transcript = json.load(f)

            chunk_start = chunk["start"]

            for segment in transcript["segments"]:

                words = []

                for word in segment["words"]:

                    words.append(
                        {
                            "word": word["word"],
                            "start": (
                                chunk_start
                                + word["start"]
                            ),
                            "end": (
                                chunk_start
                                + word["end"]
                            ),
                        }
                    )

                if not words:
                    continue

                all_segments.append(
                    {
                        "start": words[0]["start"],
                        "end": words[-1]["end"],
                        "text": segment["text"],
                        "words": words,
                        "chunk_id": chunk_id,
                    }
                )

        all_segments.sort(
            key=lambda x: x["start"]
        )

        for index, segment in enumerate(
            all_segments,
            start=1,
        ):
            segment["index"] = index

        data = {
            "version": 2,
            "language": "zh",
            "segments": all_segments,
        }

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