import json
from pathlib import Path

import ffmpeg


class AudioExtractor:

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract_chunk(
        self,
        source_video,
        chunk,
        output_path,
    ):
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            ffmpeg
            .input(
                str(source_video),
                ss=chunk["start"],
                t=chunk["duration"],
            )
            .output(
                str(output_path),
                format="wav",
                acodec="pcm_s16le",
                ac=1,
                ar=self.sample_rate,
            )
            .overwrite_output()
            .run(
                quiet=True,
            )
        )

        return output_path

    def extract_from_manifest(
        self,
        manifest_path,
        output_dir,
    ):
        manifest_path = Path(manifest_path)
        output_dir = Path(output_dir)

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as f:
            manifest = json.load(f)

        source_video = Path(
            manifest["source"]
        )

        results = []

        for chunk in manifest["chunks"]:

            output_path = (
                output_dir /
                f'{chunk["id"]}.wav'
            )

            self.extract_chunk(
                source_video,
                chunk,
                output_path,
            )

            results.append(
                {
                    "chunk_id": chunk["id"],
                    "path": str(output_path),
                }
            )

        return results