import re
from pathlib import Path

import httpx


class SrtTranslationService:

    def __init__(
        self,
        base_url="http://127.0.0.1:8090",
        timeout=120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def translate_text(
        self,
        client: httpx.AsyncClient,
        text: str,
    ) -> str:

        text = text.strip()

        if not text:
            return ""

        response = await client.post(
            f"{self.base_url}/translate",
            json={
                "text": text,
            },
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("ok"):
            raise RuntimeError(
                f"Translation failed: {result}"
            )

        translated_text = result.get(
            "text",
            "",
        )

        if translated_text is None:
            return ""

        return translated_text.strip()

    async def translate_file(
        self,
        input_path,
        output_path,
    ):
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"SRT file not found: {input_path}"
            )

        content = input_path.read_text(
            encoding="utf-8-sig"
        )

        blocks = re.split(
            r"\r?\n\s*\r?\n",
            content.strip(),
        )

        translated_blocks = []

        async with httpx.AsyncClient(
            timeout=self.timeout
        ) as client:

            total = len(blocks)

            for position, block in enumerate(
                blocks,
                start=1,
            ):

                lines = block.splitlines()

                if len(lines) < 3:
                    raise ValueError(
                        f"Invalid SRT block "
                        f"#{position}:\n{block}"
                    )

                index = lines[0].strip()
                timestamp = lines[1].strip()

                text = "\n".join(
                    lines[2:]
                ).strip()

                print(
                    f"[{position}/{total}] "
                    f"Translating..."
                )

                translated_text = (
                    await self.translate_text(
                        client,
                        text,
                    )
                )

                translated_block = (
                    f"{index}\n"
                    f"{timestamp}\n"
                    f"{translated_text}"
                )

                translated_blocks.append(
                    translated_block
                )

                print(
                    f"[{position}/{total}] "
                    f"OK: {translated_text}"
                )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            "\n\n".join(
                translated_blocks
            ) + "\n",
            encoding="utf-8",
        )

        return output_path