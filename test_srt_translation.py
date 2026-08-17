import asyncio
from pathlib import Path

from services.srt_translation_service import (
    SrtTranslationService,
)


INPUT_SRT = Path(
    r"C:\Users\HLC\PycharmProjects\tts\output\ab6f3f3413a241afa010d5bea29efa5a\subtitle\subtitle.srt"
)

OUTPUT_SRT = Path(
    r"C:\Users\HLC\PycharmProjects\tts\output\ab6f3f3413a241afa010d5bea29efa5a\subtitle\subtitle_vi.srt"
)


async def main():

    service = SrtTranslationService(
        base_url="http://127.0.0.1:8090",
    )

    result = await service.translate_file(
        input_path=INPUT_SRT,
        output_path=OUTPUT_SRT,
    )

    print()
    print("=" * 70)
    print("TRANSLATION COMPLETED")
    print("=" * 70)
    print(f"Output: {result}")


if __name__ == "__main__":
    asyncio.run(main())