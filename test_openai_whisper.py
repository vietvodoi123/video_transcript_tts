import os

from services.openai_whisper_service import (
    OpenAIWhisperService,
)


AUDIO_PATH = (
    r"C:\Users\HLC\PycharmProjects\tts\output\ab6f3f3413a241afa010d5bea29efa5a\chunks\audio\chunk_0001.wav"
)


def main():

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    service = OpenAIWhisperService()

    segments, response = (
        service.transcribe(
            AUDIO_PATH
        )
    )

    print()
    print("=" * 80)
    print("OPENAI TRANSCRIPTION")
    print("=" * 80)

    for segment in segments:

        print(
            f"{segment.start:8.2f}"
            f" --> "
            f"{segment.end:8.2f}"
            f" | "
            f"{segment.text}"
        )

        for word in segment.words:

            print(
                f"    "
                f"{word.start:8.2f}"
                f" --> "
                f"{word.end:8.2f}"
                f" | "
                f"{word.word}"
            )


if __name__ == "__main__":
    main()