from services.whisper_service import WhisperService


VIDEO_PATH = r"C:\Users\HLC\PycharmProjects\tts\input.mp4"


def main():

    service = WhisperService(
        model_size="large-v3"
    )

    segments, info = service.transcribe(
        VIDEO_PATH
    )

    print()
    print("=" * 80)
    print("TRANSCRIPTION RESULT")
    print("=" * 80)

    print(f"Language: {info.language}")
    print(f"Probability: {info.language_probability:.2f}")
    print(f"Device: {service.device}")
    print(f"Segments: {len(segments)}")

    print()

    for segment in segments:

        print(
            f"[{segment.index:04}] "
            f"{segment.start:8.2f} --> "
            f"{segment.end:8.2f} | "
            f"{segment.text}"
        )


if __name__ == "__main__":
    main()