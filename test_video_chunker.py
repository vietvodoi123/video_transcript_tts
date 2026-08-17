from services.video_chunker import VideoChunker


VIDEO_PATH = r"C:\Users\HLC\PycharmProjects\tts\input_long.mp4"
MENIFET_PATH = r"/input_long.json"

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60

    return f"{hours:02}:{minutes:02}:{seconds:05.2f}"


def main():
    chunker = VideoChunker(
        target_duration=5 * 60,
        min_duration=4 * 60,
        max_duration=6 * 60,
        silence_duration=0.8,
    )

    chunks = chunker.create_chunks(
        VIDEO_PATH,MENIFET_PATH
    )

    print()
    print("=" * 70)
    print("VIDEO CHUNKS")
    print("=" * 70)

    for chunk in chunks:

        print(
            f"Chunk {chunk.index:03} | "
            f"{format_time(chunk.start)} → "
            f"{format_time(chunk.end)} | "
            f"{chunk.duration:.2f}s"
        )

    print()
    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()