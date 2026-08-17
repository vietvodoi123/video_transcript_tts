from pathlib import Path

from workspace.pipeline_workspace import PipelineWorkspace

from services.video_chunker import VideoChunker
from services.audio_extractor import AudioExtractor
from services.whisper_service import WhisperService
from services.transcript_merger import TranscriptMerger
from services.srt_builder import SRTBuilder


INPUT_VIDEO = Path(
    r"C:\Users\HLC\PycharmProjects\tts\input_long.mp4"
)

OUTPUT_ROOT = Path(
    r"C:\Users\HLC\PycharmProjects\tts\output"
)


def main():

    # =====================================================
    # 1. Workspace
    # =====================================================

    workspace = PipelineWorkspace(
        output_root=OUTPUT_ROOT,
    )

    workspace.create()

    print()
    print(f"Job ID: {workspace.job_id}")
    print(f"Workspace: {workspace.root}")

    # =====================================================
    # 2. Video Chunker
    # =====================================================

    chunker = VideoChunker(
        target_duration=5 * 60,
        min_duration=4 * 60,
        max_duration=6 * 60,
        silence_duration=0.8,
    )

    chunks = chunker.create_chunks(
        INPUT_VIDEO,
        workspace.manifest_path,
    )

    print()
    print(
        f"Created {len(chunks)} chunks."
    )

    # =====================================================
    # 3. Audio Extractor
    # =====================================================

    extractor = AudioExtractor()

    extractor.extract_from_manifest(
        workspace.manifest_path,
        workspace.chunk_audio_dir,
    )

    print()
    print("Audio extraction completed.")

    # =====================================================
    # 4. Whisper
    # =====================================================

    whisper = WhisperService(
        model_size="large-v3",
    )

    for chunk in chunks:

        audio_path = (
            workspace.chunk_audio_path(
                chunk.id
            )
        )

        transcript_path = (
            workspace.chunk_transcript_path(
                chunk.id
            )
        )

        print()
        print(
            "=" * 70
        )
        print(
            f"Whisper: {chunk.id}"
        )
        print(
            f"{chunk.start:.2f} "
            f"-> "
            f"{chunk.end:.2f}"
        )
        print(
            "=" * 70
        )

        whisper.transcribe_to_json(
            audio_path=audio_path,
            output_path=transcript_path,
            chunk_id=chunk.id,
        )

    print()
    print("Whisper transcription completed.")

    # =====================================================
    # 5. Transcript Merger
    # =====================================================

    merger = TranscriptMerger()

    merger.merge(
        manifest_path=workspace.manifest_path,
        transcript_dir=(
            workspace.transcript_chunks_dir
        ),
        output_path=(
            workspace.merged_transcript_path
        ),
    )

    print()
    print(
        "Transcript merge completed."
    )

    # =====================================================
    # 6. SRT
    # =====================================================

    srt_builder = SRTBuilder()

    srt_builder.build(
        transcript_path=(
            workspace.merged_transcript_path
        ),
        output_path=(
            workspace.subtitle_path
        ),
    )

    print()
    print(
        "SRT generation completed."
    )

    # =====================================================
    # 7. Result
    # =====================================================

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"Workspace : {workspace.root}"
    )

    print(
        f"Manifest  : {workspace.manifest_path}"
    )

    print(
        f"Transcript : "
        f"{workspace.merged_transcript_path}"
    )

    print(
        f"SRT       : {workspace.subtitle_path}"
    )


if __name__ == "__main__":
    main()