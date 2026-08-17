from pathlib import Path

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

import gradio as gr

from pipelines.text_pipeline import TextPipeline
from pipelines.srt_pipeline import SrtPipeline

MODEL_DIR = Path("tts-model")
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


def get_voice_list():
    return sorted(
        file.stem
        for file in MODEL_DIR.glob("*.onnx")
    )


VOICES = get_voice_list()


def text_to_wav(
    text: str,
    txt_file: str | None,
    voice: str,
    output_name: str,
):
    try:

        if txt_file:
            text = Path(txt_file).read_text(
                encoding="utf-8"
            )

        if not text.strip():
            raise gr.Error("Vui lòng nhập text.")

        output_name = output_name.strip()

        if not output_name:
            raise gr.Error("Vui lòng nhập Output Name.")

        output_file = OUTPUT_DIR / f"{output_name}.wav"

        pipeline = TextPipeline(
            model_dir=MODEL_DIR,
            voice=voice,
        )

        pipeline.run(
            text=text,
            output_file=output_file,
        )

        return (
            str(output_file),
            f"✅ Done\n{output_file}"
        )

    except Exception as ex:

        return None, str(ex)


def srt_to_wav(
    srt_file: str,
    voice: str,
    output_name: str,
):
    try:

        if srt_file is None:
            raise gr.Error("Vui lòng chọn file SRT.")

        output_name = output_name.strip()

        if not output_name:
            raise gr.Error("Vui lòng nhập Output Name.")

        output_file = OUTPUT_DIR / f"{output_name}.wav"

        pipeline = SrtPipeline(
            model_dir=MODEL_DIR,
            voice=voice,
            temp_dir=TEMP_DIR,
        )

        pipeline.run(
            input_file=srt_file,
            output_file=output_file,
        )

        return (
            str(output_file),
            f"✅ Done\n{output_file}"
        )

    except Exception as ex:

        return None, str(ex)


with gr.Blocks(
    title="Piper TTS",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("# Piper TTS")

    with gr.Tab("Text"):

        voice1 = gr.Dropdown(
            label="Voice",
            choices=VOICES,
            value=VOICES[0] if VOICES else None,
        )

        txt_file = gr.File(
            label="TXT",
            file_types=[".txt"],
            type="filepath",
        )

        output_name1 = gr.Textbox(
            label="Output Name",
            placeholder="chapter001",
        )

        text = gr.Textbox(
            label="Text",
            lines=15,
        )

        btn = gr.Button(
            "Generate",
            variant="primary",
        )

        audio = gr.Audio(
            label="Output",
            type="filepath",
        )

        status = gr.Textbox(
            label="Status",
            interactive=False,
        )

        btn.click(
            fn=text_to_wav,
            inputs=[
                text,
                txt_file,
                voice1,
                output_name1,
            ],
            outputs=[
                audio,
                status,
            ],
        )

    with gr.Tab("SRT"):

        voice2 = gr.Dropdown(
            label="Voice",
            choices=VOICES,
            value=VOICES[0] if VOICES else None,
        )

        srt_file = gr.File(
            label="SRT",
            file_types=[".srt"],
            type="filepath",
        )

        output_name2 = gr.Textbox(
            label="Output Name",
            placeholder="chapter001",
        )

        btn2 = gr.Button(
            "Generate",
            variant="primary",
        )

        audio2 = gr.Audio(
            label="Output",
            type="filepath",
        )

        status2 = gr.Textbox(
            label="Status",
            interactive=False,
        )

        btn2.click(
            fn=srt_to_wav,
            inputs=[
                srt_file,
                voice2,
                output_name2,
            ],
            outputs=[
                audio2,
                status2,
            ],
        )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
    )