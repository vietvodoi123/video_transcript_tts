from pathlib import Path

import flet as ft

from pipelines.text_pipeline import TextPipeline

MODEL_DIR = Path("tts-model")
OUTPUT_DIR = Path("output")


def get_voice_list():
    return sorted(
        model.stem
        for model in MODEL_DIR.glob("*.onnx")
    )


class TextTab(ft.Column):

    def __init__(
        self,
        page: ft.Page,
        file_picker: ft.FilePicker,
    ):
        super().__init__(expand=True)

        self._page = page
        self.file_picker = file_picker

        self.selected_file = None

        voices = get_voice_list()

        self.voice = ft.Dropdown(
            label="Voice",
            value=voices[0] if voices else None,
            options=[
                ft.dropdown.Option(v)
                for v in voices
            ],
        )

        self.output_name = ft.TextField(
            label="Output file name",
        )

        self.file_name = ft.TextField(
            label="TXT File",
            read_only=True,
        )

        self.text = ft.TextField(
            label="Text",
            multiline=True,
            min_lines=15,
            expand=True,
        )

        self.result = ft.Text()

        self.file_picker.on_result = self._on_pick_file

        self.controls = [
            self.voice,
            self.output_name,
            ft.Row(
                controls=[
                    self.file_name,
                    ft.ElevatedButton(
                        "Browse",
                        on_click=self.pick_file,
                    ),
                ]
            ),
            self.text,
            ft.ElevatedButton(
                "Generate",
                on_click=self.generate,
            ),
            self.result,
        ]

    def pick_file(
        self,
        e,
    ):
        self.file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["txt"],
        )

    def _on_pick_file(
        self,
        e,
    ):
        if not e.files:
            return

        self.selected_file = e.files[0].path

        self.file_name.value = self.selected_file

        self.text.value = Path(
            self.selected_file
        ).read_text(
            encoding="utf-8",
        )

        self.update()

    def generate(
        self,
        e,
    ):
        try:

            output = (
                OUTPUT_DIR
                / f"{self.output_name.value}.wav"
            )

            pipeline = TextPipeline(
                model_dir=MODEL_DIR,
                voice=self.voice.value,
            )

            pipeline.run(
                text=self.text.value,
                output_file=output,
            )

            self.result.value = (
                f"Done: {output}"
            )

        except Exception as ex:

            self.result.value = str(ex)

        self.update()