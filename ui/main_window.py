import flet as ft

from ui.text_tab import TextTab
from ui.srt_tab import SrtTab


class MainWindow:

    def __init__(self, page: ft.Page):
        page.title = "Piper TTS"
        page.window.width = 900
        page.window.height = 700
        page.padding = 20

        text_picker = ft.FilePicker()
        srt_picker = ft.FilePicker()

        page.overlay.extend([
            text_picker,
            srt_picker,
        ])

        page.add(
            ft.Tabs(
                length=2,
                expand=True,
                content=ft.Column(
                    expand=True,
                    controls=[
                        ft.TabBar(
                            tabs=[
                                ft.Tab(label="Text"),
                                ft.Tab(label="SRT"),
                            ],
                        ),
                        ft.TabBarView(
                            expand=True,
                            controls=[
                                TextTab(
                                    page,
                                    text_picker,
                                ),
                                SrtTab(
                                    page,
                                    srt_picker,
                                ),
                            ],
                        ),
                    ],
                ),
            )
        )

        page.update()