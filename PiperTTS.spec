# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import piper
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

PIPER_DIR = Path(piper.__file__).parent
ESPEAK_DIR = PIPER_DIR / "espeak-ng-data"

for file in ESPEAK_DIR.rglob("*"):
    if file.is_file():
        rel = file.relative_to(ESPEAK_DIR)
        datas.append(
            (
                str(file),
                str(Path("piper") / "espeak-ng-data" / rel.parent),
            )
        )

packages = [
    "gradio",
    "gradio_client",
    "safehttpx",
    "groovy",
    "fastapi",
    "starlette",
    "uvicorn",
    "httpx",
    "anyio",
    "pydantic",
    "orjson",
    "requests",
]

for pkg in packages:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h



a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PiperTTS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PiperTTS",
)