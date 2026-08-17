from pathlib import Path
import shutil
from datetime import datetime

def ensure_dir(
    path: str | Path,
) -> Path:

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def clear_directory(
    path: str | Path,
):

    path = Path(path)

    if not path.exists():
        return

    for item in path.iterdir():

        if item.is_file():
            item.unlink()

        elif item.is_dir():
            shutil.rmtree(item)


def stem(
    file: str | Path,
) -> str:

    return Path(file).stem


def extension(
    file: str | Path,
) -> str:

    return Path(file).suffix.lower()


def output_wav_name(
    input_file: str | Path | None,
) -> str:

    if input_file:
        return f"{Path(input_file).stem}.wav"

    return datetime.now().strftime(
        "text_%Y%m%d_%H%M%S.wav"
    )