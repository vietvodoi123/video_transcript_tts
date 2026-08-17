from typing import Iterable


class TextSplitter:
    """
    Chia văn bản thành các câu để đưa vào TTS.
    """

    SENTENCE_ENDINGS = {
        ".",
        "!",
        "?",
        "…",
    }

    def split(
        self,
        text: str,
    ) -> list[str]:
        text = text.strip()

        if not text:
            return []

        results: list[str] = []
        buffer: list[str] = []

        for char in text:
            buffer.append(char)

            if char in self.SENTENCE_ENDINGS:
                sentence = "".join(buffer).strip()

                if sentence:
                    results.append(sentence)

                buffer.clear()

        if buffer:
            sentence = "".join(buffer).strip()

            if sentence:
                results.append(sentence)

        return results

    def split_many(
        self,
        texts: Iterable[str],
    ) -> list[str]:
        results: list[str] = []

        for text in texts:
            results.extend(
                self.split(text)
            )

        return results