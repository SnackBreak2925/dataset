import re
import html

def postprocess_answer(answer):
    if "Оператор:" in answer:
        return answer.split("Оператор: ")[-1].strip()
    return answer.strip()

class TextCleaner:
    EMAIL_REGEX = re.compile(
        r'(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))',
        re.IGNORECASE,
    )
    URL_REGEX = re.compile(
        r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)",
        re.IGNORECASE,
    )
    LOGIN_REGEX = re.compile(
        r'(логин|login)[^\wа-яё:]*:?\s*([\w.@!#$%^&*()_+=\-\[\]{}:;"\'<>.,?/~]+)',
        re.IGNORECASE,
    )
    PASSWORD_REGEX = re.compile(
        r'(пароль|password)[^\wа-яё:]*:?\s*([\w!@#$%^&*()_+=\-\[\]{}:;"\'<>.,?/~]+)',
        re.IGNORECASE,
    )
    IGNORE_GROUP_PATTERN = re.compile(
        # теперь это работает как inline ignore даже без скобок!
        r"(пароль|password)\s*(и|или|or|and|,|/)?\s*(логин|login)|"
        r"(логин|login)\s*(и|или|or|and|,|/)?\s*(пароль|password)",
        re.IGNORECASE,
    )

    _SUBSTITUTIONS = [
        (re.compile(r"<[^>]+>", re.IGNORECASE), " "),
        (re.compile(r"<br\s*/?>", re.IGNORECASE), " "),
        (EMAIL_REGEX, "[EMAIL]"),
        (re.compile(r"\[SIGNATURE\]", re.IGNORECASE), ""),
        (URL_REGEX, "[URL]"),
        (re.compile(r"[\r\n]+", re.IGNORECASE), " "),
        (re.compile(r"\s{2,}", re.IGNORECASE), " "),
        (re.compile(r"\s+\.", re.IGNORECASE), "."),
        (re.compile(r"(?:\[URL\]\s*){2,}", re.IGNORECASE), "[URL] "),
    ]

    _SIMPLE_MASK_PATTERNS = [
        (re.compile(r"(пароль|password)\s+([\S]{3,})", re.IGNORECASE), "[PASSWORD]"),
        (re.compile(r"(логин|login)\s+([\S]{3,})", re.IGNORECASE), "[LOGIN]"),
    ]

    def clean(self, text_data: str) -> str:
        if not isinstance(text_data, str) or not text_data.strip():
            return ""
        text = html.unescape(text_data)
        text = self._substitute(text)
        ignore_spans = self._find_ignore_spans(text)
        text = self._mask_pattern(text, self.LOGIN_REGEX, ignore_spans, "[LOGIN]")
        text = self._mask_pattern(text, self.PASSWORD_REGEX, ignore_spans, "[PASSWORD]")
        text = self._mask_simple_patterns(text, ignore_spans)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def _substitute(self, text: str) -> str:
        for pattern, repl in self._SUBSTITUTIONS:
            text = pattern.sub(repl, text)
        return text

    def _find_ignore_spans(self, text: str):
        return [m.span() for m in self.IGNORE_GROUP_PATTERN.finditer(text)]

    def _in_ignore_spans(self, pos: int, ignore_spans) -> bool:
        return any(s <= pos < e for s, e in ignore_spans)

    def _mask_pattern(self, text: str, regex, ignore_spans, mask_label: str) -> str:
        def repl(match):
            start = match.start(1)
            if self._in_ignore_spans(start, ignore_spans):
                return match.group(0)
            value = match.group(2).strip()
            s = match.string
            pos = match.start(1)
            left_context = s[pos - 1] if pos > 0 else ""
            right_context = s[match.end(2)] if match.end(2) < len(s) else ""
            right_tail = s[match.end(2) :].lstrip()
            if right_tail.startswith("от "):
                return match.group(0)
            if not value or value in ("и", "или", "/", ",", "и/или"):
                return match.group(0)
            if left_context in "\"'«" or right_context in "\"'»":
                return match.group(0)
            if len(value.split()) > 1:
                return match.group(0)
            if mask_label == "[LOGIN]" and value.upper() in ("[LOGIN]", "[EMAIL]"):
                return match.group(0)
            if mask_label == "[PASSWORD]" and value.upper() == "[PASSWORD]":
                return match.group(0)
            if match.end(2) < len(s) and s[match.end(2)] == "]":
                return match.group(0)
            if value in {"С", "С уважением"} or (value.istitle() and len(value) <= 3):
                return match.group(0)
            return f"{match.group(1)}: {mask_label}"

        return regex.sub(repl, text)

    def _mask_simple_patterns(self, text: str, ignore_spans) -> str:
        def repl(mask_label):
            def fn(match):
                start = match.start(1)
                if self._in_ignore_spans(start, ignore_spans):
                    return match.group(0)
                s = match.string
                right_tail = s[match.end(2) :].lstrip()
                if right_tail.startswith("от "):
                    return match.group(0)
                return f"{match.group(1)}: {mask_label}"

            return fn

        for pattern, mask_label in self._SIMPLE_MASK_PATTERNS:
            text = pattern.sub(repl(mask_label), text)
        return text
