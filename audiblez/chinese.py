# -*- coding: utf-8 -*-
"""Traditional Chinese support, by way of the simplified script.

There is no such thing as a "traditional Chinese voice" to add. A voice is a speaker's
timbre; the script a book is printed in is a property of the *text*, and it is handled
before the model ever sees it, in the grapheme-to-phoneme front end. Kokoro's Chinese front
end (misaki[zh]) is jieba for word segmentation plus pypinyin for readings, and both of
those dictionaries are keyed on simplified characters. Feed them traditional text and they
mostly limp along character by character, which costs exactly the two things the phrase
dictionaries were there to provide:

- **Polyphone disambiguation.** 乾乾淨淨 is read `qián qián jìng jìng` rather than
  `gān gān jìng jìng`, because 乾 collapses two different simplified characters (干 and 乾)
  and the phrase that would settle which one is meant is not in the dictionary. The worst
  offender is the aspect particle 著 (她笑著說), read `zhù` rather than the neutral-tone
  `zhe` -- and that one turns up in nearly every paragraph of a novel.
- **Word segmentation.** 臺灣的 segments as 臺 / 灣的 and 個人罷了 as 個 / 人罷 / 了, so the
  tone sandhi and the pauses land in the wrong places.

Converting to simplified before phonemization fixes both, and costs nothing in fidelity:
Mandarin pronunciation does not depend on which script the sentence was written in. The
audio you get is what a Mandarin reader would say when reading the traditional page aloud.

The conversion applies to the narration path only. Titles, chapter marks and everything
else written into the .m4b keep the book's original characters.
"""
import re
import warnings

# OpenCC's own config name. tw2s over the plainer t2s because of one very common word: the
# aspect particle 著 (她笑著說). t2s leaves it alone -- 著 is a simplified character in its own
# right, read zhù -- so pypinyin says "xiào zhù shuō" in every progressive sentence in the
# book. tw2s maps the particle to 着 and still leaves 著作 and 著名 as they are.
#
# Deliberately *not* tw2sp, which additionally rewrites regional vocabulary
# (軟體 -> 软件, 計程車 -> 出租车). That changes the author's words, not just their script.
OPENCC_CONFIG = 'tw2s'

# Language codes that mean Chinese: Kokoro's letter, and Qwen's full name.
CHINESE_LANG_CODES = frozenset({'z', 'chinese'})

# Codes that name no language at all. Qwen speaker names carry none, so a Qwen run without
# an explicit --lang arrives here as 'auto' and the text has to speak for itself.
UNKNOWN_LANG_CODES = frozenset({'auto', ''})

_HAN_RE = re.compile(r'[一-鿿㐀-䶿]')
# Kana and hangul mark a document as Japanese or Korean. Both scripts embed Han characters
# whose traditional/simplified mapping is a different question -- Japanese shinjitai are not
# Chinese simplified characters -- so t2s must not be let loose on them.
_KANA_RE = re.compile(r'[぀-ヿ]')
_HANGUL_RE = re.compile(r'[ᄀ-ᇿ가-힯]')

_converter = None
_converter_failed = False
_announced = False


def _get_converter():
    """The cached OpenCC converter, or None if the optional dependency is missing."""
    global _converter, _converter_failed
    if _converter is not None or _converter_failed:
        return _converter
    try:
        from opencc import OpenCC
    except ImportError:
        _converter_failed = True
        warnings.warn(
            'opencc is not installed, so traditional Chinese will be read with simplified-only '
            'dictionaries and some characters will get the wrong reading. '
            'Install it with: pip install opencc-python-reimplemented',
            stacklevel=3)
        return None
    _converter = OpenCC(OPENCC_CONFIG)
    return _converter


def to_simplified(text):
    """Convert traditional characters to simplified. A no-op if opencc is unavailable."""
    converter = _get_converter()
    if converter is None:
        return text
    return converter.convert(text)


def looks_chinese(text):
    """True for text that is Han script and not Japanese or Korean."""
    return bool(_HAN_RE.search(text)) and not _KANA_RE.search(text) and not _HANGUL_RE.search(text)


def wants_simplification(lang_code, text):
    """Whether `text` should be converted before being phonemized as `lang_code`.

    An explicit Chinese language code is enough on its own. When the caller named no
    language the text has to look Chinese, so that a Japanese or Korean book -- which also
    reaches synthesis with lang_code 'auto' -- is left alone.
    """
    code = str(lang_code or '').lower()
    if code in CHINESE_LANG_CODES:
        return True
    if code in UNKNOWN_LANG_CODES:
        return looks_chinese(text)
    return False


def normalize(text, lang_code=None, notify=None):
    """Return `text` ready for a Mandarin front end, converting the script if needed.

    `notify` is called at most once per process with a one-line explanation, the first time
    a conversion actually changes something -- a book is dozens of chapters and this is one
    fact about the whole book, not news each chapter.
    """
    global _announced
    if not wants_simplification(lang_code, text):
        return text
    converted = to_simplified(text)
    if converted != text and not _announced:
        _announced = True
        if notify:
            notify('Traditional Chinese detected: converting to simplified characters for '
                   'pronunciation. The audiobook is narrated in Mandarin; the text written into '
                   'the .m4b is untouched.')
    return converted


def reset_notice():
    """Forget that the conversion notice was shown. For tests, and for a fresh GUI run."""
    global _announced
    _announced = False
