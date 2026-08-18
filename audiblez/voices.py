# -*- coding: utf-8 -*-
import platform

flags = {'a': '🇺🇸', 'b': '🇬🇧', 'e': '🇪🇸', 'f': '🇫🇷', 'h': '🇮🇳', 'i': '🇮🇹', 'j': '🇯🇵', 'p': '🇧🇷', 'z': '🇨🇳',
         'k': '🇰🇷', 'd': '🇩🇪', 'r': '🇷🇺'}

flags_win = {'a': 'american', 'b': 'british', 'e': 'spanish', 'f': 'french', 'h': 'hindi', 'i': 'italian',
             'j': 'japanese', 'p': 'portuguese', 'z': 'chinese', 'k': 'korean', 'd': 'german', 'r': 'russian'}

voices = {
    'a': ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova',
          'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam',
          'am_michael', 'am_onyx', 'am_puck', 'am_santa'],
    'b': ['bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'],
    'e': ['ef_dora', 'em_alex', 'em_santa'],
    'f': ['ff_siwis'],
    'h': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
    'i': ['if_sara', 'im_nicola'],
    'j': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia',
          'zm_yunyang']
}

# Qwen3-TTS CustomVoice ships nine fixed speakers -- no blending, no .pt packs, no cloning.
# Names are the model's own lowercase ids (get_supported_speakers()); it matches them
# case-insensitively. Grouped by the speaker's native language, which is what the model
# card recommends using them for, though each can speak any supported language.
#
# Note how thin the English side is: two speakers, against Kokoro's 28.
qwen_voices = {
    'a': ['ryan', 'aiden'],
    'z': ['serena', 'vivian', 'uncle_fu', 'eric', 'dylan'],
    'j': ['ono_anna'],
    'k': ['sohee'],
}

VOICES_BY_MODEL = {
    'kokoro': voices,
    'qwen3-tts': qwen_voices,
}


def voices_for(model='kokoro'):
    """The voice table for a model, keyed by language code.

    One source of truth for the CLI epilog and the GUI dropdown, so neither grows its own
    conditional as models are added.
    """
    try:
        return VOICES_BY_MODEL[model]
    except KeyError:
        raise ValueError(f"Unknown model {model!r}. Choose one of: {', '.join(VOICES_BY_MODEL)}")


def flat_voices(model='kokoro'):
    """Every voice for a model, in dropdown order."""
    return [v for lang in voices_for(model) for v in voices_for(model)[lang]]


def voice_language(voice, model='kokoro'):
    """The language code a voice belongs to, or None if it is not a known voice.

    Kokoro voices encode it in the name; Qwen speakers do not, so they need the lookup.
    """
    for lang, names in voices_for(model).items():
        if voice.lower() in [n.lower() for n in names]:
            return lang
    return None


def describe_voices(model='kokoro'):
    """Voice list formatted for CLI help, one line per language."""
    labels = flags_win if platform.system() == 'Windows' else flags
    table = voices_for(model)
    return '\n'.join(f'  {labels.get(lang, lang)}:\t{", ".join(table[lang])}' for lang in table)


available_voices_str = describe_voices('kokoro')
