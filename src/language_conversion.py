import langdetect
from langdetect import detect
import sys
from transformers import *
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
language_support = {
    "af",
    "ar",
    "bg",
    "bn",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "fi",
    "fr",
    "gu",
    "he",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "kn",
    "ko",
    "lt",
    "lv",
    "mk",
    "ml",
    "mr",
    "ne",
    "nl",
    "no",
    "pa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "so",
    "sq",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh-cn",
    "zh-tw",
}


def main(text):
    lang = detect(text)
    print(lang)
    if lang not in language_support:
        print("do nothing")
    elif lang == "en":
        print("send text to model")
    else:
        en_text = translate_text(text, lang)
        print(en_text)


def translate_text(text, lang):  # copied: reference
    src = lang
    dst = "en"

    task_name = f"translation_{src}_to_{dst}"
    model_name = f"Helsinki-NLP/opus-mt-{src}-{dst}"

    translator = pipeline(
        task_name, model=model_name, tokenizer=model_name, device=device
    )
    return translator(text)[0]["translation_text"]


if __name__ == "__main__":
    # text = sys.argv[1]
    text = "elles sont toutes des menteuses"
    main(text)
