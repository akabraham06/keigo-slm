"""Japanese register markers.

Marker sets derived from Feely et al. 2019, "Controlling Japanese Honorifics in
English-to-Japanese NMT" (Table 3), https://aclanthology.org/D19-5203.pdf

The register signal in Japanese lives in the (usually sentence-final) verb inflection.
We classify into three practical bands by looking for these markers, checking the most
specific band (formal keigo) first.

NOTE: this is a heuristic, not a parser. It's intentionally simple and transparent so it
can be audited and trusted as a label source. Both rule-based and Transformer classifiers
reach ~0.97 F1 on this 3-class task (WMT23), so a marker list is enough.
"""

# sonkeigo (respectful) + kenjougo (humble) forms — the "formal-deferential" band.
# Substrings are matched anywhere in the sentence. Checked BEFORE polite markers so that
# e.g. ございます (which contains ます) is classified formal, not polite.
FORMAL_MARKERS = [
    "ございます", "でございます",
    "いらっしゃ",              # いらっしゃいます / いらっしゃる
    "おります", "おられ",
    "なさい", "なさっ", "なさる",
    "致し", "いたします",       # 致します / いたします (humble "do")
    "伺",                       # 伺います / 伺う (humble "ask/visit")
    "参り", "参ります",         # humble "go/come"
    "召し上が",                 # respectful "eat"
    "頂き", "頂く", "いただき",  # humble "receive"
    "差し上げ", "さしあげ",      # humble "give"
    "おっしゃ",                 # respectful "say"
    "申し上げ", "申します",      # humble "say"
    "拝見",                     # humble "see/look"
    "ご覧",                     # respectful "see/look"
    "存じ",                     # humble "know/think"
    "承り", "承知",             # humble "hear / understood"
]

# teineigo (です / ます family) — the "neutral-polite" band.
POLITE_MARKERS = [
    "です", "ですか", "でした", "でしょう",
    "ます", "ますか", "ました", "ません", "ませんでした", "ましょう",
    "ください", "まして", "ませ",
]

# No informal marker list is needed: a sentence with none of the above is plain form
# (jōtai), i.e. casual.
