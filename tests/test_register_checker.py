"""Tests for the source-side register checker.

The checker is the project's ground-truth label source, so it must be trustworthy.
These cases are the canonical, unambiguous ones — the aru/arimasu/gozaimasu triple
(Feely19's own worked example) plus a few clear keigo forms.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from register_checker import classify_register  # noqa: E402


# The canonical minimal-contrastive triple: same meaning, three registers, verb only.
def test_canonical_triple():
    assert classify_register("駅の近くにたくさんのお店がある。") == "informal"
    assert classify_register("駅の近くにたくさんのお店があります。") == "polite"
    assert classify_register("駅の近くにたくさんのお店がございます。") == "formal"


def test_plain_copula_is_informal():
    assert classify_register("明日は雨だ。") == "informal"
    assert classify_register("これは本だ。") == "informal"


def test_desu_masu_is_polite():
    assert classify_register("明日は雨です。") == "polite"
    assert classify_register("駅に行きます。") == "polite"
    assert classify_register("少々お待ちください。") == "polite"


def test_keigo_is_formal():
    assert classify_register("資料を拝見しました。") == "formal"          # kenjougo
    assert classify_register("こちらをご覧になりますか。") == "formal"     # sonkeigo
    assert classify_register("明日そちらに伺います。") == "formal"         # kenjougo
    assert classify_register("担当者が参ります。") == "formal"            # kenjougo


def test_formal_beats_polite_ordering():
    # ございます contains the polite substring ます but must classify as formal.
    assert classify_register("ありがとうございます。") == "formal"
