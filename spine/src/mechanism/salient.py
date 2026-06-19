"""Token classification for the uneven privacy budget.

Word tokens that are not protected (hate-carrying) fall into:

* ``function_word`` -- closed-class / stop words. Kept by default (skip epsilon)
  to preserve grammaticality; they carry little stylistic identity individually.
* ``number``        -- pure digit runs; kept (not rewritten).
* ``content``       -- the style-bearing remainder; this is what the DP rewrite
  perturbs, governed by ``epsilon.content`` (the noise dial).

Protected-token detection lives in lexicon.py (obfuscation-aware) and is applied
by the orchestrator (spine.py) before this classifier runs. The optional
classifier-saliency hook (cfg.saliency.enabled) is declared here but only does
anything when the HF extra is installed; it is off by default.
"""

from __future__ import annotations

from .tokenize import HASHTAG, MENTION, URL, WORD, Segment

# Closed-class English function / stop words. Individually low-identity.
FUNCTION_WORDS = frozenset(
    """
    a an the this that these those some any no each every both either neither
    i me my mine myself you your yours yourself he him his himself she her hers
    herself it its itself we us our ours ourselves they them their theirs
    themselves who whom whose which what
    is am are was were be been being do does did doing have has had having
    will would shall should can could may might must ought
    and or but nor for yet so because although though while if unless until
    when where why how than as that whether
    of in on at by to from with about against between into through during before
    after above below up down out off over under again further then once here
    there all very too just only also even still not no nor
    """.split()
)


def classify_word(surface: str) -> str:
    """Classify a normalised WORD surface into a budget class."""
    s = surface.strip().lower()
    if not s:
        return "content"
    if s.isdigit():
        return "number"
    # Multi-word misspelling expansions (e.g. "going to") -> classify on first.
    head = s.split()[0]
    if head in FUNCTION_WORDS:
        return "function_word"
    return "content"


def classify_segment(seg: Segment) -> str:
    """Token class for a non-protected token segment."""
    if seg.kind in (URL, MENTION, HASHTAG):
        return seg.kind          # kept verbatim
    if seg.kind == WORD:
        return classify_word(seg.text)
    return "content"


def is_rewritable(token_class: str) -> bool:
    """Only content tokens are candidates for the DP rewrite."""
    return token_class == "content"
