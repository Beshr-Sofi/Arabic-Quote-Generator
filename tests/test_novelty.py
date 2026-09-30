# tests/test_novelty.py
from src.novelty.checker import NoveltyChecker

CORPUS = [
    "الصبر مفتاح الفرج",
    "من جد وجد ومن زرع حصد",
    "العلم نور والجهل ظلام",
]

def test_verbatim_copy_is_flagged():
    chk = NoveltyChecker().fit(CORPUS)
    assert chk.check("الصبر مفتاح الفرج").memorized

def test_unrelated_sentence_passes():
    chk = NoveltyChecker().fit(CORPUS)
    assert not chk.check("القهوة الصباحية تمنح اليوم بداية هادئة").memorized

def test_empty_is_rejected():
    assert NoveltyChecker().fit(CORPUS).check("   ").memorized

test_verbatim_copy_is_flagged()
test_unrelated_sentence_passes()
test_empty_is_rejected()
print('Novelty tests passed')