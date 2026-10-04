CLUSTER_TO_THEME = {
    0: "pop-culture",      # fictional characters — route together
    1: "other",             # mixed: moving-on/last-words/catching-fire — too noisy to trust
    2: "nature",
    3: "truth-and-belief",
    4: "self",
    5: "pop-culture",       # Hunger Games characters/food refs
    6: "freedom-and-society",
    7: "growth-and-change",
    8: "inspiration",
    9: "faith-and-philosophy",
    10: "knowledge-and-leadership",
    11: "pop-culture",
    12: "time",
    13: "world-and-home",
    14: "pop-culture",      # jace/magic/fantasy — Mortal Instruments
    15: "other",             # pure metadata — see override note below
    16: "pop-culture",
    17: "journey-and-persistence",
    18: "courage",
    19: "writing-and-words",
    20: "love-and-relationships",
    21: "life",
    22: "art-and-creativity",
    23: "dreams-and-imagination",
    24: "books-and-reading",
    25: "humor",
    26: "character-and-trust",
    27: "pain-and-sadness",   # suicide exclusion applied via override, see below
    28: "people-and-family",
    29: "loss-and-peace",
}

MANUAL_OVERRIDES = {
    # sensitive exclusion — same policy as Arabic: keep grief/sadness as a theme,
    # drop rows specifically tagged with suicide content
    "suicide": None,

    # pure dataset-curation metadata, not a theme at all — don't let these
    # merge into "other" and dilute it; exclude outright
    "misattributed": None,
    "misattributed-mark-twain": None,
    "misattributed-eleanor-roosevelt": None,
    "misattributed-to-einstein": None,
    "misattributed-to-mother-teresa": None,
    "misattributed-george-carlin": None,
    "widely-misattributed": None,
    "attributed": None,
    "paraphrased": None,
    "quote-investigator": None,
    "gilbert-perreira": None,

    # cluster 1 noise — a couple of these are real themes worth rescuing
    "letting-go": "self",
    "moving-on": "self",
}

