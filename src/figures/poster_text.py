"""Text blocks placed on the poster by layout_poster.py.

METHODS is kept as a draft until it is approved; layout_poster.py reserves its
space but renders only the heading while METHODS_APPROVED is False.
"""

# Shown under the title, in the order given.
AUTHORS = ["Rafail Adam", "Martin Reczko", "Marios Miliotis", "Artemis Hatzigeorgiou"]

# Which AFFILIATIONS (1-based, in the order listed below) each author belongs to.
AUTHOR_AFFILIATIONS = {
    "Rafail Adam": [1, 2],
    "Martin Reczko": [3],
    "Marios Miliotis": [1, 2],
    "Artemis Hatzigeorgiou": [1, 2],
}

# Shown under the authors. Official names as the institutions write them.
AFFILIATIONS = [
    "Hellenic Pasteur Institute, Athens, Greece",
    "University of Thessaly, Greece",
    "Biomedical Sciences Research Center \u201cAlexander Fleming\u201d, Vari, Greece",
    "Archimedes Unit, Athena Research Center, Athens, Greece",
]

METHODS_APPROVED = True

METHODS = (
    "**Data.** Positive miRNA–MRE pairs from chimeric AGO2 eCLIP in HEK293T cells "
    "(Manakov et al. 2022), with the clustering-based negatives of miRBench "
    "(Grešová et al. 2025). The Manakov data are split into train, test and leftout "
    "sets: train and test share miRNA families, while the leftout set holds families "
    "absent from both. "
    "**CNN.** The miRNA (≤ 30 nt) and MRE (50 nt) form a pairing matrix with a learned "
    "16-channel embedding, processed by six convolutional blocks, generalized-mean "
    "pooling and a two-layer head. "
    "**CatBoost.** IntaRNA minimum-free-energy and suboptimal duplexes, shuffled-background "
    "energy z-scores and phyloP conservation yield 26 selected features. "
    "**Training and evaluation.** Both models were trained with miRNA-family-based "
    "validation splits, and the leftout set measures performance on unseen families."
)

RESULTS = (
    "On the Manakov test and leftout sets both models outperform TargetScan (APS 0.80 "
    "and 0.81): the sequence CNN reaches 0.87 on both, and CatBoost 0.84 on both. "
    "The CNN scores higher than CatBoost on every evaluation set (Fig. 4), including the "
    "unseen SAEC cells (0.81 vs 0.76). CatBoost stays natively interpretable: exact "
    "TreeSHAP attributions (Fig. 5) show its predictions are driven by the miRNA's "
    "binding energy relative to a shuffled background, seed pairing energy, seed AU "
    "content and G·U wobbles in the seed, all canonical determinants of miRNA targeting."
)

CAPTIONS = {
    1: ("Fig. 1  Method overview. Both models score the same miRNA–MRE pair. (A) The CNN "
        "reads a learned miRNA × MRE pairing matrix with six convolutional blocks and a "
        "small head. (B) CatBoost scores 26 features derived from IntaRNA duplexes, "
        "conservation and a shuffled background, and explains each prediction with "
        "TreeSHAP."),
    2: ("Fig. 2  Model selection. APS against inference time for 25 AutoGluon candidates "
        "on the Manakov leftout set. Filled points form the Pareto front; the shipped "
        "CatBoost (orange) is within 0.005 APS of the most accurate candidate at a small "
        "fraction of its inference time."),
    3: ("Fig. 3  Both models against the predictors benchmarked by miRBench, as average "
        "precision on four evaluation sets. Reference values are the miRBench benchmark "
        "table; ours come from the same predictions as Fig. 4."),
    4: ("Fig. 4  APS of both models on each evaluation set, with 95% bootstrap intervals. "
        "Grey ticks mark the APS of a random classifier; Δ is CNN minus CatBoost."),
    5: ("Fig. 5  Top CatBoost features by mean |TreeSHAP| on a family-balanced Manakov "
        "test sample. Each point is one miRNA–MRE pair, coloured by the feature's value."),
}
