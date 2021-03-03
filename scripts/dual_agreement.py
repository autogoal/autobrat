import argparse
from pathlib import Path

from scripts.agreement import main as agreement_main

from scripts.score import (
    CORRECT_A,
    CORRECT_B,
    CORRECT_C,
    INCORRECT_A,
    MISSING_A,
    MISSING_B,
    MISSING_C,
    PARTIAL_A,
    SPURIOUS_A,
    SPURIOUS_B,
    SPURIOUS_C,
)


def partial_score(keyphrase1, keyphrase2):
    intersection, union = overlap_spans(keyphrase1.spans, keyphrase2.spans)
    return intersection / union


def overlap_spans(spans1, spans2):
    """
    >>> overlap_spans([ (2,8) ], [ (4,10) ])
    (4, 8)
    >>> overlap_spans([ (2,8) ], [ (8,10) ])
    (0, 8)
    >>> overlap_spans([ (2,8), (8,10) ], [ (8,10) ])
    (2, 8)
    >>> overlap_spans([ (2,8), (9,10) ], [ (8,10) ])
    (1, 8)
    """

    tags = [0, 0] * len(spans1) + [1, 1] * len(spans2)
    spans = [x for span in spans1 + spans2 for x in span]

    state = [False, False]
    last = 0
    union = 0
    intersection = 0

    for span, tag in sorted(zip(spans, tags)):
        delta = span - last
        if all(state):
            intersection += delta
        if any(state):
            union += delta
        last = span
        state[tag] ^= True  # same as: state[tag] = not state[tag]

    return intersection, union


def concepts_agreement(data, all=True):
    assert (
        all or not data[INCORRECT_A]
    ), "For a single concept class, no incorrect matches are allowed"

    c_score = len(data[CORRECT_A])
    p_score = sum(partial_score(a, b) for a, b in data[PARTIAL_A].items())
    n = sum(
        len(data[x]) for x in [CORRECT_A, PARTIAL_A, MISSING_A, SPURIOUS_A, INCORRECT_A]
    )
    # print(c_score, p_score, len(data[PARTIAL_A]), len(data[MISSING_A]), len(data[SPURIOUS_A]), len(data[INCORRECT_A]))
    return c_score, p_score, n  # (c_score + p_score) / n


def relations_agreement(data):
    c_score = len(data[CORRECT_B])
    n = sum(len(data[x]) for x in [CORRECT_B, MISSING_B, SPURIOUS_B])
    # print(c_score, len(data[MISSING_B]), len(data[SPURIOUS_B]))
    return c_score, n  # c_score / n if n else 1.0


def attributes_agreement(data):
    c_score = len(data[CORRECT_C])
    n = sum(len(data[x]) for x in [CORRECT_C, MISSING_C, SPURIOUS_C])
    # print(c_score, len(data[MISSING_B]), len(data[SPURIOUS_B]))
    return c_score, n  # c_score / n if n else 1.0


def agreement(data):
    c_score = len(data[CORRECT_A]) + len(data[CORRECT_B]) + len(data[CORRECT_C])
    p_score = sum(partial_score(a, b) for a, b in data[PARTIAL_A].items())
    n = sum(len(ann) for ann in data.values())
    # print(c_score, p_score, len(data[PARTIAL_A]), len(data[MISSING_A]),
    #  len(data[SPURIOUS_A]), len(data[INCORRECT_A]),
    #  len(data[MISSING_B]), len(data[SPURIOUS_B]))
    return c_score, p_score, n  # (c_score + p_score) / n


def compute_metrics(data1, data2):
    c_score1, p_score1, n1 = concepts_agreement(data1)
    c_score2, p_score2, n2 = concepts_agreement(data2)

    c_score11, n11 = relations_agreement(data1)
    c_score22, n22 = relations_agreement(data2)

    c_score1111, n1111 = attributes_agreement(data1)
    c_score2222, n2222 = attributes_agreement(data2)

    c_score111, p_score111, n111 = agreement(data1)
    c_score222, p_score222, n222 = agreement(data2)

    return {
        "concepts_agreement": (c_score1 + p_score1 + c_score2 + p_score2) / (n1 + n2),
        "relations_agreement": (c_score11 + c_score22) / (n11 + n22)
        if n11 + n22
        else 1.0,
        "attributes_agreemet": (c_score1111 + c_score2222) / (n1111 + n2222)
        if n1111 + n2222
        else 1.0,
        "agreement": (c_score111 + p_score111 + c_score222 + p_score222)
        / (n111 + n222),
    }


def main(gold_dir: Path, submit_dir1: Path, submit_dir2: Path, propagate_error=True):
    _, history1 = agreement_main(gold_dir, submit_dir1, propagate_error)
    _, history2 = agreement_main(gold_dir, submit_dir2, propagate_error)

    print("================================================")
    for (label1, data1), (label2, data2) in zip(history1.items(), history2.items()):
        assert label1 == label2

        metrics = compute_metrics(data1, data2)

        for key, value in metrics.items():
            print(label1, "{0}: {1:0.4}".format(key, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gold")
    parser.add_argument("submit1")
    parser.add_argument("submit2")
    parser.add_argument("--isolate", action="store_false")
    args = parser.parse_args()
    main(Path(args.gold), Path(args.submit1), Path(args.submit2), args.isolate)
