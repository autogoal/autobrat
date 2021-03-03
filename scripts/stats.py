# coding: utf8

from collections import defaultdict
from pathlib import Path

from scripts.agreement import load_corpus
from scripts.utils import Collection, CollectionV1Handler, CollectionV2Handler


def count_labels_on(corpus: Collection):
    counter = defaultdict(int)

    for sentence in corpus.sentences:
        for kp in sentence.keyphrases:
            counter[kp.label] += 1
            for attr in kp.attributes:
                counter[attr.label] += 1
        for rel in sentence.relations:
            counter[rel.label] += 1

    return counter


def count_labels(path: Path, handler=None):
    corpus = handler.load_dir(Collection(), path) if handler else load_corpus(path)
    return count_labels_on(corpus)


def count_labels_based_on(path: Path, reference: Path):
    collection = load_corpus(path)
    reference = CollectionV1Handler.load_dir(Collection(), reference)

    sentences = []
    for ref_sent in reference.sentences:
        for sent in collection.sentences:
            if sent.text == ref_sent.text:
                sentences.append(sent)
                break

    print(len(collection))
    print(len(reference))
    print(len(sentences))

    return count_labels_on(Collection(sentences))


def count_complex_entities(path: Path):
    count = 0
    visited = set()
    corpus = load_corpus(path)
    for sentence in corpus.sentences:
        for relation in sentence.relations:
            if (
                relation.from_phrase.id not in visited
                and relation.label in ["subject", "target"]
                and relation.to_phrase.label in ["Action", "Predicate"]
            ):
                visited.add(relation.from_phrase.id)
                count += 1
    return count


def main():
    # print(count_complex_entities(Path("./data/v1/medline/phase3-review")))
    # print(count_labels(Path("./data/v1/medline/")))
    # print(
    #     count_labels_based_on(
    #         Path("./data/v1/medline/phase3-review"),
    #         Path("./data/v1/ehealth19/testing/scenario3-taskB/"),
    #     )
    # )
    pass


if __name__ == "__main__":
    main()
