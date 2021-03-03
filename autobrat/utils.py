import spacy
from scripts.utils import Collection


def load_training_entities(collection: Collection):
    nlp = spacy.load('es')
    # collection = load_training_data(corpus)

    entity_types = set(keyphrase.label for sentence in collection.sentences
                       for keyphrase in sentence.keyphrases)
    sentences = [nlp(s.text) for s in collection.sentences]

    # print(entity_types, sentences)

    mapping = [['O'] * len(s) for s in sentences]

    for entity_type in sorted(entity_types):
        entities = [[p.spans for p in s.keyphrases if p.label == entity_type]
                    for s in collection.sentences]
        bilouv = to_biluov(sentences, entities)
        # print(entities)

        for accum, tags in zip(mapping, bilouv):
            for i, (previous_tag, new_tag) in enumerate(zip(accum, tags)):
                if previous_tag == 'O' and new_tag != 'O':
                    accum[i] = f"{new_tag}_{entity_type}"

            # print(accum)

    return sentences, mapping


def to_biluov(tokensxsentence, entitiesxsentence):
    labelsxsentence = []
    for tokens, entities in zip(tokensxsentence, entitiesxsentence):
        offset = 0
        labels = []
        for token in tokens:
            # Recently found that (token.idx, token.idx + len(token)) is the span
            matches = find_match(offset, offset + len(token.text), entities)
            tag = select_tag(matches)
            labels.append(tag)
            offset += len(token.text_with_ws)
        labelsxsentence.append(labels)

    return labelsxsentence  #, "BILUOV"


def find_match(start, end, entities):
    def match(other):
        return other[0] <= start and end <= other[1]

    matches = []
    for spans in entities:

        # UNIT
        if len(spans) == 1:
            if match(spans[0]):
                matches.append((spans[0], "U"))
            continue

        # BEGIN
        begin, *tail = spans
        if match(begin):
            matches.append((begin, "B"))
            continue

        # LAST
        *body, last = tail
        if match(last):
            matches.append((last, "L"))
            continue

        # INNER
        for inner in body:
            if match(inner):
                matches.append((inner, "I"))
                break

    return matches


def select_tag(matches):
    if not matches:
        return "O"
    if len(matches) == 1:
        return matches[0][1]
    tags = [tag for _, tag in matches]
    return "U" if ("U" in tags and not "B" in tags
                   and not "L" in tags) else "V"
