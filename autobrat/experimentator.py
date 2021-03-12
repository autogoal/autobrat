import imp
import logging
from pathlib import Path
from typing import List

from spacy import load
import spacy
from scripts.utils import Collection, Sentence
from .controller import AnotatorController
from .utils import load_training_entities
from random import choices, shuffle
from functools import reduce
from dataclasses import dataclass
from spacy.tokens.doc import Doc
from .utils import make_sentence
from scripts.score import subtaskA, compute_metrics

logger = logging.getLogger('experimentator')

nlp = spacy.load('es')

class Experimentator(object):
    def __init__(self, corpus: Collection) -> None:
        logger.info(f'Corpus total sentences: {len(corpus.sentences)}')
        lines, classes = load_training_entities(corpus)
        self.unique_clases = reduce(lambda x, y: x | y,
                                    [set(c) for c in classes])
        print(self.unique_clases)

        self.train_data = {
            sentence.text: ([w.text for w in line], category)
            for sentence, line, category in zip(corpus.sentences, lines,
                                                classes)
        }
        self.original_corpus = corpus.clone()
        self.training, self.test, self.sentences = self.select_traning_sentences(
            corpus)

        self.test_spacy_doc = {s.text: nlp(s.text) for s in self.test.sentences}

        self.sentences_to_train: List[str] = [s.text for s in self.training]

        super().__init__()

    def select_traning_sentences(self, corpus: Collection):
        size_training = 300
        size_test = 100
        # return Collection([s for s in choices(corpus.sentences, k=size)])
        sentences = corpus.sentences[:]
        shuffle(sentences)

        return Collection([s for s in sentences[:size_training]]), Collection([
            s for s in sentences[size_training:size_training + size_test]
        ]), [s.text for s in sentences[size_training + size_test:]]

    def score(self, submit: Collection):
        score_data = subtaskA(self.test, submit)
        metrics = compute_metrics(score_data, skipB=True, skipC=True)
        logger.info(f'Score: {metrics}')
        return metrics['f1']

    def run_experiment(self,
                       batch_size: int,
                       db_name: str = 'experiment.json'):
        controller = AnotatorController(self.sentences,
                                        self.training,
                                        db_path=Path(db_name))

        scores = []
        # while sentences
        sentences = controller.get_batch(batch_size)
        while sentences:
            self.sentences_to_train.extend(sentences)
            lines, classes = [], []
            for s in self.sentences_to_train:
                line, cls = self.train_data[s]
                lines.append(line)
                classes.append(cls)

            controller.annotator.fit_classes(lines, classes)


            sentences = []
            predictions = controller.annotator.final_prediction([s for s in self.test_spacy_doc])
            for (s, spacy_doc), prediction in zip(self.test_spacy_doc.items(), predictions):
                sentence = make_sentence(spacy_doc, prediction, self.unique_clases)
                sentence.fix_ids()
                sentences.append(sentence)

            predicted_collection = Collection(sentences)

            scores.append(self.score(predicted_collection))
            sentences = controller.get_batch(batch_size)

        return scores

    def train_with_all(self):
        controller = AnotatorController(self.sentences,
                                self.original_corpus,
                                db_path=Path('fullcorpus.json'))
        sentences = []
        predictions = controller.annotator.final_prediction([s for s in self.test_spacy_doc])
        for (s, spacy_doc), prediction in zip(self.test_spacy_doc.items(), predictions):
            sentence = make_sentence(spacy_doc, prediction, self.unique_clases)
            sentence.fix_ids()
            sentences.append(sentence)

        predicted_collection = Collection(sentences)
        return self.score(predicted_collection)