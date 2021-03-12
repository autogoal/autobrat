from collections import defaultdict
import collections
from os import sendfile
from typing import Dict, List, Tuple
from math import log2
import spacy
from scripts.utils import Collection, Sentence
from autogoal.ml import AutoML
from autogoal.kb import Sentence, Word, Postag
from autogoal.kb import List as ag_List
import typing as t
from .utils import load_training_entities
from functools import reduce
import logging
import json

logger = logging.getLogger('SentenceAnnotator')


class SentencesAnnotator(object):
    nlp = spacy.load('es')

    def __init__(self, models: t.List[AutoML], collection_base: Collection,
                 unique_classes: t.List[str]) -> None:
        super().__init__()
        self.models = models
        self.collection_base = collection_base
        self.unique_classes = unique_classes

    def predict_prob(self, sentence: Sentence) -> Tuple[Sentence, float]:
        text = sentence.text

        clasifications = self.get_classifications(sentence.text)
        pass

    def predict(self, texts: t.List[str]) -> List[List[str]]:
        parsed_sentences = [[w.text for w in self.nlp(text)] for text in texts]
        ans = []
        for sentence in parsed_sentences:
            ans.append([])
            for classifier in self.models:
                prediction = classifier.predict([sentence])
                ans[-1].append(prediction[0])

        return ans

    def get_classifications(self, text: str):
        parsed_sentence = [w.text for w in self.nlp(text)]
        # print(parsed_sentence)
        ans = []
        for classifier in self.models:
            prediction = classifier.predict([parsed_sentence])
            ans.append(prediction[0])

        return ans

    def get_probs(self,
                  predictions: t.List[float]) -> t.List[Dict[str, float]]:

        size = len(predictions[0])
        ans = [defaultdict(lambda: 0) for i in range(size)]

        for prediction in predictions:
            for i, categorie in enumerate(prediction):
                ans[i][categorie] += 1 / len(predictions)

        return ans

    def final_prediction(self, texts: List[str]):
        predictions = self.predict(texts)
        probs = [self.get_probs(p) for p in predictions]

        ans = []
        for sentence in probs:
            ans.append([])
            for term in sentence:
                m = max( term.items(),key=lambda x: x[1])
                ans[-1].append(m[0])

        return ans

    def get_entropy(self, probs: t.List[Dict[str, float]]):
        return sum(-1 * sum([word * log2(word) for word in words.values()])
                   for words in probs)

    def get_entropy_bulk(self, sentences: t.List[str]) -> List[float]:
        predictions = self.predict(sentences)

        logger.info(json.dumps(predictions))
        probs = [self.get_probs(p) for p in predictions]
        logger.info(json.dumps(probs))
        entropy = [self.get_entropy(p) for p in probs]
        logger.info(json.dumps(entropy))

        return entropy

    @staticmethod
    def generated_classifier_from_dataset(data: Collection,
                                          number_of_models: int = 5):
        models = []
        lines, classes = load_training_entities(data)
        unique_clases = reduce(lambda x, y: x | y, [set(c) for c in classes])

        for _ in range(number_of_models):

            classifier = AutoML(
                input=ag_List(ag_List(Word())),
                output=ag_List(ag_List(Postag())),
            )

            classifier.fit([[w.text for w in l] for l in lines], classes)
            models.append(classifier)

        return SentencesAnnotator(models=models,
                                  collection_base=data,
                                  unique_classes=unique_clases)

    def fit(self, data: Collection):
        lines, classes = load_training_entities(data)
        lines = [[w.text for w in l] for l in lines]

        return self.fit_classes(lines, classes)

    def fit_classes(self, lines, classes):
        for model in self.models:
            model.best_pipeline_.send('train')
            model.best_pipeline_.run((lines, classes))
            model.best_pipeline_.send('eval')
