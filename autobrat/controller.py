import collections
from functools import reduce
from typing import List, Optional
from pathlib import Path
from .annotator import SentencesAnnotator
from scripts.utils import Collection, Sentence
from tinydb import TinyDB, Query
from random import choice
from string import ascii_lowercase, digits
import shutil
import logging

logger = logging.getLogger('AnnotatorController')

chars = ascii_lowercase + digits


def generate_random_str(size: int = 10):
    return ''.join(choice(chars) for _ in range(size))


class AnotatorController():
    def __init__(
            self,
            sentences: List[str],
            baseline_collection: Collection,
            generated_pack_path: Path = Path('./generated_packs'),
            closed_packs_path: Path = Path('./closed_packs'),
            db_path: Path = Path('./sentencedb.json'),
            sentence_annotator: Optional[SentencesAnnotator] = None) -> None:

        self.generated_pack_path = generated_pack_path
        self.closed_packs_path = closed_packs_path

        self.number_of_models = 3

        self.generated_pack_path.mkdir(exist_ok=True)
        self.closed_packs_path.mkdir(exist_ok=True)

        self.db = TinyDB(db_path)
        saved_sentences = set(s['text'] for s in self.db.all())

        self._load_sentences(sentences, saved_sentences)
        collection = baseline_collection.clone()
        collection.load_dir(closed_packs_path)

        self.annotator = sentence_annotator
        if self.annotator is None:
            self.annotator = SentencesAnnotator.generated_classifier_from_dataset(
                collection, self.number_of_models)

    def _load_sentences(self,
                        sentences: List[str],
                        ignore_sentences: List[str] = []):

        for line in sentences:
            if not line or line in ignore_sentences:
                continue

            self.db.insert({'text': line, 'in_pack': False})

    def update_selected(self, sentences):
        Senteces = Query()
        self.db.update({'in_pack': True},
                       reduce(lambda x, y: x | y,
                              [Senteces.text == s for s in sentences]))

    def build_pack(self, dest_folder: Path, pack_name: str,
                   sentences: List[str]):
        dest_folder.mkdir()

        file = dest_folder / (pack_name + '.txt')
        file.write_text('\n'.join(sentences))

    def generate_pack(self,
                      dest_folder: Optional[Path] = None,
                      pack_size: int = 10):
        if dest_folder is None:
            dest_folder = self.generated_pack_path
        pack_name = generate_random_str()

        selected = self.get_batch(pack_size)

        self.build_pack(dest_folder / (pack_name), pack_name, selected)

    def get_batch(self,
                  batch_size: int,
                  set_procesed: bool = True) -> List[str]:
        Senteces = Query()

        texts = [s['text'] for s in self.db.search(Senteces.in_pack == False)]

        entropies = self.annotator.get_entropy_bulk(texts)

        sentences = [s for s in zip(texts, entropies)]

        sentences.sort(key=lambda x: x[1], reverse=True)

        selected = [s[0] for s in sentences[:batch_size]]

        if not selected:
            return []

        if set_procesed:
            self.update_selected(selected)

        return [s[0] for s in sentences]

    def close_pack(self, path: Path):
        collection = Collection()
        collection.load_dir(path)

        self.annotator.fit(collection)

        shutil.move(str(path), str(self.closed_packs_path))
        logger.info(
            f'Finish pack moving to closed pack folder ({path}) -> ({self.closed_packs_path})'
        )

    @staticmethod
    def load_from_files(
        self,
        sentences_files: List[Path],
        baseline_collection: Path,
        generated_pack_path: Path = Path('./generated_packs'),
        closed_packs_path: Path = Path('./closed_packs'),
        db_path: Path = Path('./sentencedb.json'),
        sentence_annotator: Optional[SentencesAnnotator] = None
    ) -> "AnotatorController":

        sentences = []

        for file in sentences_files:
            sentences.extend([line[:-1] for line in file.open() if line])
        collection = Collection()
        collection.load_dir(baseline_collection)
        return AnotatorController(sentences, collection, generated_pack_path,
                                  closed_packs_path, db_path,
                                  sentence_annotator)
