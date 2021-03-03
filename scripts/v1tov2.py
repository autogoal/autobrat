# coding: utf8

# > The process is straightforward now that Collection supports `legacy`

import argparse
from pathlib import Path

from scripts.tools import (
    AnnFile,
    AttributeAnnotation,
    EntityAnnotation,
    EventAnnotation,
    RelationAnnotation,
    SameAsAnnotation,
)


def global_count():
    if hasattr(global_count, "count"):
        global_count.count += 1
    else:
        global_count.count = 0
    return global_count.count


def transform_entity(ann: EntityAnnotation):
    yield ann


def transform_event(ann: EventAnnotation, mapper: dict):
    source = ann.ref
    for label, ref in ann.args.items():
        if label.startswith("Subject"):
            label = "subject"
        elif label.startswith("Target"):
            label = "target"
        elif label.startswith("Domain"):
            label = "domain"
        elif label.startswith("Arg"):
            label = "arg"
        elif label.startswith("in-context"):
            label = "in-context"
        elif label.startswith("in-place"):
            label = "in-place"
        elif label.startswith("in-time"):
            label = "in-time"
        else:
            raise ValueError(label)

        yield RelationAnnotation("R%d" % global_count(), label, source, mapper[ref])


def transform_relation(ann: RelationAnnotation, mapper: dict):
    yield RelationAnnotation(
        "R%d" % global_count(), ann.type, mapper[ann.arg1], mapper[ann.arg2]
    )


def transform_attribute(ann: AttributeAnnotation, mapper: dict):
    yield AttributeAnnotation(ann.id, ann.type, mapper[ann.ref])


def transform_sameas(ann: SameAsAnnotation, mapper: dict):
    yield SameAsAnnotation(ann.id, ann.type, [mapper[ref] for ref in ann.args])


def transform_ann(ann, mapper: dict):
    # OO please
    if isinstance(ann, EntityAnnotation):
        return transform_entity(ann)
    elif isinstance(ann, EventAnnotation):
        return transform_event(ann, mapper)
    elif isinstance(ann, RelationAnnotation):
        return transform_relation(ann, mapper)
    elif isinstance(ann, AttributeAnnotation):
        return transform_attribute(ann, mapper)
    elif isinstance(ann, SameAsAnnotation):
        return transform_sameas(ann, mapper)
    raise TypeError()


def transform_annotated_file(ann_file: AnnFile) -> AnnFile:
    output = AnnFile()
    global_count.count = 0

    mapper = {}
    for ann in ann_file.annotations:
        if isinstance(ann, EntityAnnotation):
            mapper[ann.id] = ann.id
        elif isinstance(ann, EventAnnotation):
            mapper[ann.id] = ann.ref

    for ann in ann_file.annotations:
        transformed = transform_ann(ann, mapper)
        output.annotations.extend(transformed)

    return output


def transform_file(input_txt: Path, output_txt: Path):
    input_ann: Path = input_txt.parent / (input_txt.stem + ".ann")
    output_ann: Path = output_txt.parent / (output_txt.stem + ".ann")

    ann_file = AnnFile().load(input_ann)
    transformed = transform_annotated_file(ann_file)
    content = "\n".join(x.as_brat() for x in transformed.annotations)
    output_ann.write_text(content, encoding="utf8")

    output_txt.write_text(input_txt.read_text(encoding="utf8"), encoding="utf8")


def transform_directory(input_path: Path, output_path: Path):
    for pack in sorted(input_path.iterdir()):
        if pack.suffix != ".txt":
            continue
        output = output_path / pack.name
        transform_file(pack, output)


def main(input_path: Path, output_path: Path):
    if input_path.is_dir():
        transform_directory(input_path, output_path)
    else:
        transform_file(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    main(input_path, output_path)
