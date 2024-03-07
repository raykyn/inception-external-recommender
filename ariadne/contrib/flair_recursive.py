"""
Nested Annotation System, uses specially prepared training material.
Developed by Ismail Prada Ziegler at the University of Bern.

Otherwise, follows the same license as the rest of the forked repo.
"""

from pathlib import Path

from cassis import Cas

from flair.nn import Classifier as Tagger
from flair.data import Sentence

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE, TOKEN_TYPE


def fix_whitespaces(cas_tokens):
    # build sentence with correct whitespaces
    # (when using sentences, this should not be a problem afaik)
    text = ""
    last_end = 0
    for cas_token in cas_tokens:
        if cas_token.begin == last_end:
            text += cas_token.get_covered_text()
        else:
            text += " " + cas_token.get_covered_text()
        last_end = cas_token.end
    return text


def predict_recursive(sent, tagger, cas, layer, feature, prev_sent="", offset=0):
    if prev_sent == sent:
        return []

    sent = Sentence(sent, use_tokenizer=False)
    
    tagger.predict(sent)

    ents = sent.to_dict()["entities"]

    for entity in ents:
        entity["fixed_start_pos"] = entity["start_pos"] + offset
        entity["fixed_end_pos"] = entity["end_pos"] + offset

        begin = entity["fixed_start_pos"]
        end = entity["fixed_end_pos"]
        label = entity["labels"][0]["value"]
        prediction = create_prediction(cas, layer, feature, begin, end, label)
        cas.add(prediction)

    for entity in ents:
        if sorted(entity["labels"], key=lambda x: x["confidence"], reverse=True)[0]["value"] in ["head", "money", "date", "time.rec"]:  # NOTE IMPORTANT TO FIT THIS
            continue
        predict_recursive(entity["text"], tagger, cas, layer, feature, prev_sent=sent.to_dict()["text"], offset=offset + entity["start_pos"])


class FlairRecursiveTagger(Classifier):
    def __init__(self, model_name: str, model_directory: Path = None, split_sentences: bool = True):
        super().__init__(model_directory=model_directory)
        self._model = Tagger.load(model_name)
        self._split_sentences = split_sentences

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str): 
        # Extract the sentences from the CAS
        if self._split_sentences:
            raise NotImplementedError
            cas_sents = cas.select(SENTENCE_TYPE)
            sents = [Sentence(sent.get_covered_text(), use_tokenizer=False) for sent in cas_sents]
            offsets = [sent.begin for sent in cas_sents]

            # Find the named entities
            self._model.predict(sents)

            for offset, sent in zip(offsets, sents):
                # For every entity returned by spacy, create an annotation in the CAS
                for named_entity in sent.to_dict()["entities"]:
                    begin = named_entity["start_pos"] + offset
                    end = named_entity["end_pos"] + offset
                    label = named_entity["labels"][0]["value"]
                    prediction = create_prediction(cas, layer, feature, begin, end, label)
                    cas.add(prediction) 

        else:
            cas_tokens = cas.select(TOKEN_TYPE)

            text = fix_whitespaces(cas_tokens)

            predict_recursive(text, self._model, cas, layer, feature)
            
            
