"""
Nested Annotation System, uses specially prepared training material.
Developed by Ismail Prada Ziegler at the University of Bern.

Otherwise, follows the same license as the rest of the forked repo.
"""

from pathlib import Path

from cassis import Cas

from flair.nn import Classifier as Tagger
from flair.data import Sentence, Token

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, SENTENCE_TYPE, TOKEN_TYPE


def fix_whitespaces(cas_tokens):
    """
    Convert the CAS tokens to flair tokens with correct whitespaces.
    """
    text = []
    for cas_token, following_cas_token in zip(cas_tokens, cas_tokens[1:] + [None]):
        if following_cas_token is not None:
            dist = following_cas_token.begin - cas_token.end
        else:
            dist = 1
        token = Token(
            cas_token.get_covered_text(),
            whitespace_after=dist,
            start_position=cas_token.begin
        )
        text.append(token)
    return text


def predict_recursive(sent, tagger, cas, layer, feature, prev_sent_len=None, offset=0):
    if prev_sent_len == len(sent):
        return []

    sent = Sentence(sent)
    
    tagger.predict(sent)

    #ents = sent.to_dict()["entities"]
    ents = sent.get_spans()

    for entity in ents:
        """
        entity["fixed_start_pos"] = entity["start_pos"] + offset
        entity["fixed_end_pos"] = entity["end_pos"] + offset

        begin = entity["fixed_start_pos"]
        end = entity["fixed_end_pos"]
        """
        begin = entity.start_position
        end = entity.end_position
        label = entity.tag
        #print(label, entity.text, begin, end)  # RESTORE THIS FOR SERVER
        prediction = create_prediction(cas, layer, feature, begin, end, label)
        cas.add(prediction)

    for entity in ents:
        if entity.tag in ["head", "money", "date", "time.rec"]:  # NOTE IMPORTANT TO FIT THIS
            continue
        #if sorted(entity["labels"], key=lambda x: x["confidence"], reverse=True)[0]["value"] in ["head", "money", "date", "time.rec"]:  # NOTE IMPORTANT TO FIT THIS
        #    continue
        # copy the tokens so they can be used for the next recursion
        tokens = []
        for token in entity.tokens:
            tokens.append(
                Token(
                    token.text,
                    whitespace_after=token.whitespace_after,
                    start_position=token.start_position
                )
            )
        predict_recursive(tokens, tagger, cas, layer, feature, prev_sent_len=len(sent.tokens), offset=offset)


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
            
            
