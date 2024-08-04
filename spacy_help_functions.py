import spacy
from collections import defaultdict
from gpt3 import call_gpt3
import re
from time import sleep

spacy2bert = {
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "DATE": "DATE"
        }

bert2spacy = {
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "CITY": "GPE",
        "COUNTRY": "GPE",
        "STATE_OR_PROVINCE": "GPE",
        "DATE": "DATE"
        }

relations_of_interest = {
    "SCHOOLS_ATTENDED": {
        "entities": ["PERSON", "ORGANIZATION"],
        "subj": ["PERSON"], "obj": ["ORGANIZATION"]
    },
    "WORK_FOR": {
        "entities": ["PERSON", "ORGANIZATION"],
        "subj": ["PERSON"], "obj": ["ORGANIZATION"]
    },
    "LIVE_IN": {
        "entities": ["PERSON", "CITY", "STATE_OR_PROVINCE", "COUNTRY"],
        "subj": ["PERSON"], "obj": ["CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    },
    "TOP_MEMBER_EMPLOYEES": {
        "entities": ["PERSON", "ORGANIZATION"],
        "subj": ["PERSON"], "obj": ["ORGANIZATION"]
    },
}

def get_entities(sentence, entities_of_interest):
    return [(e.text, spacy2bert[e.label_]) for e in sentence.ents if e.label_ in spacy2bert]

def get_relation_with_gpt3(text, entity_pairs, relation, total_relations, api_key):
    backoff_timeout = 5
    result_list = []

    #for ep in entity_pairs:
    search_ep = relations_of_interest[relation]
    subj = ','.join(search_ep['subj'])
    obj = ','.join(search_ep['obj'])
    #entity_name_1 = ''
    #entity_name_2 = ''
    #if(ep[1][1] in subj and ep[2][1] in obj):
        #entity_name_1, entity_name_2 = ep[1][1], ep[2][1]
    #elif(ep[1][1] in obj and ep[2][1] in subj):
        #entity_name_2, entity_name_1 = ep[1][1], ep[2][1]
    retry = 0
    current_list = []
    while retry < 3:
        try:
            current_list = call_gpt3(relation, subj, obj, text, api_key=api_key)
            break
        except:
            print(f"\t\tEncountered Rate Limit Wait {backoff_timeout}s")
            sleep(backoff_timeout)
        retry += 1
    sleep(1.5)
    if(len(current_list) == 0):
        return []

    for pair in current_list:
        print("\n\t\t=== Extracted Relation ===")
        print("\t\tSentence: {0}".format(text))
        print("\t\tSubject: {0} ; Object: {1} ;".format(pair[0],pair[1]))
        if(pair not in result_list and pair not in total_relations):
            result_list.append(pair)
            print("\t\tAdding to the set of relations")
        else:
            print("\t\tDuplicate. Ignoring this.")
            print("\t\t==========\n")
    return result_list

def custom_tag_relations(relation, doc, relation_of_interest, total_relations, api_key):
    num_sentences = len([s for s in doc.sents])
    num_annotations = 0
    print(f"\tExtracted {num_sentences} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
    entities_of_interest = relation_of_interest["entities"]
    subj_required = relation_of_interest["subj"]
    obj_required = relation_of_interest["obj"]
    sentences_to_analyze = []
    result_list = []

    for idx, sentence in enumerate(doc.sents):
        if (idx+1) % 5 == 0:
            print(f"\tProcessed {idx+1} / {num_sentences} sentences")
        entity_pairs = create_entity_pairs(sentence, entities_of_interest, window_size = 20)
        
        def check_matching_entties(ep):
            return (ep[1][1] in subj_required and ep[2][1] in obj_required or ep[1][1] in obj_required and ep[2][1] in subj_required)
        filtered_entity_pairs = list(filter(check_matching_entties, entity_pairs))
        if(len(filtered_entity_pairs) == 0):
            continue
        num_annotations += len(filtered_entity_pairs)
        cleaned_text = re.sub("[\(\[].*?[\)\]]", "", sentence.text)
        cleaned_text = cleaned_text.strip()
        if cleaned_text in sentences_to_analyze:
            continue
        sentences_to_analyze.append(cleaned_text)
        result_list = get_relation_with_gpt3(cleaned_text, filtered_entity_pairs, relation, total_relations=total_relations, api_key=api_key)
        if(result_list is None or len(result_list) == 0):
            continue
        total_relations.extend(result_list)

        print(f"\n\n\tExtracted annotations for  {num_annotations}  out of total  {num_sentences}  sentences")
    return result_list

def custom_extract_relations(query, doc, spanbert, relation_of_interest, conf=0.7):
    num_sentences = len([s for s in doc.sents])
    entities_of_interest = relation_of_interest["entities"]
    subj_of_interest = relation_of_interest["subj"]
    obj_of_interest = relation_of_interest["obj"]

    print(f"\tExtracted {num_sentences} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
    res = defaultdict(int)
    for idx, sentence in enumerate(doc.sents):
        if (idx+1) % 5 == 0:
            print(f"\tProcessed {idx+1} / {num_sentences} sentences")
        entity_pairs = create_entity_pairs(sentence, entities_of_interest, window_size=20)
        examples = []
        for ep in entity_pairs:
            ep1_word, ep1_entity_type, _ = ep[1]
            ep2_word, ep2_entity_type, _ = ep[2]
            if ep1_entity_type in subj_of_interest and ep2_entity_type in obj_of_interest:
                examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            if ep2_entity_type in subj_of_interest and ep1_entity_type in obj_of_interest:
                examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        try:
            preds = spanbert.predict(examples)
        except:
            continue
        for ex, pred in list(zip(examples, preds)):
            relation = pred[0]
            if relation == 'no_relation':
                continue
            print("\n\t\t=== Extracted Relation ===")
            print("\t\tTokens: {}".format(ex['tokens']))
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]
            print("\t\tRelation: {} (Confidence: {:.3f})".format(relation, confidence))
            print("\t\tSubject: {}\tObject: {}".format(subj, obj))
            if confidence > conf:
                if res[(subj, relation, obj)] < confidence:
                    res[(subj, relation, obj)] = confidence
                    print("\t\tAdding to set of extracted relations")
                else:
                    print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
            else:
                print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
            print("\t\t==========\n")

    print(f"\tExtracted annotations for {len(res)}  out of total {num_sentences} sentences")

    return res

def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    '''
    Input: a spaCy Sentence object and a list of entities of interest
    Output: list of extracted entity pairs: (text, entity1, entity2)
    '''
    if entities_of_interest is not None:
        entities_of_interest = {bert2spacy[b] for b in entities_of_interest}
    ents = sents_doc.ents # get entities for given sentence

    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if entities_of_interest is not None and e1.label_ not in entities_of_interest:
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if entities_of_interest is not None and e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                continue

            if (1 <= (e2.start - e1.end) <= window_size):

                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0

                # Find end of sentence
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc

                if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start-gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start-gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs

