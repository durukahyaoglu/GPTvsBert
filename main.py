import argparse
import requests
from utils import Input, Relation, pure_text_from_html
from search import SearchEntry
from googleapiclient.discovery import build
from spanbert import SpanBERT
from spacy_help_functions import custom_extract_relations, custom_extract_relations, custom_tag_relations
from collections import defaultdict

import spacy

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
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


internal_relations = ["per:schools_attended", "per:employee_of", "per:cities_of_residence", "org:top_members/employees"]
relation_mapping = {key:value for key, value in zip(relations_of_interest.keys(), internal_relations)}

def run(input_packet: Input):
    nlp = spacy.load("en_core_web_lg")
    spanbert = None
    if input_packet.spanbert_mode:
        spanbert = SpanBERT("./pretrained_spanbert")

    print(input_packet)
    service = build("customsearch", "v1", developerKey=input_packet.google_api_key)
    query = input_packet.seed_query
    used_query = []
    docs_already_seen = []
    if(input_packet.spanbert_mode):
        total_relations = defaultdict(int)
    else:
        total_relations = []

    def one_iteration(query, iteration_counter):
        nonlocal total_relations
        print(f"=========== Iteration: {iteration_counter} - Query: {query} ===========")
        query_result = (
            service.cse()
            .list(
                q=query,
                cx=input_packet.google_engine_id,
            )
            .execute()
        )
        items = list(map(lambda item: SearchEntry.from_search_result(item), query_result["items"]))
        used_query.append(query)

        for idx, item in enumerate(items):
            if(item in docs_already_seen):
                continue
            print(f"\n\nURL ({idx+1} / {len(items)}): {item.link}")
            print(f"\tFetching text from url...")
            r = requests.get(item.link, headers=headers)
            if r.status_code // 100 != 2:
                print("Skipped because of error code:", r.status_code)
                continue
            html = r.text
            print(f"\tTrimming webpage content from {len(html)} to {10000} characters")
            text = pure_text_from_html(query, html)
            print(f"\tWebpage length (num characters): {len(text)}")
            print(f"\tAnnotating the webpage using spacy...")
            docs_already_seen.append(item)

            doc = nlp(text)
            if(input_packet.spanbert_mode):
                relation_of_interest = relations_of_interest[input_packet.relation.str_value]
                relations = custom_extract_relations(query, doc, spanbert, relation_of_interest, conf=input_packet.minimum_confidence)
                total_relations = produce_structured_output(relations, input_packet.relation, total_relations, input_packet.minimum_confidence)
            else:
                relation_of_interest = relations_of_interest[input_packet.relation.name]
                relations = custom_tag_relations(input_packet.relation.name, doc, relation_of_interest, total_relations, api_key=input_packet.open_ai_secret_key)
                total_relations = produce_structured_output_gpt3(relations, total_relations)

            print(f"\tRelations extracted from this website: {len(relations)} (Overall: {len(total_relations)})")

            if(len(total_relations) > input_packet.num_output_tuples and input_packet.gpt3_mode):
                break

    iteration_counter = 1
    while True:
        one_iteration(query, iteration_counter)
        #print(str(len(total_relations)) + " tuples extracted. Going into the next iteration.")
        print_final_results(total_relations, input_packet.relation, input_packet.spanbert_mode)
        print("Total # of iterations = " + str(iteration_counter))
        if(len(total_relations) >= input_packet.num_output_tuples):
            print("Enough relations extracted. Exiting the program.")
            return
        elif(len(total_relations) == 0):
            print("No such relations were found. Exiting the program.")
            return
        else:
            if input_packet.spanbert_mode:
                top_query = ""
                for k,v in total_relations.items():
                    top_query = k[0] + " " + k[2]
                    if(top_query not in used_query):
                        break
                #query = top_query
            else:
                for pair in total_relations:
                    top_query = pair[0] + " " + pair[1]
                    if(top_query not in used_query):
                        break
                #query = top_query
            if(query == top_query):
                #No more unique relations left - i.e. ISE has "stalled" before retrieving k high confidence tuples
                print("No more uniqie relations left. ISE has stalled before retrieving {0} high confidence tuples".format(input_packet.num_output_tuples))
                return
            query = top_query
            iteration_counter += 1

def produce_structured_output_gpt3(relations, total_relations):

    for pair in relations:
        if pair not in total_relations:
            total_relations.append(pair)

    return total_relations


def produce_structured_output(relation_list, queried_relation, total_relations, minimum_confidence):
    for key, value in relation_list.items():
        #Check if the queried relation matches with the current relation only add if they do
        if(key[1] == relation_mapping[queried_relation.name]):
            #Check if the current relation value is equal to or greater than the minimum confidence inputted
            if(value >= minimum_confidence):
                #Check if the current relation is in the dictionary, if not just add it
                if(key not in total_relations):
                    total_relations[key] = value
                #Else only replace the value only if the new value is greater than the previous value
                else:
                    filtered_keys = [ele for tkey in total_relations for ele in tkey]
                    for val in filtered_keys:
                        if(val[0] == key[0] and val[2] == key[2] and total_relations[val] < value):
                            total_relations[val] = value
                            break


    total_relations = sorted(total_relations.items(), key=lambda x:x[1], reverse=True)
    converted_dict = dict(total_relations)

    return converted_dict

def print_final_results(total_relations, queried_relation, mode):

    print("============== ALL RELATIONS for {0} ( {1} )==============".format(relation_mapping[queried_relation.name], len(total_relations)))

    if(mode):
        for key, value in total_relations.items():
            print(f"Confidence: {str(value): <10}| Subject: {key[0]: <18} | Object: {key[2]: <18}")

    else:
        for pair in total_relations:
            txt = "Subject: {0: <18} | Object: {1: <18}"
            print(txt.format(pair[0], pair[1]))
            

def produce_structured_input(args) -> Input:
    return Input(
        spanbert_mode=args.spanbert,
        gpt3_mode=args.gpt3,
        google_api_key=args.google_api_key,
        google_engine_id=args.google_engine_id,
        open_ai_secret_key=args.open_ai_secret_key,
        relation=Relation(args.r),
        minimum_confidence=args.t,
        seed_query=args.q,
        num_output_tuples=args.k,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive search engine config")
    parser.add_argument('-spanbert', action='store_true')
    parser.add_argument('-gpt3', action='store_true')
    parser.add_argument("google_api_key", help="Search Engine Json Key")
    parser.add_argument("google_engine_id", help="Google Engine ID")
    parser.add_argument("open_ai_secret_key", help="Open API Secret Key")
    parser.add_argument("r", type=int, choices=range(1, 5), help="Relationships to extract")
    parser.add_argument("t", type=float, help="Minimum extraction confidence that we request for the tuples in the output")
    parser.add_argument("q", type=str, help="Seed query")
    parser.add_argument("k", type=int, help="Number of tuples that we request in the output")
    args = parser.parse_args()

    input_packet = produce_structured_input(args)
    run(input_packet)
