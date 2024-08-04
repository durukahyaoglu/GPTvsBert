import os
import openai
import json

example_relations = {
    "WORK_FOR": "\"Alec Radford\":\"WORK_FOR\":\"OpenAI\"",
    "SCHOOLS_ATTENDED": "\"Jeff Bezos\":\"SCHOOLS_ATTENDED\":\"Princeton University\"",
    "LIVE_IN": "\"Mariah Carey\":\"LIVE_IN\":\"New York City\"",
    "TOP_MEMBER_EMPLOYEES": "\"Jensen Huang\":\"TOP_MEMBER_EMPLOYEES\":\"Nvidia\"",
}

def get_openai_completion(prompt, model, max_tokens, temperature = 0.2, top_p = 1, frequency_penalty = 0, presence_penalty = 0):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    response_text = response['choices'][0]['text']
    return response_text

def call_gpt3(relation, subject, object, sentence, api_key):
    openai.api_key = api_key
    models_li = openai.Model.list()
    prompt_text = f"Given a sentence, extract all \"{subject}\" and \"{object}\" pairs for the relationship \"{relation}\" in this sentence. " + \
                f"Output should be in the form: [\"{subject}\":\"{relation}\":\"{object}\"]. For example, {example_relations[relation]}. In the case of multiple outouts, each output seperated by ;. If no pairs are found return NO PAIRS FOUND." +  " Sentence: "

    prompt_text += sentence + " "
            #+ ". In the form: Subject:Object. List of sentences: " + sentence_str + " extracted: ""

    model = 'text-davinci-003'
    max_tokens = 100
    temperature = 0.28
    top_p = 1
    frequency_penalty = 0.2
    presence_penalty = 0.2

    response_text = get_openai_completion(prompt_text, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)
    if("NO PAIRS FOUND" in response_text):
        return []
    res = format_response(response_text)
    return res

def format_response(response_text):

    #First clean up trailing characters before the beginning of the result dictionary
    response_text = response_text.strip()
    #response_text = response_text.rsplit(';',1)[0]

    #Second clean up the characters after the last come - i.e. unfinished dictionary entries
    response_text_list = response_text.split(";")

    res = []

    for answer in response_text_list:
        parts = answer.split(":")
        if len(parts) > 3:
            print("Cannot parse:", answer)
            continue

        # Fix ill-formatted outputs
        cleaned_text = "["
        for part in parts:
            if part[0] != "\"":
                cleaned_text += "\""
            cleaned_text += part
            cleaned_text += ","
            if part[-1] != "\"":
                cleaned_text += "\""
        cleaned_text = cleaned_text[:-1] + "]"

        try:
            pairs = json.loads(cleaned_text)
            res.append((pairs[0],pairs[2]))
        except:
            print("Cannot parse: ", answer)

    return res

def main():

    prompt_text = """ Given a sentence, extract all the Nouns.

sentence: Rob is an engineer at NASA and he lives in California.
extracted: """

    model = 'text-davinci-003'
    max_tokens = 100
    temperature = 0.2
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0

    response_text = get_openai_completion(prompt_text, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)
    print(response_text)



