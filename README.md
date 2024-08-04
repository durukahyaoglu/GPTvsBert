### Information Extraction System Implementation using Iterative Set Expansion
Implementation of an information extraction system that uses Iterative Set Expansion over search results returned by Google, parsed with BeautifulSoup, spaCY and processed with GPT3 and Bert, to extract relations from the web, of a given type among four categories.

This is a project that I developped in Fall 2017 for the course COMS 6111 Advanced Database Systems taught by Professor Luis Gravano at Columbia University. Below are the instructions and the readme that I submitted for the project.

### Team Info
* Hoseung Lee (hl3605)
* Duru Kahyaoglu (dk2565)

### Files
* main.py
* search.py
* spacy_help_functions.py
* spanbert.py
* utils.py
* gpt3.py
* pytorch_pretrained_bert/file_utils.py
* pytorch_pretrained_bert/modeling.py
* pytorch_pretrained_bert/optimization.py
* pytorch_pretrained_bert/tokenization.py
* LICENSE
* references.txt
* relations.txt
* requirements.txt
* result.txt
* setup.sh
* run.sh
* pretrained_spanbert folder **please note that we didn't include this folder in our gradescope submission due to the submission size constraint but it's present in our project in our GCP VMs



### Execution Manual
* How to run the program (Must be in GCP VM)

```
. dbproj/bin/activate

cd ads/project2

sudo apt-get update
sudo apt-get install python3-pip python-setuptools python-dev build-essential
sudo -H pip3 install --upgrade pip
python3 -m spacy download en_core_web_lg
pip3 install -r requirements.txt
bash download_finetuned.sh

python3 main.py -spanbert [Client key] [Engine key] [OPEN AI key] 2 0.7 \
    "bill gates microsoft" 10
```

### Design of the Project

External Libraries Used: spaCy, spanBert, gpt3API, BeautifulSoup 
(More external libraries are used and you can reach the full list in requirements.txt)

Conceptually, this project is divided into three tiers:
1. User exposed code
2. Helper functions to annotate the documents and call the appropriate extraction function (i.e. either Spanbert or GPT) depending on the user input
3. Extraction functions to run the desired extraction approach

Tier 1: The project exposes main.py to the user which accepts and checks for the integrity of the user's inputs, runs Google custom search for the query that the user desires, gets the top 10 results, and extracts the entity pairs for the desired relation, executing iterative set expansion using either SpanBERT or Open AI's GPT-3 API, until either k tuples are reached or the program is unable to find further entity relations. After the resulting entity pairs are aggregated and filtered for the desired conditions (i.e. for SpanBert, the confidence level has to be greater than the value that the user inputs, the relation has to be what the user is looking for) and printed in the required format.

Tier 2: Depending on the user's desired approach for extracting relations, main.py calls the helper functions defined in spacy_help_functions.py, which first extracts all of the relations for the given document and then calls the appropriate set extraction function as defined in the third tier of our system, which contains the logic for the two set extraction approaches as well as the spacy functions to extract entity pairs. This tier also processes, filters, and converts the results into a suitable data structured to be returned to the user in main.py.

Tier 3: This tier contains the functions for spacy, spanbert and gpt3. Most of the code here was taken from the reference implementations shared in the projects specs and was modified to fit the needs of our project.

### How We Carried Out Step 3

Here's a step by step description of how step 3 is carried out in this project:

1. While searching for the desired relation in the search results, for every search results found in main.py, first, the resulting document is first processed to extract only the pure text from it using the BeautifulSoup library and trimmed for the first 10,000 characters (line 77 in main.py - pure_text_from_html). After this, the pure text is parsed into sentences using spacy library (nlp function on line 81 of main.py).
2. The subject and object entity pair for the relation of interest is determined (For instance, we extract "PERSON" as the subject and "ORGANIZATION" as the object.)
3. The entity pair determined on step 2 and the inputs coming from step 1 and the user are used to call the "custom_extract_relations" (line 84) or "custom_tag_relations" (line 88) function as we defined in spacy_help_functions.py. Both of these functions first call a helper function called "create_entity_pairs" to first create entity pairs for each sentence in the document. Following the extraction of the entity pairs, "custom_extract_relations" the processing function for spanbert approach, iterates through all of these entity pairs, filters for the ones that have the correct subject and object, calls the "spanbert.predict" function (spacy_help_functions.py - line 142) to extract the relations and finally filters for the relations that have a confidence level greater than or equal to what the user inputs. These relation objects are returned as a dictionary, whose key is a tuple of subject, relation, and object and value is the confidence level. "produce_structured_output_gpt3" function in main.py filters these relations for the relation that the user is looking for and adds the ones that are above the desired confidence level to the overall list of relations. If the subject and the object was seen before, it's confidence value is updated only if the recent confidence level is above what was there before.
4. We follow a similar logic for the gpt3 approach as outlined in Step 3. Main.py calls the function "custom_tag_relations" function to first create entity pairs for the given document calling the "create_entity_pairs" function. The entity pairs extracted from this relation are first checked for having the required entity pairs for the given relation one by one and if they do, the gpt3 api is called on the sentence containing this entity pair and the relevant subject, object pairs for the given relation are extracted. The api's raw text result is converted into a list of subject and ibject tuples. Unlike spanbert, if the subject - object pair was processed before, we don't add duplicate the pair to the resulting list and simply skip it. The resulting list is then further processed with the "produce_structured_output_gpt3" function where we check if the relations extracted from the current document were previously encountered. If not, we add these relations to the total relations list, which is the data structure containing all of the results.

### Google Custom Search Engine JSON API Key and Engine ID
Google Custom Search Engine JSON API Key: *
Engine ID: *


