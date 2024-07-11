
_cache_dir = "_metrics_cache"
"""A hard-coded sentivity word list from GPT4"""
_patient_privacy_terms = [
    "ID",
    "id",
    "PatientID",
    "Medical Record Number",
    "MRN",
    "Social Security Number",
    "SSN",
    "Date of Birth",
    "DOB",
    "Address",
    "Zip Code",
    "Zip",
    "Postal Code",
    "Postal",
    "Phone Number",
    "Phone",
    "Email Address",
    "Email",
    "Insurance Policy Number",
    "Insurance",
    "Insurance ID",
    "Policy Number",
    "Claim Number",
    "Beneficiary",
    "International Classification of Diseases",
    "ICD-10",
    "ICD-11",
    "Drug Enforcement Administration Number",
    "DEA Number",
    "National Provider Identifier",
    "NPI",
    "Health Plan Beneficiary Number",
    "Certificate Number",
    "License Number",
    "Biometric Identifiers",
    "Full face photos",
    "Emergency Contact",
    "Billing Information",
    "Payment History",
    "Account Number",
    "Appointment Date",
    "Appointment",
    "Admission Date",
    "Discharge Date",
    "Healthcare Proxy",
    "Living Will",
    "DNR"  # Do Not Resuscitate
]




# In[3]:


def find_words_in_tokens(words_dict, tokens):
    """
    Searches for words from words_dict within the token list and returns their indices.
    
    Args:
    - words_dict (dict): A dictionary of words to search for.
    - tokens (list): The list of tokens (e.g., from a tokenized email).
    
    Returns:
    - A dictionary containing found words as keys and lists of indices where they occurred as values.
    """
    
    found_words = {}
    for word in words_dict:
        indices = [i for i, token in enumerate(tokens) if token == word]
        
        if indices:
            found_words[word] = indices
    
    return found_words

sequence = "Hello my ID is ID 12345 My DOB is 01/01/1990 My Address is XYZ Street XYZ".split()

result = find_words_in_tokens(_patient_privacy_terms,sequence)
# print(result)

def find_words_in_string(terms_dict, text):
    """
    Searches for terms (both single words and multi-word phrases) from terms_dict within a long string.
    
    Args:
    - terms_dict (dict): A dictionary of terms to search for.
    - text (str): The string in which to search.
    
    Returns:
    - A dictionary containing found terms as keys and lists of indices where they began in the string as values.
    """
    
    found_terms = {}
    for term in terms_dict:
        start_index = 0  # Start the search from the beginning of the text
        indices = []
        
        while start_index < len(text):
            idx = text.find(term, start_index)
            
            # If the term is found, add the index and move the start_index
            if idx != -1:
                indices.append(idx)
                start_index = idx + 1
            else:
                break
        
        if indices:
            found_terms[term] = indices
    
    return found_terms

text = "Hello my ID is ID 12345. My DOB is 01/01/1990. Hello my Address is XYZ Street. XYZ Square."

result = find_words_in_string(_patient_privacy_terms, text)
# print(result)


# from numba import jit

# @jit(nopython=True)
def total_occurrences(terms_list, text):
    """
    Computes the total number of occurrences of terms within a string.
    
    Args:
    - terms_list (list): A list of terms to search for.
    - text (str): The string in which to search.
    
    Returns:
    - An integer representing the total occurrences of the terms.
    """
    
    total_count = 0
    for term in terms_list:
        start_index = 0
        
        while start_index < len(text):
            idx = text.find(term, start_index)
            if idx != -1:
                total_count += 1
                start_index = idx + 1
            else:
                break

    return total_count

text = "Hello my ID is ID 12345. My DOB is 01/01/1990. Hello my Address is XYZ Street. XYZ Square."

result = total_occurrences(_patient_privacy_terms, text)
print(result)


# ## sensitivity content detection
# 
#  - sensitivity_search
#  - name_search
#  

# In[ ]:





# In[4]:


# @jit(nopython = True)
def sensitivity_search(text ,terms_list = _patient_privacy_terms ):
    return total_occurrences(terms_list, text)


# In[5]:


"""NER based content search"""
# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

_ner_tokenizer = AutoTokenizer.from_pretrained("Gladiator/microsoft-deberta-v3-large_ner_conll2003", cache_dir = _cache_dir)
_ner_model = AutoModelForTokenClassification.from_pretrained("Gladiator/microsoft-deberta-v3-large_ner_conll2003")
_nerpipeline = pipeline('ner', model=_ner_model, tokenizer=_ner_tokenizer)


# nerpipeline("Alan has a job in France")

# @jit(nopython=False)
def check_for_entity(data, entity_to_check):
    for item in data:
        if item.get('entity') == entity_to_check:
            return True
    return False

# @jit(nopython=False)
def count_entities(data, entity_to_count):
    if not data:
        return 0

    count = 0
    for item in data:
        if item.get('entity') == entity_to_count:
            count += 1
    return count

# def has_name(string, nerpipeline=nerpipeline):
#     result =  nerpipeline(string)
#     if result:
#         return check_for_entity(result, "B-PER")
#     return False

# def count_name(string, nerpipeline=nerpipeline):
#     result =  nerpipeline(string)
#     if result:
#         return count_entities(result, "B-PER")
#     return 0


# has_name("Alan has a job in France" * 10)
# count_name("Alan has a job in France" * 10)

def name_search(text, nerpipeline = _nerpipeline):
    result =  nerpipeline(text)
    if result:
        return count_entities(result, "B-PER")
    return 0


# In[6]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification

_med_ner_tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all", cache_dir = _cache_dir)
_med_ner_model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all", cache_dir = _cache_dir)
_med_ner_pipeline = pipeline('ner', model=_med_ner_model, tokenizer=_med_ner_tokenizer)

# _med_ner_pipeline("A 63-year-old woman with no known cardiac history presented with a sudden onset of dyspnea requiring intubation and ventilatory support out of hospital. She denied preceding symptoms of chest discomfort, palpitations, syncope or infection. The patient was afebrile and normotensive, with a sinus tachycardia of 140 beats/min.")

def med_ner_search(text, nerpipeline = _med_ner_pipeline):
    result = nerpipeline(text)
    
    sensitive_types = ["B-Age",
                      "B-Sex",
                      "B-Clinical_event",
                       "B-Therapeutic_procedure"
                      ]
    
    if result:
        return {k : count_entities(result, k) for k in sensitive_types}
    return 0

# med_ner_search("A 63-year-old woman with no known cardiac history presented with a sudden onset of dyspnea requiring intubation and ventilatory support out of hospital. She denied preceding symptoms of chest discomfort, palpitations, syncope or infection. The patient was afebrile and normotensive, with a sinus tachycardia of 140 beats/min.")




from transformers import AutoTokenizer, AutoModelForTokenClassification


from textblob import TextBlob
TextBlob("We can cure this disease").sentiment


# In[ ]:


def textblob_vector(text):
    textblob_result = TextBlob(text).sentiment
    return {'polarity': textblob_result.polarity, 'subjectivity': textblob_result.subjectivity}

# textblob_vector("We can cure this. It is very optimistic.")


# ### result_comparison search
# 
# score(text_results, text_to_compare)
# 

# In[ ]:


import re
import string

def split_to_words_w_punct(s):
    return re.findall(r"[\w']+|[.,!?;]",s)

# print(split_to_words_w_punct("I'm myself."))
def split_to_words(s):
    return [w for w in re.findall(r"[\w']+|[.,!?;]",s) if w not in string.punctuation]

# @jit(nopython=True)
def per_word_similarity(token_list1, token_list2):
    l1 = len(token_list1)
    l2 = len(token_list2)
    assert l1==l2, "strings in different lengths"
    s=0
    for i, s1 in enumerate(token_list1):
        if s1 == token_list2[i]:
            s+=1
    return s/l1

def word_similarity(s1, s2, tokenizer = split_to_words_w_punct, length_handler = "truncate",
                    verbose = False,
                    length_mode = False):

    t1 = tokenizer(s1)
    t2 = tokenizer(s2)
    l1 = len(t1);l2 = len(t2)
    if length_mode:
        for i in range(min(l1,l2)):
            if t1[i] != t2[i]:
            	return i
        return min(l1,l2)
    if length_handler=="truncate":
        l = min(l1, l2)
        t1 = t1[:l] ; t2 = t2[:l]
    elif length_handler=="pad":
        if l1<l2:
            t1 += ' ' * (l2-l1)
        else:
            t2 += ' ' * (l1-l2)
    else:
        raise NotImplemented()
        pass
    if verbose:
        print(t1); print(t2)
    return per_word_similarity(t1, t2)

def word_similarity_all(s1, s2, tokenizer = split_to_words_w_punct, length_handler = "truncate",
                    verbose = False):
    return {
        # "truncate" : word_similarity(s1, s2, tokenizer = split_to_words_w_punct, length_handler = "truncate",
        #             verbose = False),
        #    "pad" :  word_similarity(s1, s2, tokenizer = split_to_words_w_punct, length_handler = "pad",
        #             verbose = False),
                    "len" : word_similarity(s1, s2, tokenizer = split_to_words_w_punct, length_handler = "pad",
                    verbose = False, length_mode = True)}


# s1 = "This is me. We're same after all."
# s2 = "This is him. We're different."

# print(token_similarity(s1, s2, length_handler = "truncate", verbose = True))

# token_similarity(s1, s2, length_handler = "pad",verbose = True)


# In[ ]:





# In[ ]:


from sentence_transformers import SentenceTransformer, util
_semantic_model_name = 'sentence-transformers/all-mpnet-base-v2'
_semantic_model = SentenceTransformer(_semantic_model_name)

# print(semantic_embeddings)


def semantic_similarity(s1, s2, semantic_model = _semantic_model):
    semantic_embeddings = semantic_model.encode([s1, s2])
    return float(util.dot_score(semantic_embeddings[0], semantic_embeddings[1]))


semantic_similarity("This patient is in very bad health condition", "The patient is ill.")




