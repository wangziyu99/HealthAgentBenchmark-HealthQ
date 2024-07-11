from NER_as_tool import extract_medication_and_symptom

def compare_sets(A, B):
    A, B = set(A), set(B)
    # Compute the common elements (intersection)
    intersection = A.intersection(B)
    num_common_elements = len(intersection)

    # Compute elements in A that are not in B (left difference)
    left_difference = A.difference(B)
    num_left_difference = len(left_difference)

    # Compute elements in B that are not in A (right difference)
    right_difference = B.difference(A)
    num_right_difference = len(right_difference)

    # Calculate the additional terms
    inter_left_ratio = num_common_elements / (num_common_elements + num_left_difference) if num_common_elements + num_left_difference > 0 else 0
    inter_right_ratio = num_common_elements / (num_common_elements + num_right_difference) if num_common_elements + num_right_difference > 0 else 0
    inter_all_ratio = num_common_elements / (num_common_elements + num_left_difference + num_right_difference) if num_common_elements + num_left_difference + num_right_difference > 0 else 0

    # Create a dictionary to store the results
    result_dict = {
        "intersection_count": num_common_elements,
        "left_difference_count": num_left_difference,
        "right_difference_count": num_right_difference,
        "inter_left_ratio": inter_left_ratio,
        "inter_right_ratio": inter_right_ratio,
        "inter_all_ratio": inter_all_ratio
    }

    return result_dict

def add_total_set(dict_of_sets):
    total_set = set()
    for set_value in dict_of_sets.values():
        total_set |= set_value
    dict_of_sets["total"] = total_set
    return dict_of_sets

def get_NER_set_diff(hypothesis, reference):
    hypo_set = add_total_set(extract_medication_and_symptom(hypothesis,return_type='set'))
    reference_set = add_total_set(extract_medication_and_symptom(reference, return_type='set'))


    result_dict = {entity_type: compare_sets(hypo_set[entity_type],reference_set[entity_type]) for entity_type in hypo_set.keys()}
    return result_dict


from _metric import *
from rouge import Rouge 
rouge = Rouge()

rouge_scoring = lambda hypothesis, reference: rouge.get_scores(hypothesis, reference)[0]




""" search tasks only require f(string)"""

_search_dict = {"sensitivity_search" : sensitivity_search,
                "name_search" : name_search,
                "med_ner_search" : med_ner_search,
                # "privacy_ner_search" : privacy_ner_search,
                "textblob_vector" : textblob_vector,
               }

""" compare task f(string, string_to_compare"""

_similarity_dict = {"semantic_similarity" : semantic_similarity,
                    "word_similarity" : word_similarity_all,

                   }

_medagent_dict = {"rouge": rouge_scoring,
                    'NER2NER': get_NER_set_diff}
import gc
import pandas as pd
def get_score_dict(s1, s2 = None, s1_tasks = _search_dict, s2_tasks = _similarity_dict,
                  gc_collect =  "none",
                  flatten = False):
    result = {}
    assert s1 is not None
    #if len(s1)*len(s2)==0:
    	
    if gc_collect == "none":

        for k, v in s1_tasks.items():
            result[k] = v(s1)
        gc.collect()
        if s2 is not None:
            for k, v in s2_tasks.items():
                result[k] = v(s1, s2)
    elif gc_collect == "outer":

        for k, v in s1_tasks.items():
            result[k] = v(s1)
        gc.collect()
        if s2 is not None:
            for k, v in s2_tasks.items():
                result[k] = v(s1, s2)
    elif gc_collect == "inner":
        for k, v in s1_tasks.items():
            gc.collect()
            result[k] = v(s1)
        
        if s2 is not None:
            gc.collect()
            for k, v in s2_tasks.items():
                result[k] = v(s1, s2)
    if flatten:
        result = pd.json_normalize(result, sep='_')
    return result






