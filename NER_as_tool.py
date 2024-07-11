# Use a pipeline as a high-level helper
from transformers import pipeline
# wrapper

pipe = pipeline("token-classification", model="d4data/biomedical-ner-all")

def extract_words(structure, text):
    words = []

    for entry in structure:
        start = entry['start']
        end = entry['end']

        # Extend start index to the left
        while start > 0 and text[start - 1] not in ' .,;!?\n' and text[start - 1] != '-':
            start -= 1

        # Extend end index to the right
        while end < len(text) and text[end] not in ' .,;!?\n' and text[end] != '-':
            end += 1

        # Extract the full word
        word = text[start:end]
        words.append(word)
        entry['full_word'] = word

    return words

def extract_symptom(data):
    symptoms = []

    for entry in data:
        entity_name = entry.get('entity', '').lower()
        if 'disease' in entity_name or 'symptom' in entity_name:
            word = entry.get('full_word', '')
            if word:
                symptoms.append(word)

    return symptoms

def extract_entities(text, sep = ';'):
    return sep.join(extract_words(pipe(text)))


def extract_medication(data):
    symptoms = []

    for entry in data:
        entity = entry.get('entity', '').lower()
        # print(entity)
        if 'medication' in entity or 'procedure' in entity:
            word = entry.get('full_word', '')
            if word:
                symptoms.append(word)

    return symptoms

# print(list(set(extract_symptoms(structure))))

def extract_symptom_comma_sep(text, pipe = pipe):
    structure = pipe(text)
    extract_words(structure,text)
    return ','.join(list(set(extract_symptom(structure))))


def extract_medication_comma_sep(text, pipe = pipe):
    structure = pipe(text)
    extract_words(structure,text)
    return ','.join(list(set(extract_medication(structure))))

def extract_medication_and_symptom(text, pipe=pipe,
    return_type='list'):
    structure = pipe(text)
    extract_words(structure,text)
    if return_type == 'list':
        return {'symptoms': (list(set(extract_symptom(structure)))),
            'medication': (list(set(extract_medication(structure))))}
    else:
        return {'symptoms': set(extract_symptom(structure)),
            'medication': set(extract_medication(structure))}

def extract_medication_and_symptom_as_text(text, pipe=pipe):
    structure= pipe(text)
    extract_words(structure,text)
    return "Symptom :" + ','.join(list(set(extract_symptom(structure))))\
    + '\n'+ "Medication: " + ','.join(list(set(extract_medication(structure))))

