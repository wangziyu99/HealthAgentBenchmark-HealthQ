"""
Wrapper of medialpy, a dictionary for medical term abbreviations
"""
# !pip install medialpy

import medialpy

def medial_search(query):
        
    if medialpy.exists(query):
        return medialpy.search(query).meaning
    else:
        return "Not Found"
    