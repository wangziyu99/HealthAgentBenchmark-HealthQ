import copy
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils import search_cases, format_retrieval_results, extract_medication_and_symptom, hardcoded

# Define prompt templates
initial_template = """
The patient states: {bot_input}.
Based on the following relevant cases:
{formatted_retrieval_results}
Ask a better question to help diagnose any health issues. This may include asking whether the patient relates to symptoms, medications, habits or professions of the searched relevant cases.
The new question should be concise and specific. Please provide only the new question in your response.
"""

reflection_template = """
The patient states: {bot_input}.
The initial question generated was: {initial_question}
Based on the following relevant cases:
{formatted_retrieval_results}
Please reflect on the initial question and assess its quality. Consider the following:
- Does the question adequately address the patient's concerns?
- Does the question make proper use of the information from the relevant cases?
- Is there any additional information from the relevant cases that could be incorporated to improve the question?
If the initial question is satisfactory, please output the satisfactory question as is.
If the initial question can be improved, please generate an updated question that better addresses the patient's concerns and incorporates relevant information from the search results.
Reply with the final question only.
"""

intermediate_reflection_template = """
The patient states: {bot_input}.
Previous reflections:
{previous_reflections}
Based on the following relevant cases:
{formatted_retrieval_results}
Please reflect on the information provided and consider the following:
- What symptoms or conditions are related to the patient's concerns?
- What additional information from the relevant cases could be useful in diagnosing the patient's health issues?
- Are there any specific questions that could help gather more relevant information from the patient?
Provide your reflections and thoughts without formulating a final question. Focus on identifying relevant information and potential areas to explore further.
Reply with the final question only.
Final question:
"""

final_reflection_template = """
The patient states: {bot_input}.
Previous reflections:
{previous_reflections}
Based on the following relevant cases:
{formatted_retrieval_results}
Considering the previous reflections and the relevant cases, please formulate a concise and specific question to help diagnose the patient's health issues. The question should incorporate the most relevant information and aim to gather additional details from the patient to aid in the diagnosis.
Reply with the final question only.
"""

# Initialize prompt templates
initial_prompt = PromptTemplate(input_variables=["bot_input", "formatted_retrieval_results"], template=initial_template)
reflection_prompt = PromptTemplate(input_variables=["bot_input", "initial_question", "formatted_retrieval_results"], template=reflection_template)
intermediate_reflection_prompt = PromptTemplate(input_variables=["bot_input", "previous_reflections", "formatted_retrieval_results"], template=intermediate_reflection_template)
final_reflection_prompt = PromptTemplate(input_variables=["bot_input", "previous_reflections", "formatted_retrieval_results"], template=final_reflection_template)

# RAG workflows
def RAG_default_workflow(bot_input, state, par, qa_chain):
    state_new = copy.deepcopy(state)
    query = f"""
    The patient states: {bot_input}.
    Based on relevant cases, ask for better statement to help diagnose any health issues.
    Ask a better question to help diagnose any health issues. This may include asking whether the patient relates to symptoms, medications, habits or professions of the searched relevant cases.
    The new question should be concise and specific. Please provide only the new question in your response."""
    result = qa_chain(query)
    output = result['result']
    state_new["current_iter"] += 1
    return output, state_new

def RAG_workflow(bot_input, state, par, llm):
    state_new = copy.deepcopy(state)
    retrieval_results = search_cases(bot_input)
    formatted_retrieval_results = format_retrieval_results(retrieval_results)
    query = initial_prompt.format(bot_input=bot_input, formatted_retrieval_results=formatted_retrieval_results)
    new_question = llm.invoke(query).content
    state_new.update({"new_question": new_question.strip(), "retrieval_results": retrieval_results})
    return new_question.strip(), state_new

def RAG_workflow_reflection(bot_input, state, par, llm):
    state_new = copy.deepcopy(state)
    retrieval_results = search_cases(bot_input)
    formatted_retrieval_results = format_retrieval_results(retrieval_results)
    initial_query = initial_prompt.format(bot_input=bot_input, formatted_retrieval_results=formatted_retrieval_results)
    initial_question = llm.invoke(initial_query).content
    reflection_query = reflection_prompt.format(bot_input=bot_input, initial_question=initial_question, formatted_retrieval_results=formatted_retrieval_results)
    reflection_output = llm.invoke(reflection_query).content
    state_new.update({"question": reflection_output, "retrieval_results": retrieval_results})
    return reflection_output, state_new

def RAG_workflow_reflection_cot(bot_input, state, par, local_llm, num_reflections=3):
    state_new = copy.deepcopy(state)
    retrieval_results = search_cases(bot_input)
    formatted_retrieval_results = format_retrieval_results(retrieval_results)
    initial_query = initial_prompt.format(bot_input=bot_input, formatted_retrieval_results=formatted_retrieval_results)
    current_output = local_llm.invoke(initial_query).content
    previous_reflections = [current_output]
    for i in range(num_reflections - 1):
        if i < num_reflections - 2:
            reflection_query = intermediate_reflection_prompt.format(bot_input=bot_input, previous_reflections="\n".join(previous_reflections), formatted_retrieval_results=formatted_retrieval_results)
        else:
            reflection_query = final_reflection_prompt.format(bot_input=bot_input, previous_reflections="\n".join(previous_reflections), formatted_retrieval_results=formatted_retrieval_results)
        reflection_output = local_llm.invoke(reflection_query).content
        previous_reflections.append(reflection_output)
    state_new.update({"question": previous_reflections[-1], "retrieval_results": retrieval_results, "reflections": previous_reflections})
    return previous_reflections[-1], state_new

def RAG_workflow_reflection_cot_sc(bot_input, state, par, local_llm, noisy_llm, num_cots=3, num_reflections=3):
    cot_outputs = []
    new_states = []
    for _ in range(num_cots):
        cot_output, new_state = RAG_workflow_reflection_cot(bot_input, state, par, noisy_llm, num_reflections)
        cot_outputs.append(cot_output)
        new_states.append(new_state)
    
    # Use the local LLM to make the final decision
    cot_results = '\n'.join([f"CoT {i+1}:\n{cot_outputs[i]}\nReflections:\n" + '\n'.join(new_states[i]['reflections']) for i in range(len(cot_outputs))])
    
    final_decision_prompt = f"""
    Given the following {num_cots} different outputs from the Chain of Thought (CoT) process:
    {cot_results}
    Please analyze the outputs and reflections, and determine the most appropriate final question to ask the patient based on the collective insights gathered from the CoT process. Consider the relevance, specificity, and potential usefulness of each question in aiding the diagnosis of the patient's health issues.
    Reply with the final question only.
    Final question:"""
    final_question = local_llm.invoke(final_decision_prompt).content
    state_new = copy.deepcopy(state)
    state_new.update({"question": final_question, "cot_outputs": cot_outputs})
    return final_question, state_new

# ReAct workflow
def ReAct_workflow(bot_input, state, par, qa_chain, llm, agent):
    state_new = copy.deepcopy(state)

    if state["step"] == "textual_conversation":
        query = f"""
        The patient states: {bot_input}.
        Based on relevant cases,        
        Ask a better question to help diagnose any health issues. This may include asking whether the patient relates to symptoms, medications, habits or professions of the searched relevant cases.
        The new question should be concise and specific. Please provide only the new question in your response."""
        new_questions = agent.run(f"""
        The patient states: {bot_input}
        What are similar symptoms and relevant behavior, profession, habits or treatments related to this symptom that the doctor should confirm with the patient?
        Phrase a concise question in doctor's voice""")
        output = new_questions
        state_new["current_subiter"] = 0
        state_new["current_iter"] += 1

    elif state["step"] == "retrieval":
        search_result = local_case_search(bot_input + ',' + ','.join(list(state['symptoms']['True'])))
        extracted = extract_medication_and_symptom(search_result)
        state['current_retrieved_results'] = extracted
        state_new['current_retrieved_results'] = extracted
        output = f"""Related symptoms are: {','.join(extracted['symptoms'])}
        Related medications are: {','.join(extracted['medication'])}"""
        new_step = "confirm_symptoms"
        state_new["step"] = new_step

        if len(extracted['symptoms']) > 0:
            output += f"""Do you have the symptom {state['current_retrieved_results']['symptoms'][0]}?
        Answer yes(1) no(2) or unsure(3)"""
        else:
            state_new['step'] = 'confirm_medication'
            return hardcoded(bot_input, state_new, par)
        state_new['current_subiter'] = 0

    elif state['step'] == "confirm_symptoms":
        l = len(state['current_retrieved_results']['symptoms'])
        if l == 0:
            state_new['step'] = "confirm_medications"
            return hardcoded(bot_input, state_new, par)
        if state['current_subiter'] > 0:
            state_new['symptoms'][hard_tf_parser(bot_input)].add(state['current_retrieved_results']['symptoms'][state['current_subiter']])
        if state['current_subiter'] == l - 1:
            new_step = "confirm_medication"
            state_new["step"] = new_step
            state_new['current_subiter'] = 0
            if len(state['current_retrieved_results']['medication']) > 0:
                output = f"""Do you have the medication {state['current_retrieved_results']['medication'][0]}?
            Answer yes(1) no(2) or unsure(3)"""
            else:
                state_new['step'] = 'confirm_medication'
                return hardcoded(bot_input, state_new, par)
        else:
            output = f"""Do you have the symptom {state['current_retrieved_results']['symptoms'][state['current_subiter'] + 1]}?
        Answer yes(1) no(2) or unsure(3)"""
            state_new['current_subiter'] = state['current_subiter'] + 1

    elif state['step'] == 'confirm_medication':
        l = len(state['current_retrieved_results']['medication'])
        if l == 0:
            state_new['step'] = "textual_conversation"
            state_new['current_iter'] = state['current_iter'] + 1
            return hardcoded(bot_input, state_new, par)
        if state['current_subiter'] > 0:
            state_new['medication'][hard_tf_parser(bot_input)].add(state['current_retrieved_results']['medication'][state['current_subiter']])
        if state['current_subiter'] == l - 1:
            new_step = "textual_conversation"
            state_new["step"] = new_step
            state_new['current_subiter'] = 0
            state_new['current_iter'] = state['current_iter'] + 1
            output = "Based on these, do you have descriptions for the symptoms?"
        else:
            output = f"""Do you have the medication {state['current_retrieved_results']['medication'][state['current_subiter'] + 1]}?
        Answer yes(1) no(2) or unsure(3)"""
            state_new['current_subiter'] = state['current_subiter'] + 1

    return output, state_new
