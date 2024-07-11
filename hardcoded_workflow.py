import copy
from NER_as_tool import extract_medication_and_symptom

def hardcoded(bot_input, state, par,local_case_search):
    state_new = copy.deepcopy(state)


    if state["step"] == "textual_conversation":
        output = "Please describe your symptom:"

        state_new["step"] = "retrieval"

        state_new["current_subiter"] = 0


    elif state["step"] == "retrieval":
        #get symptoms and medications unclassified using hardcoded search

        search_result = local_case_search(bot_input + ',' + ','.join(list(state['symptoms']['True'])))

        # print(search_result)
        #NER hardcoded
        extracted = extract_medication_and_symptom(search_result)
        



        state['current_retrieved_results'] = extracted

        state_new['current_retrieved_results'] = extracted
        output = f"""Related symptoms are: {','.join(extracted['symptoms'])}
        Related medications are: {','.join(extracted['medication'])}"""


        #get to next branch
        new_step = "confirm_symptoms"
        state_new["step"] = new_step


        if len(extracted['symptoms']) > 0:
            output  += f"""Do you have the symptom {state['current_retrieved_results']['symptoms'][0]}?
        Answer yes(1) no(2) or unsure(3)"""
        else:
            #no extracted symptom, go to another class
            state_new['step'] = 'confirm_medication'
            return hardcoded(bot_input, state_new, par)
            
        state_new['current_subiter'] = 0


    elif state['step']=="confirm_symptoms":
        l = len(state['current_retrieved_results']['symptoms'])
        if l == 0:
            state_new['step'] = "confirm_medications"
            return hardcoded(bot_input, state_new, par)
        
        if state['current_subiter']>0:
            #get bot_input as answer to previous

            state_new['symptoms'][hard_tf_parser(bot_input)].add(state['current_retrieved_results']['symptoms'][state['current_subiter']])

        if state['current_subiter']==l-1:
            new_step = "confirm_medication"
            state_new["step"] = new_step
            state_new['current_subiter'] = 0
            # ask medication branch

                    


            if len(state['current_retrieved_results']['medication']) > 0:
                output  = f"""Do you have the medication {state['current_retrieved_results']['symptoms'][0]}?
            Answer yes(1) no(2) or unsure(3)"""
            else:
                #no extracted symptom, go to another class
                state_new['step'] = 'confirm_medication'
                state_new['current_iter']+=1
                return hardcoded(bot_input, state_new, par)


        else:
            output = f"""Do you have the symptom {state['current_retrieved_results']['symptoms'][state['current_subiter']+1]}?
        Answer yes(1) no(2) or unsure(3)"""
            state_new['current_subiter'] = state['current_subiter'] + 1

    elif state['step']=='confirm_medication':
        
        l = len(state['current_retrieved_results']['medication'])

        if l == 0:
            state_new['step'] = "textual_conversation"
            state_new['current_iter'] =  state['current_iter'] + 1
            return hardcoded(bot_input, state_new, par)
        
        if state['current_subiter']>0:
            #get bot_input as answer to previous
            state_new['medication'][hard_tf_parser(bot_input)].add(state['current_retrieved_results']['medication'][state['current_subiter']])
        if state['current_subiter']==l-1:
            new_step = "textual_conversation"
            state_new["step"] = new_step
            state_new['current_subiter'] = 0
            state_new['current_iter'] =  state['current_iter'] + 1
            # ask medication branch
            output = "Based on these, do you have descriptions for the symptoms?"
            
        else:
            output = f"""Do you have the medication {state['current_retrieved_results']['medication'][state['current_subiter']+1]}?
        Answer yes(1) no(2) or unsure(3)"""
            state_new['current_subiter'] = state['current_subiter'] + 1

        

    

    return output, state_new




def hard_tf_parser(bot_in:str):
    bot_in = bot_in.lower()
    if 'yes' in bot_in or bot_in == '1':
        return 'True'
    if 'no' in bot_in or bot_in =='2' or bot_in =='0':
        return 'False'
    if 'unsure' in bot_in or 'not sure' in bot_in or bot_in=='3':
        return 'Unsure'