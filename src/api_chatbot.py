import openai
import os
import itertools
from src.common_utils import client
from llama_cpp import Llama
import time
import json
from src.openai_query import generate_response, query_typesense

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
LLAMA2_MODEL_PATH = os.getenv('LLAMA2_MODEL_PATH')


def function_call_with_retry(question, max_retries=5):
    retries = 0

    while retries < max_retries:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                temperature=0,
                messages=[
                    {"role": "system", "content": "you are an upstox chat assistant"},
                    {"role": "user", "content": question}
                ],
                functions=[
                    {
                        "name": "harmful_cs_check",
                        "description": "checks if the input contains harmful words and if they need human support.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "harmful": {
                                    "type": "string",
                                    "description": "If input contains abusive content or it contains negative sentiment towards upstox platform then return True else False.",
                                    "enum": ["True", "False"]
                                },
                                "human_support": {
                                    "type": "string",
                                    "description": "If input mentions they need to speak to an agent or human then return True else False.",
                                    "enum": ["True", "False"]
                                },
                            },
                            "required": ["harmful", "human_support"],
                        },
                    }
                ],
                function_call={"name": "harmful_cs_check"},
            )

            if completion['choices'][0]['finish_reason'] == "stop":
                val = completion['choices'][0]['message']['function_call']['arguments']
            else:
                val = completion['choices'][0]['finish_reason']
            
            return val  # Return the result if successful

        except Exception as e:
            retries += 1
            print(f"Error in API call (Attempt {retries}/{max_retries}): {str(e)}")
            if retries < max_retries:
                print("Retrying after a short delay...")
                time.sleep(2)  # Add a delay before retrying

    # If all retries fail, you can return an error message or raise an exception
    return "Max retries reached. Unable to get a response."


def generate_question(history,question,model):
    question_template = f"""
    Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
    Chat History:\"""
    {history}
    \"""
    Follow Up question: \"""
    {question}
    \"""
    Standalone question:"""

    resp = generate_response(model,question=question_template,system=".")
    return resp




def api_chatbot_raw(question,chat_history,collection_name):
    model = "openai"
    fcall = function_call_with_retry(question)
    fcall = json.loads(fcall)
    #{\n  "harmful": "False",\n  "human_support": "False"\n}
    if(fcall['harmful'] == "True"):
        resp = "This is a harmful question. Please dont use such words. (API Function Call)"
        response = {
        "question":question,
        "answer":resp,}
        return response
    elif(fcall['human_support'] == "True"):
        resp = "Connecting you to our human customer support. (API Function Call)"
        response = {
        "question":question,
        "answer":resp,}
        return response
    else:
        pass


    if(len(chat_history) < 1):
        get_question = question

    else:
        get_question = generate_question(chat_history,question,model)
    print(get_question)


    
    vallur = query_typesense(str(get_question),collection_name)
    vall = [value['text'] for value in vallur.values()]
    source_links = [value['source'] for value in vallur.values()]
    #print(vall)
    context = "| ".join(vall)
    #print(context)

    chat_template = f"""
    Context:\"""
    
    {context}
    \"""
    Question:\"
    {get_question}
    \"""
    
    Helpful Answer:"""
    resp = generate_response(model,question=chat_template,system="You are a friendly assistant. Use the following context to answer. It's ok if you don't know the answer, Just say 'I dont know.' ")
    #fut = generate_future(get_question,chat_history,resp)
    response = {
        "question":get_question,
        "answer":resp,
        "source_links":source_links,
        
    }
    return response