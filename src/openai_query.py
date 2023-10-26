import openai
import os
import itertools
from src.common_utils import client
from llama_cpp import Llama

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
LLAMA2_MODEL_PATH = os.getenv('LLAMA2_MODEL_PATH')


def generate_response(model,question,system="You are a helpful assistant."):

    if(model == "openai"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question}
            ],
            temperature=0
            # stream=True
        )
        return response['choices'][0]['message']['content']
    elif(model == "Llama-2"):
        print(question)
        LLM = Llama(model_path=LLAMA2_MODEL_PATH)
        output = LLM(question)
        return output['choices'][0]['text']
    else:
        return "Sorry, I don't know how to answer that."


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


def query_typesense(query,collection_name, field='title', top_k=10):
    typesense_client = client()
    print("hi")
    print(collection_name)

    # Creates embedding vector from user query
    # openai.api_key = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    embedded_query = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )['data'][0]['embedding']
    

    typesense_results = typesense_client.multi_search.perform({
        "searches": [{
            "q": "*",
            "collection": collection_name,
            "vector_query": f"content_vector:([{','.join(str(v) for v in embedded_query)}], k:{top_k})"
        }]
    }, {})

    hits = typesense_results['results'][0]['hits']
    res_dict = {}
    res_list = []
    threshold = 0.90  # Threshold value

    for c, hit in itertools.islice(enumerate(hits), top_k):
        vector_distance = hit['vector_distance']
        if c <= 2:
            values_append = {
                #  'answer': hit['document']['answer'],
                #  'vector_distance': hit['vector_distance'],
                 'text': hit['document']['content'],
                 'source': hit['document']['url']
             }
            res_list.append(hit['document']['content'])
            res_dict[c] = values_append
    res_list = res_list[:3]
    #print(res_list)
    return res_dict


def chatbot_raw(question,chat_history,collection_name,model):
    if(len(chat_history) < 1):
        get_question = question
    elif(model == 'Llama-2'):
        get_question = question
    else:
        get_question = generate_question(chat_history,question,model)
    print(get_question)


    
    vallur = query_typesense(str(get_question),collection_name)
    vall = [value['text'] for value in vallur.values()]
    source_links = [value['source'] for value in vallur.values()]
    

    #print(vall)
    context = "| ".join(vall)
    if(model == 'Llama-2'):
        context = context[:200]
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

def whisper_translate_query(path):
    audio_file = open(path, "rb") 
    transcript = openai.Audio.transcribe("whisper-1", audio_file) 
    return transcript['text']
