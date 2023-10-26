from dotenv import load_dotenv
import openai
import time
import os
import typesense
from ast import literal_eval
import random
import string
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
TYPESENSE_HOST = os.getenv('TYPESENSE_HOST')
TYPESENSE_PORT = os.getenv('TYPESENSE_PORT')
TYPESENSE_PROTOCOL = os.getenv('TYPESENSE_PROTOCOL')
TYPESENSE_API_KEY = os.getenv('TYPESENSE_API_KEY')



def generate_random_id(length):
    characters = string.ascii_letters + string.digits  # Letters and digits
    return ''.join(random.choice(characters) for _ in range(length))

MAX_RETRIES = 5  # Maximum number of retries
RETRY_DELAY = 5  # Delay between retries in seconds

def get_embedding_with_retry(query):
    retries = 0

    while retries < MAX_RETRIES:
        try:
            embedded_query = openai.Embedding.create(
                input=query,
                model=EMBEDDING_MODEL,
            )['data'][0]['embedding']

            return embedded_query  # Return the result if successful

        except openai.error.APIError as e:
            if e.status == 504:  # Check if the error is a timeout error
                print(f"Timeout error occurred. Retrying... ({retries+1}/{MAX_RETRIES})")
                retries += 1
                time.sleep(RETRY_DELAY)
            else:
                raise e  # Raise the exception if it's not a timeout error

    # If maximum retries exceeded, raise an exception or return a default value
    raise Exception("Maximum retries exceeded. Unable to get the desired output.")

# Function to split text into rows while ensuring word boundaries and character limit
def split_text(text, limit):
    words = text.split()
    rows = []
    current_row = ""
    
    for word in words:
        if len(current_row) + len(word) + 1 <= limit:
            if current_row:
                current_row += " " + word
            else:
                current_row = word
        else:
            rows.append(current_row)
            current_row = word
    
    if current_row:
        rows.append(current_row)
    
    return rows


def client():
    node = {
        "host": TYPESENSE_HOST,
        "port": TYPESENSE_PORT,
        "protocol": TYPESENSE_PROTOCOL
    }
    typesense_client = typesense.Client(
        {
            "nodes": [node],
            "api_key": TYPESENSE_API_KEY,
            "connection_timeout_seconds": 2
        }
    )
    return typesense_client


## create collection
def create_collection(collection_name,cv_length,delete_collection=True):
    typesense_client = client()
    if(delete_collection):
        try:
            typesense_client.collections[collection_name].delete()
        except Exception as e:
            pass

    # Create a new collection
    schema = {
        "name": collection_name,
        "fields": [
            {
                "name": "content_vector",
                "type": "float[]",
                "num_dim": cv_length
            },
            {
                "name": "url",
                "type": "string"
            },
            {
            "name": "content",
            "type": "string"
            },
            {
                "name": "title",
                "type": "string"
            },
            {
                "name": "header",
                "type": "string"
            }
        ]
    }

    create_response = typesense_client.collections.create(schema)

    print(f"Created new collection {collection_name}")
    return True

def push_to_typesense(df,collection_name):
    typesense_client = client()
    #df['content_vector'] = df['content_vector'].apply(lambda x: literal_eval(x))
    df = df.dropna()
    cv_length = len(df['content_vector'][0])

    # create collection
    create_collection(collection_name,cv_length,delete_collection=True)

    document_counter = 0
    documents_batch = []

    for k,v in df.iterrows():
        # Create a document with the vector data
        document = {
            # "id":v["id"],
            "content_vector":v["content_vector"],
            "url": v["url"],
            "content": v["content"],
            "title": v["title"],
            "header": v["header"],
        }
        
        documents_batch.append(document)
        document_counter = document_counter + 1

        # Upsert a batch of 100 documents
        if document_counter % 100 == 0 or document_counter == len(df):
            response = typesense_client.collections[collection_name].documents.import_(documents_batch)
            #client.collections['companies'].documents.create(document)
            # print(response)

            documents_batch = []
            print(f"Processed {document_counter} / {len(df)} ")



