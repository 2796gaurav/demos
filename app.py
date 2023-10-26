from flask import Flask, jsonify, request,render_template, Response
from src.scraping import scrap_main
from src.openai_query import chatbot_raw, whisper_translate_query
from src.api_chatbot import api_chatbot_raw
import pandas as pd
# creating a Flask app
app = Flask(__name__)

# Scrap
@app.route('/scrap/v1',methods=['POST'])
def scrap():
    scrap_url = request.json['scrap_url']
    id = request.json['id']
    if(id not in ["car_review","indian_tourist"]):
        result_df = scrap_main(scrap_url,id)
        result_df.to_csv(f"data/{id}.csv",index=False)
    return jsonify({"id":id})

@app.route('/display/<id>')
def display_data(id):
    try:
        data = pd.read_csv(f"data/{id}.csv")
        #data = data.head(5)
        data = data.drop(["content_vector"],axis=1)
        html_table = data.to_html(classes='table table-bordered table-hover')
        return html_table
    except Exception as e:
        return str(e)

# chat

chat_history = []
@app.route('/chat/query',methods=['POST'])
def scrap_query():
    global chat_history
    #data_url = request.json['data_url']
    collection_name = request.json['id']
    user_question = request.json['user_question']
    model = request.json['model']

    print(model)
    result = chatbot_raw(user_question,chat_history,collection_name,model)
    print(result)
    chat_history.append((result["question"], result["answer"]))


    return jsonify(result)

api_calling_chat_history = []
@app.route('/api_calling/query',methods=['POST'])
def api_calling_query():
    global api_calling_chat_history
    collection_name = request.json['id']
    user_question = request.json['user_question']
    
    result = api_chatbot_raw(user_question,api_calling_chat_history,collection_name)
    api_calling_chat_history.append((result["question"], result["answer"]))


    return jsonify(result)


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return 'No audio file provided', 400

    audio_file = request.files['audio']
    path = 'audio/audio.wav'
    audio_file.save(path)
    text = whisper_translate_query(path)
    print(text)
    return jsonify({"translated_text":text})


@app.route('/scrap')
def scrap_html():
    return render_template("scrap.html")


@app.route('/chat')
def chat_html():
    return render_template("chatbot.html")

@app.route('/api_calling')
def api_calling_html():
    return render_template("api_calling.html")

@app.route('/')
def main():
    return render_template("index.html")



@app.route('/health')
def health():
    resp = jsonify(ok=True)
    resp.status_code = 200
    return resp

app.run(port=5003,debug=True)
