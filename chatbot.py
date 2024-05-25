import random
import json
import pickle
import numpy as np
import nltk
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize lemmatizer and load resources
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:/chatbot/chatbot/intents.json').read())
words = pickle.load(open('C:/chatbot/chatbot/words.pkl', 'rb'))
classes = pickle.load(open('C:/chatbot/chatbot/classes.pkl', 'rb'))

# Check if the model file exists before loading
model_path = 'C:/chatbot/chatbot/chatbot_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No file or directory found at {model_path}")

model = load_model(model_path)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
