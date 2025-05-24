import tkinter as tk
from tkinter import Scrollbar, Text
import random
import json
import pickle
import numpy as np
import spacy
from keras.models import load_model

# Load spaCy model and chatbot data
nlp = spacy.load("en_core_web_sm")
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Clean and tokenize input
def clean_up_sentence(sentence):
    doc = nlp(sentence)
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]

# Bag of words creation
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# Get chatbot response
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understood that. Can you try rephrasing?"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'].lower() == tag.lower():
            return random.choice(intent['responses'])
    return "Sorry, I don't have an answer for that yet."

# GUI message handler
def send(event=None):
    msg = user_input.get()
    if msg.strip() != "":
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + msg + "\n")
        chat_log.see(tk.END)

        ints = predict_class(msg)
        res = get_response(ints, intents)

        chat_log.insert(tk.END, "Bot: " + res + "\n\n")
        chat_log.config(state=tk.DISABLED)
        user_input.delete(0, tk.END)

# Build the chat window
window = tk.Tk()
window.title("Labour Rights Chatbot")
window.state("zoomed")  # Full screen
window.resizable(width=True, height=True)

# Layout configuration
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

frame = tk.Frame(window)
frame.grid(sticky="nsew", padx=10, pady=10)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

# Chat log
chat_log = Text(frame, bd=1, bg="white", font="Arial", wrap="word", state=tk.DISABLED)
chat_log.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 5))

# Scrollbar
scrollbar = Scrollbar(frame, command=chat_log.yview)
scrollbar.grid(row=0, column=2, sticky='ns')
chat_log['yscrollcommand'] = scrollbar.set

# User input
user_input = tk.Entry(frame, bd=0, bg="lightgray", font="Arial")
user_input.grid(row=1, column=0, sticky="ew", pady=10, ipady=6)
user_input.bind("<Return>", send)  # Enter key triggers send

# Send button
send_button = tk.Button(frame, text="Send", bd=0, bg="#4CAF50", fg='white', command=send)
send_button.grid(row=1, column=1, sticky="ew", pady=10)

frame.grid_columnconfigure(0, weight=5)
frame.grid_columnconfigure(1, weight=1)

window.mainloop()
