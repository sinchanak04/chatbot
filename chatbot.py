import random
import json
import pickle
import numpy as np
import spacy


from keras.models import load_model

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load intents and model data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


# Tokenization and Lemmatization using spaCy
def clean_up_sentence(sentence):
    doc = nlp(sentence)
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]


# Create bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Predict the intent class
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


# Get response from predicted intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understood that. Can you try rephrasing?"

    predicted_tag = intents_list[0]['intent']
    
    for intent in intents_json['intents']:
        if intent['tag'].lower() == predicted_tag.lower():
            if intent.get('responses'):
                return random.choice(intent['responses'])
            else:
                return "Hmm... I couldn't find a proper response for that."
    
    return f"I'm not trained to answer that yet. (intent: {predicted_tag})"


# Start the chatbot
print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
