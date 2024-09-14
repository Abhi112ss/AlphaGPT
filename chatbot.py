import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random


lemmatizer = WordNetLemmatizer()

# Load the intents file with UTF-8 encoding
with open(r"D:\FIles\IntelliJ IDEA Community Edition 2023.2.5\PROJECT A.L.P.H.A\intents.json", encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Tokenizing and processing intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatizing, removing duplicates, and sorting
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Print summary
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words")

# Save words and classes for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Preparing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # Create output_row where current tag is 1, others are 0
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)

# Fix: Convert training list to NumPy array
training = np.array(training, dtype=object)

# Separate the features (X) and labels (Y)
train_x = np.array([i[0] for i in training])  # Features (bag of words)
train_y = np.array([i[1] for i in training])  # Labels (one-hot encoded output)

# Print sizes of train_x and train_y for verification
print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

# Building the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot-model.h5', hist)

print("Model created and saved!")


model = load_model('chatbot-model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Helper functions for cleaning input and creating the bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Predict the class of a given sentence
def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get a response based on the predicted intent
def get_response(intent, intents_json):
    tag = intent['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I don't understand."

# Chat with the bot
print("Chatbot is running! Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() in ['quit', 'exit', 'bye']:
        print("Chatbot: Goodbye!")
        break

    predicted_intents = predict_class(message, model)
    if predicted_intents:
        response = get_response(predicted_intents[0], intents)
        print("Chatbot:", response)
    else:
        print("Chatbot: Sorry, I don't understand.")