#spacy is a bitch and it made me use python 3.12 instead of 3.13 due to a numpy issue.
#chat.py

#I need to make a training dataset.

import os,random,json,torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json") as json_data:
    intents = json.load(json_data)

data_dir = os.path.join(os.path.dirname(__file__))
FILE = os.path.join(data_dir, 'chatdata.pth')
data = torch.load(FILE,weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "iris-NLP"
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    words=X
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if (tag=="weather"):
                    print("The weather in where?")
                    sentence = input("You: ")
                    if sentence == "quit":
                        break
                    locationinput =tokenize(sentence)
                    print (locationinput)
                    locationexists=findcitycountry(sentence)
                    if (locationexists=="N/A"):
                        return "Something went wrong."
                    else:
                        city_weather = get_weather(locationexists)
                        if city_weather is not None:
                            return "In " + locationexists + ", the current weather is: " + city_weather
                else:
                    return random.choice(intent['responses'])
            
    return "I do not understand..."



import requests
import api
api_key=api.api_key


def get_weather(city_name):
    api_url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}".format(city_name, api_key)
    response = requests.get(api_url)
    response_dict = response.json()

    weather = response_dict["weather"][0]["description"]
    
    if response.status_code == 200:
        return weather
    else:
        print('[!] HTTP {0} calling [{1}]'.format(response.status_code, api_url))
        return None


import geonamescache
gc=geonamescache.GeonamesCache()


def gen_dict_extract(var, key):
    if isinstance(var, dict):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, (dict, list)):
                yield from gen_dict_extract(v, key)
    elif isinstance(var, list):
        for d in var:
            yield from gen_dict_extract(d, key)

def findcitycountry(input):
    
    countries=gc.get_countries()
    cities=gc.get_cities()
    cities = [*gen_dict_extract(cities, 'name')]
    countries = [*gen_dict_extract(countries, 'name')]
    if (isinstance(input,str)):
        if (input.upper() in (city.upper() for city in cities) or input.upper() in (country.upper() for country in countries)):
            return input
    else:
        for word in input:
            if (input.upper() in (city.upper() for city in cities) or input.upper() in (country.upper() for country in countries)):
                return input
    return "N/A"





if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        resp = get_response(sentence)
        print(resp)

