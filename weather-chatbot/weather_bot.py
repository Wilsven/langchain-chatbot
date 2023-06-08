import os
import requests
import dotenv

import spacy

# loads .env file with your OPEN_WEATHER_API_KEY
dotenv.load_dotenv()

OPEN_WEATHER_API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

nlp = spacy.load("en_core_web_md")


def get_weather(city_name: str) -> str:
    """Makes a GET request to API endpoint to get weather information of a city.

    Parameters
    ----------
    city_name: str
        City name.

    Returns
    -------
    weather: str
        Weather description.
    """
    API_URL = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={OPEN_WEATHER_API_KEY}"

    response = requests.get(API_URL)
    response_dict = response.json()

    if response.status_code == 200:
        weather = response_dict["weather"][0]["description"]
        return weather
    else:
        print(f"[!] HTTP {response.status_code} calling [{API_URL}]")
        return None


def chatbot(statement: str, min_similarity: float = 0.50) -> str:
    """Return a response on the weather of a city based on the question.

    Parameters
    ----------
    statement: str
        Question about the weather of a city.

    min_similarity: float
        Simialarity threshold.

    Returns
    -------
    response: str
        Chatbot response.
    """
    weather = nlp("Current weather in a city")
    statement = nlp(statement)

    print(weather.similarity(statement))

    if weather.similarity(statement) >= min_similarity:
        for ent in statement.ents:
            print(ent.text)
            if (
                ent.label_ == "GPE" or weather.similarity(statement) >= min_similarity
            ):  # GeoPolitical Entity
                city = ent.text
                break
            else:
                return "You need to tell me a city to check."

        city_weather = get_weather(city)
        if city_weather is not None:
            return "In " + city + ", the current weather is: " + city_weather
        else:
            return "Something went wrong."
    else:
        return "Sorry, I don't understand that. Please rephrase your statement."


response = chatbot("What's the weather like in Singapore today?")
print(response)
