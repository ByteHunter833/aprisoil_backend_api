import os 
from dotenv import load_dotenv
load_dotenv()
open_weather_api_key = os.getenv('OPENWEATHER_API_KEY')
print(f"OPENWEATHER_API_KEY: {open_weather_api_key}")