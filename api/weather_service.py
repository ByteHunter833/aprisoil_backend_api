from __future__ import annotations

from typing import Any

import requests
from django.conf import settings


WEATHER_URL = "https://api.openweathermap.org/data/2.5"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
FORECAST_POINTS_5_DAYS = 40
FORECAST_POINTS_24_HOURS = 8


def _fallback_weather() -> dict[str, Any]:
    return {
        "temperature": 25.0,
        "humidity": 60.0,
        "wind_speed": 5.0,
        "description": "данные недоступны",
        "rain_mm": 0.0,
    }


def get_current_weather(lat: float, lon: float) -> dict[str, Any]:
    api_key = settings.OPENWEATHER_API_KEY
    if not api_key:
        return _fallback_weather()

    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "lang": "ru",
    }
    try:
        response = requests.get(f"{WEATHER_URL}/weather", params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": float(data["main"]["temp"]),
            "humidity": float(data["main"]["humidity"]),
            "wind_speed": float(data["wind"]["speed"]),
            "description": str(data["weather"][0]["description"]),
            "rain_mm": float(data.get("rain", {}).get("1h", 0.0)),
        }
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return _fallback_weather()


def get_weather_forecast(lat: float, lon: float) -> list[dict[str, Any]]:
    api_key = settings.OPENWEATHER_API_KEY
    if not api_key:
        return []

    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "lang": "ru",
        "cnt": FORECAST_POINTS_5_DAYS,
    }
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=5)
        response.raise_for_status()
        items = response.json().get("list", [])
        forecast = []
        for item in items:
            forecast.append(
                {
                    "date": item["dt_txt"],
                    "temp": float(item["main"]["temp"]),
                    "humidity": float(item["main"]["humidity"]),
                    "rain": float(item.get("rain", {}).get("3h", 0.0)),
                    "wind_speed": float(item["wind"]["speed"]),
                    "description": str(item["weather"][0]["description"]),
                }
            )
        return forecast
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return []


def check_disaster_risk(lat: float, lon: float) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    weather = get_current_weather(lat, lon)
    forecast = get_weather_forecast(lat, lon)

    if weather["humidity"] > 80 and weather["wind_speed"] > 10:
        alerts.append(
            {
                "type": "hail",
                "level": "high",
                "message": (
                    f'Высокий риск града: влажность {weather["humidity"]}%, '
                    f'ветер {weather["wind_speed"]} м/с'
                ),
            }
        )

    next_24_hours = forecast[:FORECAST_POINTS_24_HOURS]
    total_rain_24h = sum(float(item.get("rain", 0.0)) for item in next_24_hours)
    if total_rain_24h > 30:
        alerts.append(
            {
                "type": "flood",
                "level": "high",
                "message": f"Риск селя: ожидается {round(total_rain_24h, 1)}мм осадков за 24 часа",
            }
        )

    return alerts
