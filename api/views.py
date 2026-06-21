from __future__ import annotations

import json
from typing import Any, Optional

from django.conf import settings
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from ml.predictor import check_water_needed, predict_moisture_7days

from .models import Alert, Field, Prediction, SoilReading
from .weather_service import check_disaster_risk, get_current_weather


def _error(message: str, status: int = 400, code: str = "bad_request") -> JsonResponse:
    return JsonResponse({"error": message, "code": code}, status=status)


def _method_not_allowed(method: str) -> JsonResponse:
    return _error(f"Method {method} is not allowed for this endpoint.", status=405, code="method_not_allowed")


def _parse_json_body(request: HttpRequest) -> tuple[Optional[dict[str, Any]], Optional[JsonResponse]]:
    if not request.body:
        return {}, None
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, _error("Invalid JSON body.", status=400, code="invalid_json")
    if not isinstance(payload, dict):
        return None, _error("JSON body must be an object.", status=400, code="invalid_json")
    return payload, None


def _get_field(field_id: int) -> tuple[Optional[Field], Optional[JsonResponse]]:
    try:
        return Field.objects.get(id=field_id), None
    except Field.DoesNotExist:
        return None, _error("Field not found.", status=404, code="field_not_found")


def _read_string(
    payload: dict[str, Any],
    key: str,
    *,
    default: Optional[str] = None,
    required: bool = False,
    max_length: int = 100,
) -> tuple[Optional[str], Optional[str]]:
    value = payload.get(key, default)
    if value is None:
        if required:
            return None, f"{key} is required."
        return default, None
    if not isinstance(value, str):
        return None, f"{key} must be a string."
    value = value.strip()
    if required and not value:
        return None, f"{key} is required."
    if len(value) > max_length:
        return None, f"{key} must be at most {max_length} characters."
    return value, None


def _read_number(
    payload: dict[str, Any],
    key: str,
    *,
    default: Optional[float] = None,
    required: bool = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> tuple[Optional[float], Optional[str]]:
    value = payload.get(key, default)
    if value is None:
        if required:
            return None, f"{key} is required."
        return default, None
    if isinstance(value, bool):
        return None, f"{key} must be a number."
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"{key} must be a number."
    if minimum is not None and number < minimum:
        return None, f"{key} must be greater than or equal to {minimum}."
    if maximum is not None and number > maximum:
        return None, f"{key} must be less than or equal to {maximum}."
    return number, None


def _validation_error(message: Optional[str]) -> Optional[JsonResponse]:
    if message is None:
        return None
    return _error(message, status=400, code="validation_error")


def _field_payload(field: Field) -> dict[str, Any]:
    return {
        "id": field.id,
        "name": field.name,
        "latitude": field.latitude,
        "longitude": field.longitude,
        "area_hectares": field.area_hectares,
        "crop_type": field.crop_type,
    }


def _ensure_development_field() -> None:
    if not settings.DEBUG or Field.objects.exists():
        return
    Field.objects.get_or_create(
        name="Тестовый сектор ApriSoil",
        defaults={
            "latitude": 42.8746,
            "longitude": 74.5698,
            "area_hectares": 12.5,
            "crop_type": "apricot",
        },
    )


@csrf_exempt
def field_list(request: HttpRequest) -> JsonResponse:
    if request.method != "GET":
        return _method_not_allowed(request.method)

    _ensure_development_field()
    fields = [_field_payload(field) for field in Field.objects.all().order_by("id")]
    return JsonResponse({"fields": fields})


@csrf_exempt
def field_create(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return _method_not_allowed(request.method)

    payload, error_response = _parse_json_body(request)
    if error_response is not None:
        return error_response
    assert payload is not None

    name, error = _read_string(payload, "name", required=True, max_length=100)
    if (response := _validation_error(error)) is not None:
        return response
    latitude, error = _read_number(payload, "latitude", required=True, minimum=-90, maximum=90)
    if (response := _validation_error(error)) is not None:
        return response
    longitude, error = _read_number(payload, "longitude", required=True, minimum=-180, maximum=180)
    if (response := _validation_error(error)) is not None:
        return response
    area, error = _read_number(payload, "area_hectares", default=1.0, minimum=0.01)
    if (response := _validation_error(error)) is not None:
        return response
    crop_type, error = _read_string(payload, "crop_type", default="apricot", max_length=50)
    if (response := _validation_error(error)) is not None:
        return response

    field = Field.objects.create(
        name=name if name is not None else "",
        latitude=latitude if latitude is not None else 0.0,
        longitude=longitude if longitude is not None else 0.0,
        area_hectares=area if area is not None else 1.0,
        crop_type=crop_type if crop_type is not None else "apricot",
    )
    return JsonResponse({"field": _field_payload(field), "id": field.id, "name": field.name}, status=201)


def soil_data(request: HttpRequest, field_id: int) -> JsonResponse:
    if request.method != "GET":
        return _method_not_allowed(request.method)

    field, error_response = _get_field(field_id)
    if error_response is not None:
        return error_response
    assert field is not None

    readings = list(
        SoilReading.objects.filter(field=field)
        .order_by("-timestamp")
        .values("id", "moisture", "temperature", "ph", "depth", "nitrogen", "timestamp")[:20]
    )
    return JsonResponse(
        {
            "field": field.name,
            "field_id": field.id,
            "current": readings[0] if readings else None,
            "history": readings,
        }
    )


@csrf_exempt
def soil_add(request: HttpRequest, field_id: int) -> JsonResponse:
    if request.method != "POST":
        return _method_not_allowed(request.method)

    field, error_response = _get_field(field_id)
    if error_response is not None:
        return error_response
    assert field is not None

    payload, error_response = _parse_json_body(request)
    if error_response is not None:
        return error_response
    assert payload is not None

    moisture, error = _read_number(payload, "moisture", required=True, minimum=0, maximum=100)
    if (response := _validation_error(error)) is not None:
        return response
    temperature, error = _read_number(payload, "temperature", required=True, minimum=-50, maximum=80)
    if (response := _validation_error(error)) is not None:
        return response
    ph, error = _read_number(payload, "ph", default=6.5, minimum=0, maximum=14)
    if (response := _validation_error(error)) is not None:
        return response
    depth, error = _read_number(payload, "depth", default=0.3, minimum=0, maximum=10)
    if (response := _validation_error(error)) is not None:
        return response
    nitrogen, error = _read_number(payload, "nitrogen", default=None, minimum=0, maximum=10000)
    if (response := _validation_error(error)) is not None:
        return response

    reading = SoilReading.objects.create(
        field=field,
        moisture=moisture if moisture is not None else 0.0,
        temperature=temperature if temperature is not None else 0.0,
        ph=ph if ph is not None else 6.5,
        depth=depth if depth is not None else 0.3,
        nitrogen=nitrogen,
    )
    return JsonResponse(
        {
            "status": "ok",
            "id": reading.id,
            "reading": {
                "id": reading.id,
                "moisture": reading.moisture,
                "temperature": reading.temperature,
                "ph": reading.ph,
                "depth": reading.depth,
                "nitrogen": reading.nitrogen,
                "timestamp": reading.timestamp,
            },
        },
        status=201,
    )


def _save_prediction(field: Field, forecast_data: list[dict[str, Any]], days_to_water: Optional[int]) -> None:
    prediction = Prediction.objects.filter(field=field).order_by("-created_at").first()
    if prediction is None:
        Prediction.objects.create(
            field=field,
            forecast_json=forecast_data,
            days_until_water=days_to_water,
        )
        return

    Prediction.objects.filter(field=field).exclude(id=prediction.id).delete()
    prediction.forecast_json = forecast_data
    prediction.days_until_water = days_to_water
    prediction.save(update_fields=["forecast_json", "days_until_water"])


def forecast(request: HttpRequest, field_id: int) -> JsonResponse:
    if request.method != "GET":
        return _method_not_allowed(request.method)

    field, error_response = _get_field(field_id)
    if error_response is not None:
        return error_response
    assert field is not None

    last_reading = SoilReading.objects.filter(field=field).order_by("-timestamp").first()
    current_moisture = last_reading.moisture if last_reading else 40.0
    current_temp = last_reading.temperature if last_reading else 25.0

    moisture_forecast = predict_moisture_7days(current_moisture, current_temp)
    days_to_water = check_water_needed(moisture_forecast)
    forecast_data = [
        {"day": index + 1, "moisture": value, "needs_water": value < 30}
        for index, value in enumerate(moisture_forecast)
    ]

    _save_prediction(field, forecast_data, days_to_water)

    recommendation = (
        f"Полив нужен через {days_to_water} дн."
        if days_to_water
        else "Состояние нормальное"
    )
    return JsonResponse(
        {
            "field": field.name,
            "field_id": field.id,
            "current_moisture": round(current_moisture, 1),
            "forecast": forecast_data,
            "recommendation": recommendation,
            "days_until_water": days_to_water,
        }
    )


def weather(request: HttpRequest, field_id: int) -> JsonResponse:
    if request.method != "GET":
        return _method_not_allowed(request.method)

    field, error_response = _get_field(field_id)
    if error_response is not None:
        return error_response
    assert field is not None

    current = get_current_weather(field.latitude, field.longitude)
    return JsonResponse({"field": field.name, "field_id": field.id, "weather": current})


def _upsert_alert(field: Field, alert_type: str, level: str, message: str) -> None:
    alerts = list(Alert.objects.filter(field=field, alert_type=alert_type).order_by("id"))
    if not alerts:
        Alert.objects.create(
            field=field,
            alert_type=alert_type,
            level=level,
            message=message,
            is_active=True,
        )
        return

    keep = alerts[0]
    keep.level = level
    keep.message = message
    keep.is_active = True
    keep.save(update_fields=["level", "message", "is_active"])
    Alert.objects.filter(field=field, alert_type=alert_type).exclude(id=keep.id).update(is_active=False)


def alerts(request: HttpRequest, field_id: int) -> JsonResponse:
    if request.method != "GET":
        return _method_not_allowed(request.method)

    field, error_response = _get_field(field_id)
    if error_response is not None:
        return error_response
    assert field is not None

    active_alert_types: set[str] = set()
    for alert in check_disaster_risk(field.latitude, field.longitude):
        alert_type = str(alert["type"])
        active_alert_types.add(alert_type)
        _upsert_alert(field, alert_type, str(alert["level"]), str(alert["message"]))

    last = SoilReading.objects.filter(field=field).order_by("-timestamp").first()
    if last and last.moisture < 30:
        active_alert_types.add("water")
        _upsert_alert(
            field,
            "water",
            "high",
            f"Влажность {round(last.moisture, 1)}% — нужен полив",
        )

    managed_types = {"water", "hail", "flood"}
    Alert.objects.filter(field=field, alert_type__in=managed_types - active_alert_types).update(is_active=False)
    all_alerts = list(
        Alert.objects.filter(field=field, is_active=True)
        .order_by("-created_at")
        .values("alert_type", "level", "message", "created_at")
    )

    return JsonResponse({"field": field.name, "field_id": field.id, "alerts": all_alerts, "count": len(all_alerts)})
