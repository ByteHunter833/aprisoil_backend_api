# ApriSoil Backend API

A Django-based backend API for intelligent agricultural soil monitoring and crop management. ApriSoil combines real-time soil sensors, weather forecasting, and physics-informed neural networks (PINNs) to provide data-driven irrigation recommendations for farmers growing apricots and other crops.

## Features

- **Field Management** — Create and track multiple agricultural fields with GPS coordinates, crop types, and area metrics
- **Soil Monitoring** — Collect real-time soil sensor data including moisture, temperature, pH, nitrogen levels, and measurement depth
- **7-Day Moisture Forecast** — ML-powered predictions using physics-informed neural networks trained on soil properties
- **Smart Irrigation Alerts** — Automatic recommendations for watering based on soil moisture thresholds and forecasts
- **Weather Integration** — Real-time weather data and disaster risk detection (hail, flooding) via OpenWeatherMap
- **Cross-Origin Support** — CORS-enabled for Flutter/mobile clients

## Stack

- **Language:** Python 3.12
- **Framework:** Django 6.0.5 + Django REST Framework 3.17.1
- **ML/Scientific:** TensorFlow 2.21.0, NumPy, Pandas, scikit-learn
- **Server:** Gunicorn + Uvicorn (dual-mode deployment)
- **Database:** SQLite (development), configurable for production
- **Containerization:** Docker

## How It's Organized

```
aprisoil_backend_api/
  api/                    # Main Django app
    models.py            # Field, SoilReading, Alert, Prediction, WeatherData ORM models
    views.py             # REST endpoints for fields, soil data, forecasts, weather, alerts
    urls.py              # URL routing configuration
    weather_service.py   # OpenWeatherMap API integration, disaster risk detection
    admin.py             # Django admin configuration
    migrations/          # Database schema versions

  aprisoil/              # Django project configuration
    settings.py          # Core settings: DEBUG, DATABASES, INSTALLED_APPS, CORS
    urls.py              # Root URL dispatcher
    wsgi.py              # WSGI application entry point (Gunicorn)
    asgi.py              # ASGI application entry point (Uvicorn)

  ml/                    # Machine learning models
    pinn_model.py        # Physics-Informed Neural Network (Richards equation soil water dynamics)
    predictor.py         # High-level API for moisture forecasting & irrigation checks
    sandy_loam_nod.csv   # Training dataset (soil moisture/depth/time measurements)

  manage.py              # Django command-line utility
  pyrun.py               # Simple Python runner script
  requirements.txt       # Python dependencies
  Dockerfile             # Docker image configuration (Python 3.12 slim)
  Procfile               # Heroku/Railway deployment config
  pyrightconfig.json     # Pyright type checker configuration
```

## How It Fits Together

1. **Request Flow:**
   - Client (Flutter app) sends HTTP requests to REST endpoints in `api/views.py`
   - Views validate input, query/update Django models via ORM
   - For forecast requests, the ML pipeline is invoked

2. **Data Flow:**
   - Soil sensors → `/api/soil/<field_id>/add` → SoilReading model
   - Latest reading + ML model → `/api/forecast/<field_id>` → Moisture prediction (7 days) → Irrigation recommendation
   - GPS coordinates → Weather service → `/api/weather/<field_id>` → Current conditions + disaster alerts

3. **ML Integration:**
   - PINN model trained offline on `sandy_loam_nod.csv` (soil physics data for sandy loam texture)
   - `ml/predictor.py` exposes `predict_moisture_7days()` and `check_water_needed()`
   - Predictions cached in `Prediction` model to avoid redundant computation

## How to Run It

### Prerequisites

- Python 3.12
- pip
- Docker (optional)

### Local Development

1. **Clone and setup:**
   ```bash
   git clone https://github.com/ByteHunter833/aprisoil_backend_api.git
   cd aprisoil_backend_api
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env  # or create .env manually
   ```
   Required variables:
   ```
   SECRET_KEY=your-secret-key-here
   DEBUG=True
   OPENWEATHER_API_KEY=your-api-key-from-openweathermap.org
   ```

3. **Initialize database:**
   ```bash
   python manage.py migrate
   ```

4. **Run development server:**
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```
   Server will be available at `http://localhost:8000`

### Docker Deployment

```bash
docker build -t aprisoil-api .
docker run -p 7860:7860 \
  -e SECRET_KEY=your-secret-key \
  -e DEBUG=False \
  -e OPENWEATHER_API_KEY=your-api-key \
  aprisoil-api
```

### Production Deployment (Railway/Heroku)

The `Procfile` is configured for automatic deployment:
```bash
web: python manage.py migrate && gunicorn aprisoil.wsgi:application
```

Set environment variables in your platform's dashboard, then deploy.

## API Endpoints

### Fields
- `GET /api/fields/` — List all fields
- `POST /api/fields/create/` — Create a new field

**Request body (POST):**
```json
{
  "name": "North Orchard",
  "latitude": 42.8746,
  "longitude": 74.5698,
  "area_hectares": 12.5,
  "crop_type": "apricot"
}
```

### Soil Data
- `GET /api/soil/<field_id>/` — Get soil readings (latest 20)
- `POST /api/soil/<field_id>/add/` — Record new soil reading

**Request body (POST):**
```json
{
  "moisture": 35.2,
  "temperature": 22.5,
  "ph": 6.8,
  "depth": 0.3,
  "nitrogen": 120
}
```

### Forecasts
- `GET /api/forecast/<field_id>/` — 7-day moisture forecast & watering recommendation

### Weather
- `GET /api/weather/<field_id>/` — Current weather + conditions

### Alerts
- `GET /api/alerts/<field_id>/` — Active alerts (watering, hail risk, flood risk, etc.)

## Environment Variables

| Variable | Required | Example | Notes |
|----------|----------|---------|-------|
| `SECRET_KEY` | ✓ | `django-insecure-...` | Django secret; use strong random value in production |
| `DEBUG` | ✗ | `True` | Set to `False` in production |
| `OPENWEATHER_API_KEY` | ✓ | `abc123...` | Free tier available at openweathermap.org |
| `PORT` | ✗ | `8000` | Server port (auto-set by Railway) |
| `DATABASE_URL` | ✗ | `postgres://...` | Optional PostgreSQL connection string |

## Development

### Running Tests
```bash
python manage.py test
```

### Code Quality
```bash
pyright .              # Type checking
python manage.py check # Django system checks
```

### Database Migrations
```bash
python manage.py makemigrations  # After model changes
python manage.py migrate         # Apply pending migrations
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'api'` | Ensure `api` is in `INSTALLED_APPS` in `settings.py` |
| CORS errors from Flutter | Confirm `CORS_ALLOW_ALL_ORIGINS = True` in settings |
| ML model slow on first request | PINN initialization is expensive; allow 10-15s first call |
| Weather alerts not appearing | Verify `OPENWEATHER_API_KEY` is set and valid |
| Database locked errors | Delete `db.sqlite3` and re-run migrations for development |

## Related Repositories

- **Frontend:** [ApriSoil Flutter App](https://github.com/Aikushka/aprisoil) — Mobile client for farmers
- **ML Research:** Physics-informed neural networks for soil water dynamics (Richards equation)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

## License

Not specified. Contact repository owner for licensing details.

## Contact & Support

- **Repository Owner:** [@ByteHunter833](https://github.com/ByteHunter833)
- **Upstream Project:** [ApriSoil](https://github.com/Aikushka/aprisoil)
- **Issues:** [GitHub Issues](https://github.com/ByteHunter833/aprisoil_backend_api/issues)
