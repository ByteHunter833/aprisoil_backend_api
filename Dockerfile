FROM python:3.12-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Порт 7860 обязателен для Hugging Face
# Если у тебя FastAPI:
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

# Если у тебя Flask, закомментируй строку выше и раскомментируй эту:
# CMD ["gunicorn", "main:app", "-b", "0.0.0.0:7860"]