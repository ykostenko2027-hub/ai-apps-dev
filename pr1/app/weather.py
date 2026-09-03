import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT = 10.0


class WeatherError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class CityNotFoundError(WeatherError):
    def __init__(self, city: str):
        super().__init__(f"Місто '{city}' не знайдено", status_code=404)


class WeatherServiceError(WeatherError):
    def __init__(self, message: str = "Помилка сервісу погоди", status_code: int = 502):
        super().__init__(message, status_code=status_code)


class WeatherTimeoutError(WeatherError):
    def __init__(self, message: str = "Перевищено час очікування відповіді від сервісу погоди"):
        super().__init__(message, status_code=504)


def find_city(name: str):
    if not name or not name.strip():
        raise WeatherError("Назву міста не вказано", status_code=400)

    clean_name = name.strip()
    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": clean_name, "count": 1, "language": "uk", "format": "json"},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.Timeout:
        raise WeatherTimeoutError()
    except requests.RequestException:
        raise WeatherServiceError("Не вдалося з'єднатися із сервісом геокодування")

    if response.status_code >= 400:
        raise WeatherServiceError(f"Сервіс геокодування повернув помилку: {response.status_code}", status_code=502)

    try:
        data = response.json()
    except ValueError:
        raise WeatherServiceError("Некоректний JSON від сервісу геокодування")

    results = data.get("results")
    if not results or not isinstance(results, list):
        raise CityNotFoundError(clean_name)

    first = results[0]
    lat = first.get("latitude")
    lon = first.get("longitude")
    city_name = first.get("name", clean_name)
    country = first.get("country", "")

    if lat is None or lon is None:
        raise WeatherServiceError("Відсутні координати міста у відповіді сервісу")

    return {"name": city_name, "country": country, "latitude": lat, "longitude": lon}


def get_current_weather(city: str):
    city_info = find_city(city)

    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": city_info["latitude"],
                "longitude": city_info["longitude"],
                "current": "temperature_2m,wind_speed_10m",
            },
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.Timeout:
        raise WeatherTimeoutError()
    except requests.RequestException:
        raise WeatherServiceError("Не вдалося з'єднатися із сервісом погоди")

    if response.status_code >= 400:
        raise WeatherServiceError(f"Сервіс погоди повернув помилку: {response.status_code}", status_code=502)

    try:
        data = response.json()
    except ValueError:
        raise WeatherServiceError("Некоректний JSON від сервісу погоди")

    current = data.get("current")
    units = data.get("current_units", {})

    if not current or not isinstance(current, dict):
        raise WeatherServiceError("Немає даних про поточну погоду у відповіді сервісу")

    temp = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")

    if temp is None or wind is None:
        raise WeatherServiceError("Відсутні дані про температуру або швидкість вітру")

    return {
        "city": city_info["name"],
        "country": city_info["country"],
        "latitude": city_info["latitude"],
        "longitude": city_info["longitude"],
        "temperature": temp,
        "temperature_unit": units.get("temperature_2m", "°C"),
        "wind_speed": wind,
        "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
    }
