from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import (HumanMessage)

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

json_string_message = """
[
  {
    "url": "https://www.weatherapi.com/",
    "content": "{'location': {'name': 'Bali', 'region': 'North Sumatra', 'country': 'Indonesia', 'lat': -0.1395, 'lon': 98.186, 'tz_id': 'Asia/Jakarta', 'localtime_epoch': 1734870804, 'localtime': '2024-12-22 19:33'}, 'current': {'last_updated_epoch': 1734870600, 'last_updated': '2024-12-22 19:30', 'temp_c': 28.5, 'temp_f': 83.3, 'is_day': 0, 'condition': {'text': 'Clear', 'icon': '//cdn.weatherapi.com/weather/64x64/night/113.png', 'code': 1000}, 'wind_mph': 6.9, 'wind_kph': 11.2, 'wind_degree': 307, 'wind_dir': 'NW', 'pressure_mb': 1007.0, 'pressure_in': 29.74, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 73, 'cloud': 23, 'feelslike_c': 32.2, 'feelslike_f': 90.0, 'windchill_c': 28.5, 'windchill_f': 83.3, 'heatindex_c': 32.2, 'heatindex_f': 90.0, 'dewpoint_c': 23.1, 'dewpoint_f': 73.6, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 0.0, 'gust_mph': 9.4, 'gust_kph': 15.2}}"
  },
  {
    "url": "https://www.detik.com/bali/berita/d-7697846/perkiraan-cuaca-suhu-dan-kelembapan-udara-bali-22-desember-2024",
    "content": "Bali - Cuaca di Bali pada Minggu, 22 Desember 2024 bervariasi, mulai dari udara kabur hingga hujan ringan. Prakiraan cuaca itu berdasarkan pantauan dari Balai Besar Meteorologi Klimatologi Geofisika (BMKG) Wilayah III Denpasar. Secara umum, cuaca Bali pada hari ini diprediksi udara kabur."
  }
]
"""

response = llm.invoke([
    HumanMessage(content=json_string_message)
])

print(response.content)
