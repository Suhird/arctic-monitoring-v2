import requests
import datetime

date_str = "2026-01-10"
style = "visual"
dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
yyyy = dt.strftime("%Y")
mm = dt.strftime("%b").lower()
yyyymmdd = dt.strftime("%Y%m%d")

url = f"https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250/{yyyy}/{mm}/Arctic/asi-AMSR2-n6250-{yyyymmdd}-v5.4_{style}.png"
print(f"Testing URL: {url}")

try:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Content Type: {response.headers.get('Content-Type')}")
    print(f"Content Length: {response.headers.get('Content-Length')}")
except Exception as e:
    print(f"Error: {e}")
