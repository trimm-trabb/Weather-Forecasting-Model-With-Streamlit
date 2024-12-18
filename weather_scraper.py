import requests
from bs4 import BeautifulSoup
import json

class WeatherScraper:
    """
    A class to scrape weather data from the Australian Bureau of Meteorology (BOM).
    """
    BASE_URL = "http://www.bom.gov.au/climate/dwo/IDCJDW{}.latest.shtml"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    def __init__(self, location_id):
        """
        Initialize the WeatherScraper with a location ID.
        
        Args:
            location_id (int): Location ID for BOM URL.
        """
        self.location_id = location_id
        self.url = self.BASE_URL.format(location_id)
        self.weather_data = {}

    def fetch_weather_page(self):
        """
        Fetch the weather page HTML content.

        Returns:
            BeautifulSoup: Parsed HTML content of the weather page.
        """
        try:
            response = requests.get(self.url, headers=self.HEADERS)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None

    def parse_table_row(self, cols):
        """
        Parse a weather data table row into a dictionary.

        Args:
            cols (list): List of BeautifulSoup elements representing table columns.
        """
        self.weather_data = {
            'MinTemp': cols[1].text.strip(),
            'MaxTemp': cols[2].text.strip(),
            'Rainfall': cols[3].text.strip(),
            'Evaporation': cols[4].text.strip(),
            'Sunshine': cols[5].text.strip(),
            'WindGustDir': cols[6].text.strip(),
            'WindGustSpeed': cols[7].text.strip(),
            'Temp9am': cols[9].text.strip(),
            'Humidity9am': cols[10].text.strip(),
            'Cloud9am': cols[11].text.strip(),
        }

        # Handle WindDir9am and WindSpeed9am
        if cols[12].text.strip() != 'Calm':
            self.weather_data.update({
                'WindDir9am': cols[12].text.strip(),
                'WindSpeed9am': cols[13].text.strip(),
                'Pressure9am': cols[14].text.strip(),
                'Temp3pm': cols[15].text.strip(),
                'Humidity3pm': cols[16].text.strip(),
                'Cloud3pm': cols[17].text.strip(),
            })
            wind_dir_3pm, wind_speed_3pm, pressure_3pm = cols[-3], cols[-2], cols[-1]
        else:
            self.weather_data.update({
                'WindDir9am': None,
                'WindSpeed9am': 0,
                'Pressure9am': cols[13].text.strip(),
                'Temp3pm': cols[14].text.strip(),
                'Humidity3pm': cols[15].text.strip(),
                'Cloud3pm': cols[16].text.strip(),
            })
            wind_dir_3pm, wind_speed_3pm, pressure_3pm = cols[-3], cols[-1], cols[-1]

        # Handle WindDir3pm and WindSpeed3pm
        if wind_speed_3pm.text.strip() != 'Calm':
            self.weather_data.update({
                'WindDir3pm': wind_dir_3pm.text.strip(),
                'WindSpeed3pm': wind_speed_3pm.text.strip(),
                'Pressure3pm': pressure_3pm.text.strip(),
            })
        else:
            self.weather_data.update({
                'WindDir3pm': None,
                'WindSpeed3pm': 0,
                'Pressure3pm': pressure_3pm.text.strip(),
            })

    def get_weather_info(self):
        """
        Scrape the weather data for the given location ID.

        Returns:
            dict: Weather data as a dictionary.
        """
        soup = self.fetch_weather_page()
        if not soup:
            return None

        try:
            # Find the table and last row containing weather data
            day_stats = soup.find_all('tbody')[0]
            last_row = day_stats.find_all('tr')[-1]
            cols = last_row.find_all('td')

            self.parse_table_row(cols)
            return self.weather_data

        except (IndexError, AttributeError) as e:
            print(f"Error parsing weather data: {e}")
            return None

    def to_json(self):
        """
        Convert the scraped weather data to JSON format.

        Returns:
            str: Weather data in JSON format.
        """
        if self.weather_data:
            return json.dumps(self.weather_data, indent=4)
        return "{}"