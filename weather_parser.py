import requests
from bs4 import BeautifulSoup
import json

# Get today's weather for selected location from Australian Bureau of Meteorology
def get_weather_info(location_id):
    # Construct the url
    url = "http://www.bom.gov.au/climate/dwo/IDCJDW" + str(location_id) + ".latest.shtml" 
    # Add a user-agent header
    headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    # Send a request to the website and get the content
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    day_stats = soup.find_all('tbody')[0] # Table with daily stats
    last_row = day_stats.find_all('tr')[-1] # Last row = today
    cols = last_row.find_all('td') # Get all columns
    weather_data_part_1 = {
        'MinTemp': cols[1].text.strip(),
        'MaxTemp': cols[2].text.strip(),
        'Rainfall': cols[3].text.strip(),
        'Evaporation': cols[4].text.strip(),
        'Sunshine': cols[5].text.strip(),
        'WindGustDir': cols[6].text.strip(),
        'WindGustSpeed': cols[7].text.strip(),
        'Temp9am': cols[9].text.strip(),
        'Humidity9am': cols[10].text.strip(),
        'Cloud9am': cols[11].text.strip()
    }
    # Workaround for when there's no wind and WindDir9am and WindSpeed9am are merged into one column
    if cols[12].text.strip() != 'Calm':
        weather_data_part_2 = {   
            'WindDir9am': cols[12].text.strip(),        
            'WindSpeed9am': cols[13].text.strip(),
            'Pressure9am': cols[14].text.strip(),
            'Temp3pm': cols[15].text.strip(),
            'Humidity3pm': cols[16].text.strip(),      
            'Cloud3pm': cols[17].text.strip()
        }
    else:
        weather_data_part_2 = {   
            'WindDir9am': None,        
            'WindSpeed9am': 0,
            'Pressure9am': cols[13].text.strip(),
            'Temp3pm': cols[14].text.strip(),
            'Humidity3pm': cols[15].text.strip(),      
            'Cloud3pm': cols[16].text.strip()
        }
    # Workaround for when there's no wind and WindDir3pm and WindSpeed3pm are merged into one column    
    if cols[-2].text.strip() != 'Calm':
        weather_data_part_3 = {  
            'WindDir3pm': cols[-3].text.strip(),
            'WindSpeed3pm': cols[-2].text.strip(),  
            'Pressure3pm': cols[-1].text.strip()
          }
    else:
        weather_data_part_3 = {  
            'WindDir3pm': None,
            'WindSpeed3pm': 0,  
            'Pressure3pm': cols[-1].text.strip()
          }
    # Convert to JSON
    weather_data = dict(weather_data_part_1)
    weather_data.update(weather_data_part_2)
    weather_data.update(weather_data_part_3)
    json_data = json.loads(json.dumps(weather_data, indent=4)) 
    
    return json_data