import requests
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import os
import pandas as pd
import json
from requests.exceptions import HTTPError, Timeout, RequestException

#Set the environment variable "CALCULUS_API_KEY ' to your token and restart the device
token = os.getenv("CALCULUS_API_KEY", "***")
headers = {
    "CalculusApiKey" : token
}

base_url = 'https://api.calculus.group/v3'

def query_endpoint(endpoint, header, assetid=None, start_time=None, end_time=None, dry_run=True,timeout_seconds = 100):
    """
    Queries the specified API endpoint.

    Parameters:
        base_url (str): The base URL of the API.
        endpoint (str): The specific endpoint to query.
        id (int, optional): The identifier to include in the endpoint URL.
        start_time (datetime, optional): The start time for time-based queries. 
            Should be a timezone-aware datetime object.
        end_time (datetime, optional): The end time for time-based queries. 
            Should be a timezone-aware datetime object.
        dry_run (bool, optional): If True, only prints the URL without making the request. 
            Default is False.

    Returns:
        dict or None: The JSON response from the API if the request is successful,
            None if there is an error or if dry_run is True.
    """
    url = f"{base_url}/assets"
    
    if assetid is not None:
        url += f"/{assetid}"
    
    url += f"/{endpoint}"
    
    if start_time is not None and end_time is not None:
        start_unix = datetime_to_unix(start_time)
        end_unix = datetime_to_unix(end_time)
        url += f"?unixTimestampStart={start_unix}&unixTimestampEnd={end_unix}"
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout_seconds)
        response.raise_for_status()  # Raises HTTPError if response status code is not 2xx
        return response.json()
    except Timeout:
        # print("The request timed out")
        return 'Timeout'
    except HTTPError as e:
        # print(f"HTTP error occurred: {e}")
        return 'HTTP Error'
    except RequestException as e:
        # print(f"An error occurred while making the request: {e}")
        return 'Other API Error'



def datetime_to_unix(dt):
    """Converts a datetime object to Unix timestamp."""
    # Ensure the datetime object has time zone information
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("Datetime object must have time zone information")

    # Convert datetime to UTC timezone
    dt_utc = dt.astimezone(ZoneInfo("UTC"))

    # Calculate Unix timestamp
    unix_timestamp = (dt_utc - datetime(1970, 1, 1, tzinfo=ZoneInfo("UTC"))).total_seconds()
    return int(unix_timestamp)

def extract_reading_data(data, pid):
    reading_data = []
    for source in data['dataSources']:
        sensor_name = source['name']
        for series in source['dataSeries']:
            key_parts = series['key'].split('|')
            sensor_key = key_parts[1].split('#')[0]  # Extracting sensor key (e.g., 'battery', 'co2', etc.)
            for entry in series['value']:
                timestamp = entry['key']
                value = entry['value']
                reading_data.append({'SensorID': pid, 'SensorType': sensor_name, 'Timestamp': timestamp, sensor_key: value})
    return reading_data

def load_assets_from_json(filepath):
    """
    Load asset data from a specified JSON file.
    
    This function reads a JSON file from the given filepath and parses its content into a Python data structure.
    
    Parameters:
        filepath (str): The path to the JSON file to be read.
        
    Returns:
        A list
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


