import pygrib
import pandas as pd
import numpy as np
import os
import shutil
import tarfile
import logging
import sys
import requests
from datetime import datetime, timedelta

from pvlib import irradiance
from pvlib.solarposition import get_solarposition
from pvlib import irradiance, solarposition

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))

EINDHOVEN_LAT = 51.4416
EINDHOVEN_LON = 5.4697
DATASET_NAME = "harmonie_arome_cy40_p1"
DATASET_VERSION = "0.2"


class OpenDataAPI:
    def __init__(self, api_token: str):
        self.base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
        self.headers = {"Authorization": api_token}

    def __get_data(self, url, params=None):
        return requests.get(url, headers=self.headers, params=params).json()

    def list_files(self, dataset_name: str, dataset_version: str, params: dict):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files",
            params=params,
        )

    def get_file_url(self, dataset_name: str, dataset_version: str, file_name: str):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{file_name}/url"
        )


def download_file_from_temporary_download_url(download_url, filename):
    try:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        logger.exception("Unable to download file using download URL")
        sys.exit(1)

    logger.info(f"Successfully downloaded dataset file to {filename}")


def unpack_tar_file(tar_path: str) -> str:
    """Extract all contents of a tar file in the same directory as the tar file.

    Args:
        tar_path (str): Path to the tar file to extract

    Returns:
        str: Path to the folder where the tar file was extracted
    """

    # Create destination folder path
    dest_folder = os.path.join(
        os.path.dirname(tar_path), os.path.basename(tar_path).rsplit(".", 1)[0]
    )
    os.makedirs(dest_folder, exist_ok=True)  # Ensure the destination folder exists

    # Extract all contents of file in destination folder path
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dest_folder)

    # Remove the tar file
    os.remove(tar_path)

    return dest_folder


def read_grib_folder(grib_folder: str) -> pd.DataFrame:
    """Read all grib files in a folder and extract the weather data for Eindhoven.

    Args:
        grib_folder (str): Path to the folder containing the grib files

    Returns:
        pd.DataFrame: DataFrame containing the weather data for Eindhoven
    """

    global EINDHOVEN_LON, EINDHOVEN_LAT

    # List needed parameters, see code matrix KNMI
    parameters = {
        "temperature": "11",
        "windU": "33",
        "windV": "34",
        "globalRadiation": "117",
    }

    # Initialize list to hold the data
    data_list = []

    # Loop over each file
    for file_name in os.listdir(grib_folder):
        if file_name.endswith("_GB"):
            grib_file = os.path.join(grib_folder, file_name)
            grbs = pygrib.open(grib_file)

            # Retrieve the lat/lon grid
            first_message = grbs.message(1)
            lats, lons = first_message.latlons()

            # Find the closest grid point
            distance = np.sqrt(
                (lats - EINDHOVEN_LAT) ** 2 + (lons - EINDHOVEN_LON) ** 2
            )
            min_index = distance.argmin()

            data_date = str(first_message.dataDate)  # Format: YYYYMMDD
            data_time = first_message.dataTime  # Format: HHMM

            # Create the base datetime object from dataDate and dataTime
            base_datetime = datetime.strptime(
                f"{data_date} {data_time:04d}", "%Y%m%d %H%M"
            )
            step_range = float(first_message.stepRange)
            valid_datetime = base_datetime + timedelta(hours=step_range)

            # Initialize a dictionary to hold the data for this file
            data_dict = {
                "file_name": file_name,
                "datetime": valid_datetime,  # ,
            }

            # Extract data for each parameter
            for param_name in parameters:
                try:
                    grb_message = grbs.select(parameterName=parameters[param_name])[
                        0
                    ]  # First instance
                    eindhoven_value = grb_message.values.flat[min_index]
                    data_dict[param_name] = eindhoven_value
                except (IndexError, ValueError):
                    data_dict[param_name] = (
                        np.nan
                    )  # When parameter is not found in grib file

            grbs.close()

            # Append dictionary to list
            data_list.append(data_dict)

    # Convert list of dictionaries to DF
    weather_df = pd.DataFrame(data_list)

    weather_df["windSpeed"] = np.sqrt(
        weather_df["windU"] ** 2 + weather_df["windV"] ** 2
    )
    weather_df["temperature"] = weather_df["temperature"] - 272.15

    weather_df["globalRadiation"] = weather_df["globalRadiation"].fillna(0)
    # Calculate the difference between consecutive rows
    weather_df["Q"] = weather_df["globalRadiation"].diff()
    weather_df["Q"] = weather_df["Q"] / 3600

    # Get solar position for the dates / times
    solpos_df = solarposition.get_solarposition(
        weather_df["datetime"],
        latitude=EINDHOVEN_LAT,
        longitude=EINDHOVEN_LON,
        altitude=0,
        temperature=weather_df["temperature"],
    )
    solpos_df.index = weather_df.index

    # Method 'Erbs' to go from GHI to DNI and DHI
    irradiance_df = irradiance.erbs(
        weather_df["Q"], solpos_df["zenith"], weather_df.index
    )
    irradiance_df["ghi"] = weather_df["Q"]

    # Add DNI and DHI to weather_df
    weather_df["dni"] = irradiance_df["dni"]
    weather_df["dhi"] = irradiance_df["dhi"]

    df = weather_df.copy()
    df = df.drop(["windU", "windV"], axis=1)
    df = df[:-1]
    df = df.fillna(0)

    # Remove the folder with the grib files
    shutil.rmtree(grib_folder)

    return df


def group_by_data_and_cache(df: pd.DataFrame) -> pd.DataFrame:
    """Group the weather data by day and cache it in a CSV file.

    Args:
        df (pd.DataFrame): The full dataframe containing the weather data for Eindhoven per hour

    Returns:
        pd.DataFrame: The weather data grouped by day
    """

    df["date"] = df["datetime"].dt.date
    df = df.drop(["file_name", "datetime"], axis=1)
    grouped = df.groupby("date")

    # Aggregate hourly values into lists per day
    daily_data = grouped.agg(list)
    keeps = {
        "temperature": "temperature_sequence",
        "windSpeed": "wind_speed_sequence",
        "dni": "dni_sequence",
        "dhi": "dhi_sequence",
        "globalRadiation": "global_irradiance_sequence",
    }
    daily_data = daily_data[keeps.keys()].rename(keeps, axis=1)
    daily_data.index.name = "date"

    # Get the index of the first date
    date = daily_data.index[0]

    # Save it in a way that can be cached
    daily_data.to_csv(f"energy_data/{date}.csv")

    return daily_data


def fetch_data_from_api() -> pd.DataFrame:
    """Fetch the latest weather data from the KNMI API and cache it.

    Returns:
        pd.DataFrame: The weather data for Eindhoven grouped by day
    """

    global DATASET_NAME, DATASET_VERSION

    api_key = os.getenv("KNMI_API_KEY")
    # For this its ok to launch a connection to the API every time
    # because there wont be many requests
    api = OpenDataAPI(api_token=api_key)

    # Fetch the latest 4 files and order them by creation date
    params = {"maxKeys": 4, "orderBy": "created", "sorting": "desc"}

    logger.info(f"Fetching latest file of {DATASET_NAME} version {DATASET_VERSION}")

    # Sort the files in descending order and only retrieve the first file
    response = api.list_files(DATASET_NAME, DATASET_VERSION, params)

    # Warn if there was an error in the response
    if "error" in response:
        logger.error(f"Unable to retrieve list of files: {response['error']}")
        return

    # Filter files that end with '00.tar'
    filtered_files = [
        f for f in response["files"] if f.get("filename").endswith("00.tar")
    ]

    if not filtered_files:
        logger.error("No files ending with '00.tar' found")
        return

    # Assuming files are already sorted by creation date in the response, get the latest
    latest_file = filtered_files[0].get("filename")
    logger.info(f"Latest file is: {latest_file}")

    # Fetch the download url and download the tar file
    response = api.get_file_url(DATASET_NAME, DATASET_VERSION, latest_file)
    download_file_from_temporary_download_url(
        response["temporaryDownloadUrl"], latest_file
    )

    # Untar the file into a directory
    logger.info(f"Unpacking {latest_file}")

    grib_folder = unpack_tar_file(latest_file)

    logger.info("Reading grib files and extracting data")

    # Read the grib files and extract the data
    daily_data = read_grib_folder(grib_folder)

    logger.info("Grouping data by date and caching it")

    # Group the data by date and cache it
    daily_data = group_by_data_and_cache(daily_data)

    return daily_data


def get_cached_data(today: datetime) -> pd.DataFrame | None:
    """Check if the data for today is cached and return it if it is.

    Args:
        today (datetime): The current date

    Returns:
        pd.DataFrame: The cached data if it exists, None otherwise
    """

    if not os.path.exists("energy_data"):
        os.makedirs("energy_data")

    # Get the date of the first file
    date = today.date()

    # Check if the file exists
    if os.path.exists(f"weather_data/{date}.csv"):
        return pd.read_csv(f"weather_data/{date}.csv", index_col="date")

    # If the file does not exist, return None
    return None


def get_predicted_data() -> pd.DataFrame:
    """Get the predicted data for today and tomorrow.

    Returns:
        pd.DataFrame: The predicted data for today and tomorrow
    """

    today = datetime.now()
    cached_data = get_cached_data(today)

    if cached_data is not None:
        logger.info(f"Using cached data for {today.date()}")
        return cached_data

    logger.info("Fetching data from the KNMI API")

    return fetch_data_from_api()
