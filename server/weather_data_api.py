import pygrib
import pandas as pd
import numpy as np
import os
import tarfile
import logging
import sys
import requests
from datetime import datetime, timedelta

from pvlib import pvsystem, modelchain, location, irradiance
from pvlib.solarposition import get_solarposition
from pvlib import irradiance, solarposition

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))

EINDHOVEN_LAT = 51.4416
EINDHOVEN_LON = 5.4697


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


def unpack_tar_file(tar_path: str):
    """Extract all contents of a tar file in the same directory as the tar file.

    Args:
        tar_path (str): Path to the tar file to extract
    """

    # Create destination folder path
    dest_folder = os.path.join(
        os.path.dirname(tar_path), os.path.basename(tar_path).rsplit(".", 1)[0]
    )
    os.makedirs(dest_folder, exist_ok=True)  # Ensure the destination folder exists

    # Extract all contents of file in destination folder path
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dest_folder)


def read_grib_folder(grib_folder: str):
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
            nearest_point_lat = lats.flat[min_index]
            nearest_point_lon = lons.flat[min_index]

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
                #'latitude': nearest_point_lat,
                #'longitude': nearest_point_lon
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
    gribData = pd.DataFrame(data_list)
    df = gribData.copy()

    df["windSpeed"] = np.sqrt(df["windU"] ** 2 + df["windV"] ** 2)
    df["temperature"] = df["temperature"] - 272.15

    df["globalRadiation"] = df["globalRadiation"].fillna(0)
    # Calculate the difference between consecutive rows
    df["Q"] = df["globalRadiation"].diff()
    df["Q"] = df["Q"] / 3600

    weather_df = df.copy()
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

    return df


def group_by_data_and_cache(df: pd.DataFrame):
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

    date = daily_data.index[0]

    daily_data.to_csv(f"energy_data/{date}.csv")


def main():
    api_key = os.getenv("KNMI_API_KEY")
    dataset_name = "harmonie_arome_cy40_p1"
    dataset_version = "0.2"
    logger.info(f"Fetching latest file of {dataset_name} version {dataset_version}")

    api = OpenDataAPI(api_token=api_key)

    # sort the files in descending order and only retrieve the first file
    params = {"maxKeys": 4, "orderBy": "created", "sorting": "desc"}
    response = api.list_files(dataset_name, dataset_version, params)
    # print(response)
    if "error" in response:
        logger.error(f"Unable to retrieve list of files: {response['error']}")
        sys.exit(1)

    # Filter files that end with '00.tar'
    filtered_files = [
        f for f in response["files"] if f.get("filename").endswith("00.tar")
    ]

    if not filtered_files:
        logger.error("No files ending with '00.tar' found")
        sys.exit(1)

    # Assuming files are already sorted by creation date in the response, get the latest
    latest_file = filtered_files[0].get("filename")
    logger.info(f"Latest file is: {latest_file}")

    # fetch the download url and download the file
    response = api.get_file_url(dataset_name, dataset_version, latest_file)
    download_file_from_temporary_download_url(
        response["temporaryDownloadUrl"], latest_file
    )
