import googlemaps.client
import requests
import numpy as np
import googlemaps
import logging

from PIL import Image

from dotenv import load_dotenv
import os

ZOOM = 20
IMAGE_SIZE = 640
gmaps: googlemaps.client.Client = None


def load_google_maps_api():
    global gmaps

    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    gmaps = googlemaps.Client(key=api_key)


def unload_google_maps_api():
    global gmaps
    gmaps = None


def check_cache(center: str):
    """This function will check if the corresponding image is already in the cache.
    If it is, it will return the image. If it is not, it will return None.

    Args:
        center (str): The center of the image.

    Returns:
        Image: The image if it is in the cache, None otherwise.
    """

    # Currently the "cache" is just gonna be a folder in the same directory as the script

    # Check if the cache folder exists
    if not os.path.exists("cache"):
        os.makedirs("cache")

    # Check if the image is already in the cache
    if os.path.exists(f"cache/{center}.png"):
        return Image.open(f"cache/{center}.png").convert("RGB")
    else:
        return None


def read_image(image, center: str) -> Image:
    """Read the image by savinng it as a file and then deleting
    that file afterwards

    Args:
        image: The image to be saved.
        center (str): The center of the image.

    Returns:
        Image: The image.
    """

    with open(f"cache/{center}.png", "wb") as f:
        for chunk in image:
            f.write(chunk)

    # Read the image from the cache
    img = Image.open(f"cache/{center}.png").convert("RGB")

    # Remove the image from the cache
    os.remove(f"cache/{center}.png")

    return img


def fetch_google_maps_static_image(center: str) -> Image:
    """Fetch the static image from Google Maps.

    Args:
        center (str): The center of the image.

    Returns:
        Image: The image.
    """

    global ZOOM, IMAGE_SIZE, gmaps
    maptype = "satellite"

    image = check_cache(center)

    if image is not None:
        return image

    image = gmaps.static_map(
        center=center,
        zoom=ZOOM,
        size=IMAGE_SIZE,
        maptype=maptype,
    )

    image = read_image(image, center)

    return image


def pixels_to_lat_lng(center: str, pixel: tuple) -> tuple:
    """Convert the pixel to latitude and longitude.

    Args:
        center (str): The center of the image.
        pixel (tuple): The pixel to be converted.

    Returns:
        tuple: The latitude and longitude.
    """

    global ZOOM, IMAGE_SIZE

    x, y = pixel
    lat, lng = map(float, center.split(","))

    parallelMultiplier = np.cos(lat * np.pi / 180)
    degreesPerPixelX = 360 / np.power(2, ZOOM + 8)
    degreesPerPixelY = 360 / np.power(2, ZOOM + 8) * parallelMultiplier
    pointLat = lat - degreesPerPixelY * (y - IMAGE_SIZE / 2)
    pointLng = lng + degreesPerPixelX * (x - IMAGE_SIZE / 2)

    return pointLat, pointLng


def fetch_roof_information(center: str) -> dict:
    """Fetch the roof information from the building with the given center
    from the google maps Solar API

    Args:
        center (str): The center of the solar panel which belongs to the building

    Returns:
        dict: The roof slope and azimuth in a dictionary
    """

    api_key = gmaps.key
    url = "https://solar.googleapis.com/v1/buildingInsights:findClosest"

    required_quality = "LOW"
    lat, lng = map(float, center.split(","))

    params = {
        "location.latitude": lat,
        "location.longitude": lng,
        "requiredQuality": required_quality,
        "key": api_key,
    }

    logging.info(f"Fetching roof information for {center}")

    res = requests.get(url, params=params)

    if res.status_code != 200:
        logging.error(f"Error fetching roof information: {res.text}")
        return None

    data = res.json()

    # roofSegmentStats is a list and we only want the first value
    roofSegmentStats = data["solarPotential"]["roofSegmentStats"][0]

    # From data we can extract the azimuth and the tilt of the roof
    return {
        "azimuth": roofSegmentStats["azimuthDegrees"],
        "tilt": roofSegmentStats["pitchDegrees"],
    }
