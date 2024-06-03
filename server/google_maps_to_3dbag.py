import googlemaps.client
import requests
import numpy as np
from pyproj import CRS, Transformer
import googlemaps

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


def lat_lng_to_amerfoort_rd(coords: tuple):
    # The default coordinate reference system is Amersfoort / RD New + NAP height,
    # EPSG:7415 (https://www.opengis.net/def/crs/EPSG/0/7415)
    # bbox is a list of 4 coordinates (min x min y max x max y)
    wgs84 = CRS("EPSG:4326")
    rd_new = CRS("EPSG:7415")

    # Create a transformer
    transformer = Transformer.from_crs(wgs84, rd_new)

    x, y = transformer.transform(coords[0], coords[1])

    return x, y


def get_3dbag_information(bbox: list):
    url = "https://api.3dbag.nl//collections/pand/items"

    # Convert bbox into array of numbers and then add it into the url
    bbox = ",".join(map(str, bbox))
    url += f"?bbox={bbox}"

    res = requests.get(url)

    res.raise_for_status()

    data = res.json()

    return data["features"]
