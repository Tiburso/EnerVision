import requests
import json
import numpy as np
from pyproj import CRS, Transformer

from PIL import Image

from dotenv import load_dotenv
import os

ZOOM = 20
IMAGE_SIZE = 640


def check_cache(center: str):
    """This function will check if the corresponding image is already in the cache.
    If it is, it will return the image. If it is not, it will return None."""

    # Currently the "cache" is just gonna be a folder in the same directory as the script

    # Check if the cache folder exists
    if not os.path.exists("cache"):
        os.makedirs("cache")

    # Check if the image is already in the cache
    if os.path.exists(f"cache/{center}.png"):
        return Image.open(f"cache/{center}.png")
    else:
        return None


def save_image_to_cache(image, center: str):
    with open(f"cache/{center}.png", "wb") as f:
        f.write(image.content)

    # Read the image from the cache
    return Image.open(f"cache/{center}.png")


def fetch_google_maps_static_image(center: str, key: str):
    global ZOOM, IMAGE_SIZE
    maptype = "satellite"
    url = "https://maps.googleapis.com/maps/api/staticmap"
    size = f"{IMAGE_SIZE}x{IMAGE_SIZE}"

    image = check_cache(center)

    if image:
        return image

    res = requests.get(
        url,
        params={
            "center": center,
            "zoom": ZOOM,
            "size": size,
            "maptype": maptype,
            "key": key,
        },
    )

    res.raise_for_status()

    image = save_image_to_cache(res, center)

    return image


def pixels_to_lat_lng(center: str, pixel: tuple):
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


if __name__ == "__main__":
    load_dotenv()

    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    center = "51.425722,5.50894"

    image = fetch_google_maps_static_image(center, GOOGLE_MAPS_API_KEY)

    # Run the machine learning model here
    lower_left = pixels_to_lat_lng(center, (0, IMAGE_SIZE))
    upper_right = pixels_to_lat_lng(center, (IMAGE_SIZE, 0))

    # Only features that have a geometry that intersects the
    # bounding box are selected. The bounding box is provided as four numbers:
    #
    # Lower left corner, coordinate axis 1 (min. x)
    # Lower left corner, coordinate axis 2 (min. y)
    # Upper right corner, coordinate axis 1 (max. x)
    # Upper right corner, coordinate axis 2 (max. y)
    new_lower_left = lat_lng_to_amerfoort_rd(lower_left)
    new_upper_right = lat_lng_to_amerfoort_rd(upper_right)

    bbox = [
        new_lower_left[0],
        new_lower_left[1],
        new_upper_right[0],
        new_upper_right[1],
    ]

    data = get_3dbag_information(bbox)

    print(data[0]["CityObjects"][list(data[0]["CityObjects"].keys())[0]])
