from typing import Union
from dotenv import load_dotenv
import os

from fastapi import FastAPI

from losses import LossJaccard
from google_maps_to_3dbag import fetch_google_maps_static_image, pixels_to_lat_lng
from inference import segmentation_inference

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

app = FastAPI()


@app.get("/segmentation")
async def segment_solar_panel(center: str):
    image = fetch_google_maps_static_image(center, GOOGLE_MAPS_API_KEY)

    # Run the machine learning model here
    polygons, seg_centers, boundaries = segmentation_inference(image)

    # Convert the centers and the polygon values into real world coordinates
    seg_centers = [pixels_to_lat_lng(center, seg_center) for seg_center in seg_centers]
    polygons = [
        [pixels_to_lat_lng(center, point) for point in polygon] for polygon in polygons
    ]

    return {
        "polygons": polygons,
        "centers": seg_centers,
    }


# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}
