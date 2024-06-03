from typing import Union
from dotenv import load_dotenv
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.google_maps_to_3dbag import (
    load_google_maps_api,
    unload_google_maps_api,
    fetch_google_maps_static_image,
    pixels_to_lat_lng,
)
from server.inference import segmentation_inference, load_model, clean_up_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_google_maps_api()
    load_model()

    yield

    clean_up_model()
    unload_google_maps_api()


origins = ["localhost", os.getenv("FRONTEND_URL")]

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/segmentation")
async def segment_solar_panel(center: str):
    image = fetch_google_maps_static_image(center)

    # Run the machine learning model here
    polygons, seg_centers, boundaries = segmentation_inference(image)

    # Convert the centers and the polygon values into real world coordinates
    seg_centers = [pixels_to_lat_lng(center, seg_center) for seg_center in seg_centers]

    polygons = [
        [pixels_to_lat_lng(center, point[0]) for point in polygon]
        for polygon in polygons
    ]

    panels = [
        {"polygon": polygon, "center": seg_center}
        for polygon, seg_center in zip(polygons, seg_centers)
    ]

    return {
        "panels": panels,
    }
