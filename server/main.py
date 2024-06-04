import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.google_maps_api import (
    load_google_maps_api,
    unload_google_maps_api,
    fetch_google_maps_static_image,
    pixels_to_lat_lng,
    fetch_roof_information,
)

from server.weather_data_api import get_predicted_data

from server.inference import (
    segmentation_inference,
    energy_prediction,
    load_models,
    clean_up_models,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_google_maps_api()
    load_models()

    yield

    clean_up_models()
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
    polygons, seg_centers, pvtypes = segmentation_inference(image)

    # Convert the centers and the polygon values into real world coordinates
    seg_centers = [pixels_to_lat_lng(center, seg_center) for seg_center in seg_centers]

    polygons = [
        [pixels_to_lat_lng(center, point[0]) for point in polygon]
        for polygon in polygons
    ]

    panels = [
        {"polygon": polygon, "center": seg_center, "type": pvtype}
        for polygon, seg_center, pvtype in zip(polygons, seg_centers, pvtypes)
    ]

    return {
        "panels": panels,
    }


@app.get("/predictions")
async def predict_pv_energy(center: str, type: str):
    # Get the weather forecast for the next 2 days
    weather_data = get_predicted_data()

    # Get the azimuth and the tilt of the solar panels from the google api
    roof_data = fetch_roof_information(center)

    if roof_data is None:
        raise HTTPException(status_code=404, detail="Roof data not found")

    # Extend the weather data with the roof data
    weather_data["azimuth"] = roof_data["azimuth"]
    weather_data["tilt"] = roof_data["tilt"]
    weather_data["module_type"] = type

    # Run the inference model for the energy production
    predictions = energy_prediction(weather_data)

    # Return the normal parameters for the today and tomorrow
    return predictions
