from typing import Union

from fastapi import FastAPI

from google_maps_to_3dbag import google_to_lat_lng

app = FastAPI()


@app.get("/segmentation")
async def segment_solar_panel(center: str):
    lat, lng = google_to_lat_lng(center)

    return {"panels": [{"lat": lat, "lng": lng}]}


# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}
