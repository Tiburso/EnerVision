"use server";

import { SolarPanel, BACKEND_URL } from "@/lib/types";

/**
 * Function sends a request to the backend service to get a segmentation analysis of the 
 * google maps static image
 * 
 * @param lat - The latitude of the center of the google maps static image.
 * @param lng - The longitude of the center of the google maps static image.
 * @returns - A promise of the solar panel segmentation analysis.
 */
export async function getSolarPanel(lat: number, lng: number): Promise<SolarPanel[]> {
    const center = `${lat},${lng}`;
    const url = new URL(`${BACKEND_URL}/segmentation`);

    url.searchParams.append("center", center);

    try {
        const response = await fetch(url.toString());

        const data = await response.json();

        return data["panels"].map((panel: any) => ({
            center: { lat: panel.center[0], lng: panel.center[1] },
            polygon: panel.polygon.map((point: number[]) => ({ lat: point[0], lng: point[1] })),
            type: panel.type
        }));

    } catch (error) {
        console.error(error);

        return [];
    }
}

/**
 * Function sends a request to the backend service to get the energy prediction for a solar panel.
 * 
 * @param lat - The latitude of the center of the solar panel.
 * @param lng - The longitude of the center of the solar panel.
 * @param type - The type of the solar panel.
 * @param area - The area of the solar panel.
 * @returns - A promise of the energy prediction ( a 48 sized array with predictions for each hour)
*/
export async function getEnergyPrediction(lat: number, lng: number, type: string, area: number): Promise<number[]> {
    const center = `${lat},${lng}`;
    const url = new URL(`${BACKEND_URL}/predictions`);

    url.searchParams.append("center", center);
    url.searchParams.append("type", type);

    try {
        // cache this response for 24 hours
        const response = await fetch(url.toString(), { next: { revalidate: 3600 * 24 } });

        const data = await response.json();

        // Append the key "today" and "tomorrow" into a single array
        const predictions = [...data["today"], ...data["tomorrow"]];

        // predictions scale linearly with the area of the solar panel
        // also have to filter out the negative values because they are not possible
        return predictions
            .map((value) => Math.max(0, value))
            .map((value) => value * area);

    } catch (error) {
        console.error(error);

        return [];
    }
}