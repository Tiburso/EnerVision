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
        return predictions
            .map((value) => Math.min(0, value))
            .map((value) => value * area);

    } catch (error) {
        console.error(error);

        return [];
    }
}