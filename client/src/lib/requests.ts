"use server";

import { SolarPanel, GaussianPrediction, BACKEND_URL } from "@/lib/types";

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
        }));

    } catch (error) {
        console.error(error);

        return [];
    }
}

function gaussian(x: number, mean: number, std: number, amplitude: number): number {
    return amplitude * Math.exp(-((x - mean) ** 2) / (2 * std ** 2));
}

export async function getEnergyPrediction(lat: number, lng: number, type: string, area: number): Promise<number[]> {
    const center = `${lat},${lng}`;
    const url = new URL(`${BACKEND_URL}/prediction`);

    url.searchParams.append("center", center);
    url.searchParams.append("type", type);

    try {
        const response = await fetch(url.toString());

        const data = await response.json();

        // Predictions is gonna be an array with 2 elements -> [mean, std, amplitude]
        // Each element corresponds to a day, (0) -> today, (1) -> tomorrow
        const predictions: number[] = data["predictions"].map((prediction: GaussianPrediction) => {
            const mean = prediction.mean;
            const std = prediction.std;
            const amplitude = prediction.amplitude;

            return Array.from({ length: 24 }, (_, i) => gaussian(i, mean, std, amplitude));
        });

        // predictions scale linearly with the area of the solar panel
        return predictions.flat().map((value) => value * area);
    } catch (error) {
        console.error(error);

        return [];
    }
}