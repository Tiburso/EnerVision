"use server";

import { SolarPanel, BACKEND_URL } from "@/lib/types";

export async function getSolarPanel(lat: number, lng: number): Promise<Partial<SolarPanel[]>> {
    const center = `${lat},${lng}`;

    try {
        const response = await fetch(`${BACKEND_URL}/segmentation?center=${center}`);

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