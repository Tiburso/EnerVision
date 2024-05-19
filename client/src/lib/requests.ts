export interface SolarPanel {
    id: number
    center: google.maps.LatLng
    polygon: google.maps.LatLng[]
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export async function getSolarPanel(lat: number, lng: number): Promise<Partial<SolarPanel[]>> {
    const center = `${lat},${lng}`;

    try {
        const response = await fetch(`${BACKEND_URL}/segmentation?center=${center}`);

        const data = await response.json();

        return data["panels"].map((panel: any) => ({
            center: new google.maps.LatLng(panel.center[0], panel.center[1]),
            polygon: panel.polygon.map((point: number[]) => new google.maps.LatLng(point[0], point[1]))
        }));

    } catch (error) {
        console.error(error);

        return [];
    }
} 