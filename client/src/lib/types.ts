/**
 * Types and constants used throughout the application
 */
export interface LatLng {
    lat: number
    lng: number
}

export interface SolarPanel {
    id: number
    center: LatLng
    polygon: LatLng[]
    type: "monocrystalline" | "polycrystalline"
}

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export { BACKEND_URL }