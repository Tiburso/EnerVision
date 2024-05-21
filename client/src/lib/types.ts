export interface SolarPanel {
    id: number
    center: google.maps.LatLng
    polygon: google.maps.LatLng[]
}

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export { BACKEND_URL }