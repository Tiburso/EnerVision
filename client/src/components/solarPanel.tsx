import {
    PolygonF,
} from '@react-google-maps/api';

import { Marker } from '@/components/marker';
import { SolarPanel, LatLng } from '@/lib/types';
import { getEnergyPrediction } from '@/lib/requests';
import { useEffect, useState } from 'react';

interface SolarPanelProps {
    key: number
    solarPanel: SolarPanel
}

const calculateArea = (vertices: LatLng[]): number => {
    const earthRadiusSquared = 6371009 // Earth's radius in meters

    const deg2rad = (degrees: number): number => {
        return degrees * Math.PI / 180;
    }

    const latLngToCartesian = (lat: number, lng: number, latDist: number): { x: number; y: number } => {
        const phi = deg2rad(lat);
        const lambda = deg2rad(lng);
        const x = lambda * latDist * Math.cos(phi);
        const y = phi * latDist;
        return { x, y };
    };
    
    // Initialize the total cross product to 0
    let area = 0;

    // Iterate over the vertices of the polygon
    for (let i = 0; i < vertices.length; i++) {
        const p1 = vertices[i];
        const p2 = vertices[(i + 1) % vertices.length];

        // Convert the latitude and longitude to cartesian coordinates
        const p1Cartesian = latLngToCartesian(p1.lat, p1.lng, earthRadiusSquared);
        const p2Cartesian = latLngToCartesian(p2.lat, p2.lng, earthRadiusSquared);

        // Calculate the cross product of the two points
        area += p1Cartesian.x * p2Cartesian.y - p1Cartesian.y * p2Cartesian.x;
    }

    return Math.abs(area / 2);
}

/** 
 * The SolarPanelF component is a functional component that renders a solar panel on the embedded google maps.
 * 
 * @param key - The key of the solar panel.
 * @param solarPanel - The solar panel object.
*/
function SolarPanelF({ key, solarPanel } : SolarPanelProps) {
    const [energyPrediction, setEnergyPrediction] = useState<number[]>([]);
    const area = calculateArea(solarPanel.polygon);

    useEffect(() => {
        getEnergyPrediction(solarPanel.center.lat, solarPanel.center.lng, solarPanel.type, area)
            .then(setEnergyPrediction)
            .catch(console.error);
    }, [solarPanel, area]);

    return (
        <>
            <Marker
                key={key}
                center={solarPanel.center}
                energyPrediction={energyPrediction}
            />

            <PolygonF
                key={key}
                path={solarPanel.polygon}
                options={{
                    strokeColor: '#FF0000',
                    strokeOpacity: 0.8,
                    strokeWeight: 2,
                    fillColor: '#FF0000',
                    fillOpacity: 0.35,
                    clickable: false,
                    draggable: false,
                    editable: false,
                    visible: true,
                }}
            />
        </>
    );
}

export { SolarPanelF }