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
    const earthRadiusSquared = 6371 * 6371;

    const convertToRadians = (degrees: number): number => {
        return degrees * Math.PI / 180;
    }
    
    // Initialize the total cross product to 0
    let area = 0;

    for (let i = 0; i < vertices.length - 1; i++) {
        const p1 = vertices[i];
        const p2 = vertices[i + 1];

        area += convertToRadians(p2.lng - p1.lng) * (2 + Math.sin(convertToRadians(p1.lat)) + Math.sin(convertToRadians(p2.lat)));
    }

    area = area * earthRadiusSquared / 2;

    // Convert the total cross product to the area by dividing by 2 and taking the absolute value
    return Math.abs(area);   
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