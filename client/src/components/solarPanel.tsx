import {
    PolygonF,
} from '@react-google-maps/api';

import { Marker } from '@/components/marker';
import { SolarPanel, LatLng } from '@/lib/types';
import { useMemo } from 'react';

interface SolarPanelProps {
    key: number
    solarPanel: SolarPanel
}


/**
 * Calculate the area of a polygon given its vertices using the shoelace formula.
 * Converts first from latitude and longitude to cartesian coordinates.
 * 
 * @param vertices - The vertices of the polygon.
 * @returns The area of the polygon.
 */
const calculateArea = (vertices: LatLng[]): number => {
    const earthRadiusSquared = 6371009 // Earth's radius in meters
    
    const latLngToCartesian = (lat: number, lng: number, radiusSquared: number) => {
        // Convert the latitude and longitude to radians
        const latRadians = lat * Math.PI / 180;
        const lngRadians = lng * Math.PI / 180;

        // Calculate the x and y coordinates of the point on the sphere
        return {
            x: radiusSquared * Math.cos(latRadians) * Math.cos(lngRadians),
            y: radiusSquared * Math.cos(latRadians) * Math.sin(lngRadians),
        };
    } 

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
    const area = useMemo(() => calculateArea(solarPanel.polygon), [solarPanel.polygon]);

    return (
        <>
            <Marker
                key={key}
                center={solarPanel.center}
                type={solarPanel.type}
                area={area}
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