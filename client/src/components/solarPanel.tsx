import {
    PolygonF,
    InfoWindowF,
    MarkerF,
} from '@react-google-maps/api';

import React from 'react';
import { useState } from 'react';

import { LineGraph } from './graph';
import { SolarPanel, LatLng } from '@/lib/types';

interface SolarPanelProps {
    key: number
    solarPanel: SolarPanel
}

/** 
 * The SolarPanelF component is a functional component that renders a solar panel on the embedded google maps.
 * 
 * @param key - The key of the solar panel.
 * @param solarPanel - The solar panel object.
*/
const SolarPanelF: React.FC<SolarPanelProps> = ({ key, solarPanel }) => {
    const [isOpen, setIsOpen] = useState(false);

    const calculateArea = (vertices: LatLng[]): number => {
        // Helper function inside calculatePolygonArea to calculate the cross product of two LatLng coordinates
        const crossProduct = (a: LatLng, b: LatLng): number => {
            return a.lat * b.lng - a.lng * b.lat;
        };

        // Use reduce to sum the cross products between consecutive pairs of LatLng
        const totalCrossProduct = vertices.reduce((sum, current, index, arr) => {
            const nextIndex = (index + 1) % arr.length; // Wrap around to the first element after the last
            return sum + crossProduct(current, arr[nextIndex]);
        }, 0);

        // Convert the total cross product to the area by dividing by 2 and taking the absolute value
        return Math.abs(totalCrossProduct / 2);
    }
    
    return (
        <>
            <MarkerF
                key={key}
                position={solarPanel.center}
                onClick={() => setIsOpen(!isOpen)}
            />
            
            {isOpen && 
            <InfoWindowF
                key={key}
                position={solarPanel.center}
                zIndex={1}
                onCloseClick={() => setIsOpen(!isOpen)}
            >   
                <LineGraph 
                    lat={solarPanel.center.lat} 
                    lng={solarPanel.center.lng}
                    type={solarPanel.type}
                    area={calculateArea(solarPanel.polygon)}
                />
            </InfoWindowF>}

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