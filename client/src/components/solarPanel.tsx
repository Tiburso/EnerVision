import {
    PolygonF,
    InfoWindowF,
    MarkerF,
} from '@react-google-maps/api';

import React from 'react';
import { useState } from 'react';

import { LineGraph } from './graph';
import { LatLng } from '@/lib/types';

interface SolarPanelProps {
    key: number
    center: LatLng
    polygon: LatLng[]
}

/** 
 * The SolarPanelF component is a functional component that renders a solar panel on the embedded google maps.
 * 
 * @param key - The key of the solar panel.
 * @param center - The center of the solar panel in lat lng.
 * @param polygon - The polygon path of the solar panel.
*/
const SolarPanelF: React.FC<SolarPanelProps> = ({ key, center, polygon, }) => {
    const [isOpen, setIsOpen] = useState(false);
    
    return (
        <>
            <MarkerF
                key={key}
                position={center}
                onClick={() => setIsOpen(!isOpen)}
            />
            
            {isOpen && 
            <InfoWindowF
                key={key}
                position={center}
                zIndex={1}
                onCloseClick={() => setIsOpen(!isOpen)}
            >   
                <LineGraph />
            </InfoWindowF>}

            <PolygonF
                key={key}
                path={polygon}
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