import {
    PolygonF,
    InfoWindowF,
    MarkerF,
} from '@react-google-maps/api';

import React from 'react';
import { useState } from 'react';

import { LineGraph } from './graph';
import { SolarPanel } from '@/lib/types';

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
                <LineGraph />
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