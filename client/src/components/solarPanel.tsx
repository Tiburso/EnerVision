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

const SolarPanelF: React.FC<SolarPanelProps> = ({ key, center, polygon, }) => {
    const [isOpen, setIsOpen] = useState(false);
    const data = [{name: 'Page A', uv: 400}, {name: 'Page B', uv: 100}];
    
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