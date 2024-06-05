"use client";

import {
    InfoWindowF,
    MarkerF,
} from '@react-google-maps/api';

import React, {useState, useMemo} from 'react';

import { 
    LineChart, 
    Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine
 } from 'recharts';
import { LatLng } from '@/lib/types';

interface MarkerProps {
    key: number
    center: LatLng
    energyPrediction: number[]
}

interface ChartData {
    name: string // date + hour
    energy: number
}

/**
 * The LineGraph component is a functional component that renders a line graph of the energy production.
 */
const Marker: React.FC<MarkerProps> = ({key, center, energyPrediction}) => {
    // data is a fetch from the backend
    const [isOpen, setIsOpen] = useState(false);
    
    const now = useMemo(() => new Date(), []);

    // Have to be in the same format as the data
    const currentHour = useMemo(() => {
        const hour = now.getHours();
        const date = now.toISOString().split('T')[0];
        return `${date} ${hour + 1}:00`;
    }
    , [now]);

    // data must be in the format of [{name: day + hour, energy: energy}]
    const data: ChartData[] = energyPrediction.map((energy, index) => {
        // If the index is less than 24, then it is today
        if (index < 24) {
            // Now get just the date without the time
            const date = now.toISOString().split('T')[0];

            return { name: `${date} ${index + 1}:00`, energy };
        }

        // Get tomorrows date
        const tomorrow = new Date(now);
        tomorrow.setDate(now.getDate() + 1);
        const date = tomorrow.toISOString().split('T')[0];

        return { name: `${date} ${index - 24 + 1}:00`, energy };
    });

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
                <div className='text-blue-800 rounded'>
                <p className='text-center font-bold'>Energy Production</p>
                    <LineChart
                        title='Energy Production' 
                        data={data} 
                        className='pr-7'
                        width={300} height={200}
                    >
                        <Line 
                            type="monotone" 
                            dataKey="energy" 
                            stroke="#8884d8" 
                        />
                        {/* Reference Line with the current hour make it a very light red dashd */}
                        <ReferenceLine 
                            x={currentHour} 
                            stroke='red' 
                            opacity={80} 
                            strokeDasharray="1 1"
                            label={{ value: 'Current Hour', position: 'insideTopRight' }}
                        />
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                            scale="auto"
                            dataKey="name"
                            label={{ value: 'Time', position: 'insideBottom' }}
                        />
                        <YAxis 
                            label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip />
                    </LineChart>
                </div>
            </InfoWindowF>}
        </>
    );
}

export { Marker };