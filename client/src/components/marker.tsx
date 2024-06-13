"use client";

import {
    InfoWindowF,
    MarkerF,
} from '@react-google-maps/api';

import React, {useState, useCallback} from 'react';

import { 
    LineChart, 
    Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine
 } from 'recharts';
import { LatLng } from '@/lib/types';

import { getEnergyPrediction } from '@/lib/requests';

interface MarkerProps {
    key: number
    center: LatLng
    type: string
    area: number
}

interface ChartData {
    name: string // date + hour
    energy: number
}

/**
 * 
 * @param key The key of the marker for react
 * @param center The center of the solar panel that represents the marker
 * @param type The type of the solar panel
 * @param area The area of the solar panel
 */
const Marker: React.FC<MarkerProps> = ({key, center, type, area}) => {
    const [chartData, setChartData] = useState<ChartData[]>([]);
    const [formattedHour, setFormattedHour] = useState<string>('');
    const [isOpen, setIsOpen] = useState(false);
    

    // Fetch the energy prediction for the solar panel
    // This callback will be called when the marker is clicked
    const fetchEnergyPrediction = useCallback(async () => {
        
        // Set the modal open because it was lagging the first time
        setIsOpen(!isOpen);

        // Get the current date to set the Xaxis of the graph
        const now = new Date()

        const hour = now.getHours();
        const date = now.toISOString().split('T')[0];
        const formatedHour = `${date} ${hour + 1}:00`;

        const prediction = await getEnergyPrediction(center.lat, center.lng, type, area);
        
        // Map each of the indexes to a specific day
        // (today or tomorrow)
        const data = prediction.map((energy, index) => {
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
    
            // data must be in the format of [{name: day + hour, energy: energy}]
            return { name: `${date} ${index - 24 + 1}:00`, energy };
        });

        setFormattedHour(formatedHour);
        setChartData(data);
    }, [center, type, area, isOpen]);

    return (
        <>
            <MarkerF
                key={key}
                position={center}
                onClick={fetchEnergyPrediction}
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
                        data={chartData} 
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
                            x={formattedHour} 
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