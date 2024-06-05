import { getEnergyPrediction } from '@/lib/requests';
import React from 'react';

import { 
    LineChart, 
    Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
 } from 'recharts';

interface LineGraphProps {
    lat: number
    lng: number
    type: string
    area: number
}

interface ChartData {
    name: string // date + hour
    energy: number
}

// Set the value for the caching time to 24 hours
export const cacheTime = 24 * 60 * 60 * 1000;

/**
 * The LineGraph component is a functional component that renders a line graph of the energy production.
 */
const LineGraph: React.FC<LineGraphProps> = async ({lat, lng, type, area}) => {
    // data is a fetch from the backend
    const energyPrediction = await getEnergyPrediction(lat, lng, type, area);

    // data must be in the format of [{name: day + hour, energy: energy}]
    const data: ChartData[] = energyPrediction.map((energy, index) => {
        const now = new Date();

        // If the index is less than 24, then it is today
        if (index < 24) {
            // Now get just the date without the time
            const date = now.toISOString().split('T')[0];

            return { name: `${date} ${index}:00`, energy };
        }

        // Get tomorrows date
        const tomorrow = new Date(now);
        tomorrow.setDate(now.getDate() + 1);
        const date = tomorrow.toISOString().split('T')[0];

        return { name: `${date} ${index - 24}:00`, energy };
    
    });

    return (
        <div className='text-blue-800 border-2 border-gray-100 rounded'>
            <p className='text-center font-bold'>Energy Production</p>
            <ResponsiveContainer width="100%" height="100%">
                <LineChart
                    title='Energy Production' 
                    width={200} 
                    height={200} 
                    data={data} 
                    className='pr-7'
                >
                    <Line 
                        type="monotone" 
                        dataKey="energy" 
                        stroke="#8884d8" 
                    />
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

export { LineGraph };
