import { getEnergyPrediction } from '@/lib/requests';
import React from 'react';

import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';

interface LineGraphProps {
    lat: number
    lng: number
    type: string
    area: number
}

/**
 * The LineGraph component is a functional component that renders a line graph of the energy production.
 */
const LineGraph: React.FC<LineGraphProps> = async ({lat, lng, type, area}) => {
    // data is a fetch from the backend
    const data = await getEnergyPrediction(lat, lng, type, area);

    return (
        <div className='text-blue-800 border-2 border-gray-100 rounded'>
            <p className='text-center font-bold'>Energy Production</p>
            <LineChart
                title='Energy Production' 
                width={200} 
                height={200} 
                data={data} 
                className='pr-7'
            >
                <Line type="monotone" dataKey="uv" stroke="#8884d8" />
                <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
            </LineChart>
        </div>
    );
}

export { LineGraph };
