import React from 'react';

import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';

const LineGraph: React.FC = () => {
    // data is a fetch from the backend

    const data = [
        {name: 'Page A', uv: 400},
        {name: 'Page B', uv: 100},
        {name: 'Page C', uv: 200},
        {name: 'Page D', uv: 250}
    ];

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
