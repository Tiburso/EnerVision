"use server";

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
        <LineChart width={200} height={200} data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <Line type="monotone" dataKey="uv" stroke="#8884d8" />
            <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
        </LineChart>
    );
}

export { LineGraph };
