"use client";

import {
    useLoadScript,
    GoogleMap,
    PolygonF
} from '@react-google-maps/api';

import React from 'react';
import { useMemo, useState } from 'react';

import { Button } from '@/components/ui/button';

import { getSolarPanel } from '@/lib/requests';

export default function Home() {
  const [lat, setLat] = useState(51.425722);
  const [lng, setLng] = useState(5.50894);
  const [satellites, setSatellites] = useState([]);

  const mapCenter = useMemo(() => ({ lat: lat, lng: lng }), [lat, lng]);

  const mapOptions = useMemo<google.maps.MapOptions>(
      () => ({
        disableDefaultUI: true,
        clickableIcons: true,
        scrollwheel: false,
        zoomControl: false,
        isFractionalZoomEnabled: false,
        mapTypeId: 'satellite',
        tilt: 0,
        }),
      []
  );

  const { isLoaded } = useLoadScript({
      googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY as string,  
  });
  
  if (!isLoaded) {
      return <p>Loading...</p>;
  }

  return (
      <div className='flex items-center justify-center'>
        <div className='flex flex-col items-center justify-center w-4/5 h-screen'>
            <GoogleMap
                mapContainerClassName='w-full h-4/5'
                options={mapOptions}
                center={mapCenter}
                zoom={20}
                onLoad={(map) => console.log('Map Loaded')}
              >
              
              {/* Each polygon corresponds to the polygon segmentation mask */}
              {/* <PolygonF
                paths={
                  [
                    { lat: 51.425722, lng: 5.509 },
                    { lat: 51.425722, lng: 5.50894 },
                    { lat: 51.425722, lng: 5.50894 },
                    { lat: 51.425726, lng: 5.50894 },
                  ]
                }
                onClick={
                  () => {
                    console.log('Polygon Clicked');
                  }
                }
              /> */}

            </GoogleMap>

            <Button
              className='rounded mt-4 w-full'
              variant='default'
              onClick={async () => {
                  const solarPanel = await getSolarPanel(lat, lng);
                  console.log(solarPanel); 
              }}
            >
              Scan block
            </Button>
        </div>
      </div>
  );
};