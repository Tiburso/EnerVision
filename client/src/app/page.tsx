"use client";

import {
    useLoadScript,
    GoogleMap,
    PolygonF
} from '@react-google-maps/api';

import React from 'react';
import { Button } from '@/components/ui/button';

import { useMemo, useState } from 'react';

export default function Home() {
  const [lat, setLat] = useState(27.672932021393862);
  const [lng, setLng] = useState(85.31184012689732);

  const mapCenter = useMemo(() => ({ lat: lat, lng: lng }), [lat, lng]);

  const mapOptions = useMemo<google.maps.MapOptions>(
      () => ({
        disableDefaultUI: true,
        clickableIcons: true,
        scrollwheel: false,
        zoomControl: false,
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
        <div className='flex flex-col items-center justify-center w-3/5 h-screen'>
            <GoogleMap
                mapContainerClassName='w-full h-4/5'
                mapTypeId={google.maps.MapTypeId.SATELLITE}
                options={mapOptions}
                center={mapCenter}
                zoom={20}
                onLoad={(map) => console.log('Map Loaded')}
              >
              
              {/* Each polygon corresponds to the polygon segmentation mask */}
              <PolygonF
                paths={
                  [
                    { lat: 27.672932021393862, lng: 85.31184012689732 },
                    { lat: 27.672932021393862, lng: 85.31184012689732 },
                    { lat: 27.672932021393862, lng: 85.31184012689732 },
                    { lat: 27.672932021393862, lng: 85.31184012689732 },
                  ]
                }
                onClick={
                  () => {
                    console.log('Polygon Clicked');
                  }
                }
              />

            </GoogleMap>

            <Button
              className='rounded mt-4 w-full'
              variant='default'
              onClick={() => {
                  setLat(27.672932021393862);
                  setLng(85.31184012689732);

                  console.log('Map Centered');
                  console.log('Latitude:', lat);
                  console.log('Longitude:', lng);
              }}
            >
              Scan block
            </Button>
        </div>
      </div>
  );
};