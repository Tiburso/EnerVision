"use client";

import {
    useJsApiLoader,
    GoogleMap,
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

  const { isLoaded } = useJsApiLoader({
      googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY as string,  
  });
  
  if (!isLoaded) {
      return <p>Loading...</p>;
  }

  return (
      <div>
          <GoogleMap
              mapTypeId={google.maps.MapTypeId.SATELLITE}
              options={mapOptions}
              center={mapCenter}
              zoom={20}
              mapContainerStyle={{ width: '800px', height: window.innerHeight }}
              onLoad={(map) => console.log('Map Loaded')}
          >
          </GoogleMap>

          <Button
            variant='secondary'
            onClick={() => {
                console.log('Button clicked');
            }}
          >
            Click me
          </Button>
      </div>
  );
};