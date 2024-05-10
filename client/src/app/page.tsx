"use client";

import {
    useLoadScript,
    GoogleMap,
} from '@react-google-maps/api';

import React from 'react';
import { useMemo, useState, useEffect } from 'react';

import { Button } from '@/components/ui/button';
import { Spinner } from '@/components/ui/spinner';
import { Searchbar } from '@/components/searchbar';
import { SolarPanelF } from '@/components/SolarPanel';

import { getSolarPanel, SolarPanel } from '@/lib/requests';

export default function Home() {
  const [mapInstance, setMapInstance] = useState<google.maps.Map | null>(null);

  const [loading, setLoading] = useState(false);
  const [solarPanels, setSolarPanels] = useState<SolarPanel[]>([]);

  const libraries = useMemo(() => ['places'], []);
  const mapCenter = useMemo(() => ({ lat: 51.425722, lng: 5.50894 }), []);

  const mapOptions = useMemo<google.maps.MapOptions>(
      () => ({
        disableDefaultUI: true,
        clickableIcons: true,
        scrollwheel: true,
        zoomControl: true,
        isFractionalZoomEnabled: false,
        mapTypeId: 'satellite',
        tilt: 0,
        }),
      []
  );

  const { isLoaded } = useLoadScript({
      googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY as string, 
      libraries: libraries as any,
  });

  const handleMapLoad = (map: google.maps.Map) => {
    setMapInstance(map);
  };

  const scanArea = async () => {
      setLoading(true);

      if (!mapInstance) {
        return;
      }

      const center = mapInstance.getCenter();

      if (!center) {
        return;
      }

      const lat = center.lat();
      const lng = center.lng();

      try {
        const results = await getSolarPanel(lat, lng);
        setSolarPanels([...solarPanels, ...results] as SolarPanel[]);
      } catch (error) {
        console.error(error);
      } 

      setLoading(false);
  }
  
  if (!isLoaded) {
      return <p>Loading...</p>;
  }

  return (
      <div className='flex items-center justify-center'>
        
        <div className='flex flex-col items-center justify-center w-4/5 h-screen'>
            <Searchbar className='w-3/4 relative my-5 shadow'/>

            {/* Add a spinner animation */}
            {loading? <Spinner /> : null}

            <GoogleMap
                mapContainerClassName='w-full h-4/5'
                options={mapOptions}
                center={mapCenter}
                zoom={20}
                onLoad={handleMapLoad}
              >
              
              {solarPanels.map((solarPanel, index) => (
                  <SolarPanelF
                    key={index}
                    center={solarPanel.center}
                    polygon={solarPanel.polygon}
                  />
              ))}

            </GoogleMap>

            <Button
              className='rounded w-full my-5'
              variant='default'
              onClick={scanArea}
              disabled={loading}
            >
              Scan Area
            </Button>
        </div>
      </div>
  );
};