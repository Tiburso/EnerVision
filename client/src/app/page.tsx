"use client";

import {
    useLoadScript,
    GoogleMap,
    PolygonF
} from '@react-google-maps/api';

import React from 'react';
import { useMemo, useState } from 'react';

import { Button } from '@/components/ui/button';

import { getSolarPanel, SolarPanel } from '@/lib/requests';

export default function Home() {
  const [lat, setLat] = useState(51.425722);
  const [lng, setLng] = useState(5.50894);
  const [mapInstance, setMapInstance] = useState<google.maps.Map | null>(null);

  const [loading, setLoading] = useState(false);
  const [solarPanels, setSolarPanels] = useState<SolarPanel[]>([]);

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

  const handleMapLoad = (map: google.maps.Map) => {
    setMapInstance(map);
  };

  const handleCenterChange = () => {
    if (mapInstance) {
      const center = mapInstance.getCenter();

      if (center) {
        setLat(center.lat());
        setLng(center.lng());
      }
    }
  };

  const { isLoaded } = useLoadScript({
      googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY as string,  
  });

  const scanArea = async () => {
      setLoading(true);
      try {
        const results = await getSolarPanel(lat, lng);
        setSolarPanels([...solarPanels, ...results] as SolarPanel[]);
      } catch (error) {
        console.error(error);
      } 

      setLoading(false);
  }

  const Spinner = () => (
  <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-e-transparent align-[-0.125em] text-surface motion-reduce:animate-[spin_1.5s_linear_infinite] dark:text-white fixed z-50"
      role="status">
  </div>
  );
  
  if (!isLoaded) {
      return <p>Loading...</p>;
  }

  return (
      <div className='flex items-center justify-center'>
        
        <div className='flex flex-col items-center justify-center w-4/5 h-screen'>
            {/* Add a spinner animation */}
            {loading? <Spinner /> : null}

            <GoogleMap
                mapContainerClassName='w-full h-4/5'
                options={mapOptions}
                center={mapCenter}
                zoom={20}
                onLoad={handleMapLoad}
                onCenterChanged={handleCenterChange}
              >
              
              {/* Each polygon corresponds to the polygon segmentation mask */}
              {solarPanels.map((solarPanel, index) => (
                  <PolygonF
                      key={index}
                      path={solarPanel.polygon}
                      options={{
                          strokeColor: '#FF0000',
                          strokeOpacity: 0.8,
                          strokeWeight: 2,
                          fillColor: '#FF0000',
                          fillOpacity: 0.35,
                          clickable: true,
                          draggable: false,
                          editable: false,
                          visible: true,
                      }}
                  />
              ))}

            </GoogleMap>

            <Button
              className='rounded mt-4 w-full'
              variant='default'
              onClick={scanArea}
              disabled={loading}
            >
              Scan block
            </Button>
        </div>
      </div>
  );
};