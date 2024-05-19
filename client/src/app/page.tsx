"use client";

import {
    useLoadScript,
    GoogleMap,
    RectangleF,
} from '@react-google-maps/api';

import React from 'react';
import { useMemo, useState, useEffect } from 'react';

import { Button } from '@/components/ui/button';
import { Spinner } from '@/components/ui/spinner';
import { Searchbar } from '@/components/searchbar';
import { SolarPanelF } from '@/components/solarPanel';

import { getSolarPanel, SolarPanel } from '@/lib/requests';

export default function Home() {
  const [mapInstance, setMapInstance] = useState<google.maps.Map | null>(null);

  const [loading, setLoading] = useState(false);
  const [solarPanels, setSolarPanels] = useState<SolarPanel[]>([]);

  const libraries = useMemo(() => ['places'], []);
  const mapCenter = useMemo(() => ({ lat: 51.448388, lng: 5.490198 }), []);

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

      mapCenter.lat = lat;
      mapCenter.lng = lng;

      try {
        const results = await getSolarPanel(lat, lng);
        
        if (results.length === 0) {
          return;
        }

        // new solar panels must be old with new but not duplicated
        const newSolarPanels = results.filter((result) => {
          return solarPanels.every((solarPanel) => {
            return result && 
              solarPanel.center.lat() !== result.center.lat() &&
              solarPanel.center.lng() !== result.center.lng();
          });
        });

        setSolarPanels([...solarPanels, ...newSolarPanels] as SolarPanel[]);

        // sleep for 1 second
        // await new Promise((resolve) => setTimeout(resolve, 1000));
      } catch (error) {
        console.error(error);
      } finally {
        setLoading(false);
      } 
  }
  
  if (!isLoaded) {
      return <p>Loading...</p>;
  }

  return (
      <div className='flex items-center justify-center'>
        
        <div className='flex flex-col items-center justify-center w-4/5 h-screen'>
            <Searchbar className='w-3/4 relative my-5 shadow z-50'
              onClick={(val) => {
                if (!mapInstance) {
                  return;
                }

                mapInstance.setCenter(val);
              }}
            />

            {/* Add a spinner animation */}
            {loading && <Spinner />}

            <GoogleMap
                mapContainerClassName='w-full h-4/5'
                options={mapOptions}
                center={mapCenter}
                zoom={19}
                onLoad={handleMapLoad}
              >

              {loading && <RectangleF
                options={{
                  strokeColor: '#CCCCCC',
                  strokeOpacity: 0.5,
                  strokeWeight: 1,
                  fillColor: '#FFFFFF',
                  fillOpacity: 0.35,
                  clickable: false,
                  draggable: false,
                  editable: false,
                  visible: true,

                  // the bounds must be a 640pixel x 640pixel square
                  // the values are calculated based on the center of the map and the zoom level
                  bounds: {
                    north: mapCenter.lat + 0.00026759,
                    south: mapCenter.lat - 0.00026759,
                    east: mapCenter.lng + 0.00043,
                    west: mapCenter.lng - 0.00043,
                  },
                }}
              />}
              
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