"use client";

import {
  useJsApiLoader,
  GoogleMap,
  MarkerF,
  CircleF,
} from '@react-google-maps/api';

import { useMemo, useState } from 'react';

import styles from './page.module.css';

export default function Home() {
  const [lat, setLat] = useState(27.672932021393862);
  const [lng, setLng] = useState(85.31184012689732);

  const libraries = useMemo(() => ['places'], []);
  const mapCenter = useMemo(() => ({ lat: lat, lng: lng }), [lat, lng]);

  const mapOptions = useMemo<google.maps.MapOptions>(
      () => ({
        disableDefaultUI: true,
        clickableIcons: true,
        scrollwheel: false,
        }),
      []
  );

  const { isLoaded } = useJsApiLoader({
      id: 'google-map-script',
      googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY as string,
  });

  console.log(process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY);

  if (!isLoaded) {
      return <p>Loading...</p>;
  }

  return (
      <div className={styles.homeWrapper}>
      
          <GoogleMap
              options={mapOptions}
              zoom={20}
              center={mapCenter}
              mapTypeId={google.maps.MapTypeId.SATELLITE}
              mapContainerStyle={{ width: '800px', height: '800px' }}
              onLoad={(map) => console.log('Map Loaded')}
          >
              <MarkerF
                  position={mapCenter}
                  onLoad={() => console.log('Marker Loaded')}
              />

              {/* {[1000, 2500].map((radius, idx) => {
                  return (
                      <CircleF
                      key={idx}
                      center={mapCenter}
                      radius={radius}
                      onLoad={() => console.log('Circle Load...')}
                          options={{
                              fillColor: radius > 1000 ? 'red' : 'green',
                              strokeColor: radius > 1000 ? 'red' : 'green',
                              strokeOpacity: 0.8,
                          }}
                      />
                  );
              })} */}
          </GoogleMap>
      </div>
  );
};