import {
    PolygonF,
    InfoWindowF,
    MarkerF,
} from '@react-google-maps/api';

interface SolarPanelProps {
    key: number
    center: google.maps.LatLng
    polygon: google.maps.LatLng[]
}

const SolarPanelF: React.FC<SolarPanelProps> = ({ key, center, polygon, }) => {
    return (
        <>
            <MarkerF
                key={key}
                position={center}
            />
            <PolygonF
                key={key}
                path={polygon}
                options={{
                    strokeColor: '#FF0000',
                    strokeOpacity: 0.8,
                    strokeWeight: 2,
                    fillColor: '#FF0000',
                    fillOpacity: 0.35,
                    clickable: false,
                    draggable: false,
                    editable: false,
                    visible: true,
                }}
            />
        </>
    );
}

export { SolarPanelF }