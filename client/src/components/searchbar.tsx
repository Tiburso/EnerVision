import { SearchIcon } from "lucide-react"
import { Input } from "@/components/ui/input"

import usePlacesAutocomplete, {
  LatLng,
  getGeocode,
  getLatLng,
} from 'use-places-autocomplete';

export interface SearchbarProps {
  className?: string
  onClick?: (val: LatLng) => void
}

/**
 * The Searchbar component is a functional component that renders a search bar for possible 
 * google maps locations.
 * 
 * @param className - The class name of the search bar.
 * @param onClick - The function to call when a location is clicked. 
 */
export function Searchbar({className, onClick} : SearchbarProps) {
  const {
    ready,
    value,
    suggestions: { status, data },
    setValue,
    clearSuggestions,
  } = usePlacesAutocomplete({
    requestOptions: { componentRestrictions: { country: 'nl' } },
    debounce: 300,
    cache: 86400,
  });

  return (
    <div className={className}>
      <SearchIcon className="text-gray-500 translate-y-8 ml-1" />
      <Input
        className="w-full pl-8 rounded bg-w text-gray-800 focus-visible:ring-0 focus-visible:outline-none"
        placeholder="Search location..."
        type="search"
        disabled={!ready}
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      
      {status === 'OK' && (
        <ul className="absolute w-full z-50 text-sm text-gray-700 bg-white rounded mt-1">
        {data.map((suggestion) => {
          const {
            place_id,
            structured_formatting: { main_text, secondary_text },
            description,
          } = suggestion;

          return (
            <li key={place_id}>
            <a onClick={async () => {
              const geocode = await getGeocode( { address: description } );
              const latLng = getLatLng(geocode[0]);

              setValue(description, false);
              clearSuggestions();
              onClick && onClick(latLng);
            }} className="block rounded px-4 py-2 hover:bg-gray-200 cursor-pointer">
              <strong>{main_text}</strong> <small>{secondary_text}</small>
            </a>
          </li>
         )}
        )}
      </ul>
      )}
    </div>
  )
}