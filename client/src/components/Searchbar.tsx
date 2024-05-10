import {
    Calculator,
    Calendar,
    CreditCard,
    Settings,
    Smile,
    User,
  } from "lucide-react"
   
  import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
    CommandSeparator,
    CommandShortcut,
  } from "@/components/ui/command"

import React from "react";

export interface SearchbarProps {
    className?: string;
}

const Searchbar = ({className} : SearchbarProps) => {
    return (
    <Command className="rounded-lg border shadow-md">
    <CommandInput placeholder="Type a command or search..." />
    <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        <CommandGroup heading="Suggestions">
        <CommandItem>
            <Calendar className="mr-2 h-4 w-4" />
            <span>Calendar</span>
        </CommandItem>
        <CommandItem>
            <Smile className="mr-2 h-4 w-4" />
            <span>Search Emoji</span>
        </CommandItem>
        <CommandItem>
            <Calculator className="mr-2 h-4 w-4" />
            <span>Calculator</span>
        </CommandItem>
        </CommandGroup>
    </CommandList>
    </Command>
    )
}

// const PlacesAutocomplete = ({
//     onAddressSelect,
//   }: {
//     onAddressSelect?: (address: string) => void;
//   }) => {
//     const {
//       ready,
//       value,
//       suggestions: { status, data },
//       setValue,
//       clearSuggestions,
//     } = usePlacesAutocomplete({
//       requestOptions: { componentRestrictions: { country: 'us' } },
//       debounce: 300,
//       cache: 86400,
//     });
  
//     const renderSuggestions = () => {
//       return data.map((suggestion) => {
//         const {
//           place_id,
//           structured_formatting: { main_text, secondary_text },
//           description,
//         } = suggestion;
  
//         return (
//           <li
//             key={place_id}
//             onClick={() => {
//               setValue(description, false);
//               clearSuggestions();
//               onAddressSelect && onAddressSelect(description);
//             }}
//           >
//             <strong>{main_text}</strong> <small>{secondary_text}</small>
//           </li>
//         );
//       });
//     };
  
//     return (
//       <div className={styles.autocompleteWrapper}>
//         <input
//           value={value}
//           className={styles.autocompleteInput}
//           disabled={!ready}
//           onChange={(e) => setValue(e.target.value)}
//           placeholder="123 Stariway To Heaven"
//         />
  
//         {status === 'OK' && (
//           <ul className={styles.suggestionWrapper}>{renderSuggestions()}</ul>
//         )}
//       </div>
//     );
//   };

export { Searchbar };