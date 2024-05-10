import { SearchIcon } from "lucide-react"
import { Input } from "@/components/ui/input"

export interface SearchbarProps {
  className?: string
  onClick?: () => void
}

export function Searchbar({className, onClick} : SearchbarProps) {
  return (
    <div className={className}>
      <SearchIcon className="text-gray-500 translate-y-8 ml-1" />
      <Input
        className="w-full pl-8 rounded bg-w text-gray-800 focus-visible:ring-0 focus-visible:outline-none"
        placeholder="Search location..."
        type="search"
      />
      
      <ul className="text-sm text-gray-700 bg-white rounded mt-1">
        {[1, 2, 3].map((option, index) => (
          <li key={index}>
          <a onClick={onClick} className="block rounded px-4 py-2 hover:bg-gray-200 cursor-pointer">
            Recent Search {option}
          </a>
        </li>
        ))}
      </ul>
      
    </div>
  )
}
