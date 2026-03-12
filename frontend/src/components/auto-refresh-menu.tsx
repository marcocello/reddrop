import { ChevronDown, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'

type AutoRefreshMenuProps = {
  valueSeconds: number
  onValueChange: (value: number) => void
}

const AUTO_REFRESH_OPTIONS = [
  { value: 0, label: 'off', short: 'off' },
  { value: 5, label: 'every 5 seconds', short: '5s' },
  { value: 30, label: 'every 30 seconds', short: '30s' },
  { value: 60, label: 'every 1 minute', short: '1m' },
]

function shortLabel(valueSeconds: number): string {
  const match = AUTO_REFRESH_OPTIONS.find((item) => item.value === valueSeconds)
  return match?.short ?? 'off'
}

export function AutoRefreshMenu({ valueSeconds, onValueChange }: AutoRefreshMenuProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant='outline' size='sm' className='h-8 gap-1 px-2 text-xs'>
          <RefreshCw className={cn('size-4', valueSeconds > 0 && 'animate-spin [animation-duration:2s]')} />
          <ChevronDown className='size-4' />
          {shortLabel(valueSeconds)}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align='end' className='min-w-[12rem]'>
        <DropdownMenuRadioGroup
          value={String(valueSeconds)}
          onValueChange={(next) => onValueChange(Number(next))}
        >
          {AUTO_REFRESH_OPTIONS.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={String(option.value)}>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
