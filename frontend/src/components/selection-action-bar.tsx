import type { ReactNode } from 'react'
import { Trash2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'

type SelectionActionBarProps = {
  selectedCount: number
  entityLabel: string
  onClear: () => void
  onDelete: () => void
  deleteLabel?: string
  deleteDisabled?: boolean
  children?: ReactNode
  className?: string
}

export function SelectionActionBar({
  selectedCount,
  entityLabel,
  onClear,
  onDelete,
  deleteLabel = 'Delete selected',
  deleteDisabled = false,
  children,
  className,
}: SelectionActionBarProps): ReactNode | null {
  if (selectedCount <= 0) return null

  const normalizedEntity = entityLabel.trim() || 'item'
  const entityText = selectedCount === 1 ? normalizedEntity : `${normalizedEntity}s`

  return (
    <div
      className={cn(
        'fixed bottom-6 left-1/2 z-50 w-fit max-w-[calc(100vw-1.5rem)] -translate-x-1/2',
        className
      )}
    >
      <div className='flex items-center gap-2 rounded-2xl border bg-background/95 px-3 py-2 shadow-xl backdrop-blur supports-[backdrop-filter]:bg-background/70'>
        <Button
          variant='ghost'
          size='icon'
          className='size-8 rounded-full'
          onClick={onClear}
          aria-label={`Clear selected ${entityText}`}
        >
          <X className='size-4' />
        </Button>
        <Separator orientation='vertical' className='h-5' />
        <span className='rounded-full bg-slate-900 px-3 py-1 text-xs font-semibold text-white'>
          {selectedCount}
        </span>
        <span className='text-sm font-medium whitespace-nowrap'>
          {entityText} selected
        </span>
        <Separator orientation='vertical' className='h-5' />
        {children}
        <Button
          size='sm'
          className='bg-red-600 text-white hover:bg-red-700'
          onClick={onDelete}
          disabled={deleteDisabled}
        >
          <Trash2 className='size-4' />
          {deleteLabel}
        </Button>
      </div>
    </div>
  )
}
