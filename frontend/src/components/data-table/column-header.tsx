import {
  ArrowDownIcon,
  ArrowUpIcon,
  CaretSortIcon,
} from '@radix-ui/react-icons'
import { type Column } from '@tanstack/react-table'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

type DataTableColumnHeaderProps<TData, TValue> =
  React.HTMLAttributes<HTMLDivElement> & {
    column: Column<TData, TValue>
    title: string
  }

export function DataTableColumnHeader<TData, TValue>({
  column,
  title,
  className,
}: DataTableColumnHeaderProps<TData, TValue>) {
  if (!column.getCanSort()) {
    return <div className={cn(className)}>{title}</div>
  }

  const sorted = column.getIsSorted()

  return (
    <div className={cn('flex items-center space-x-2', className)}>
      <Button
        type='button'
        variant='ghost'
        size='sm'
        className='h-8'
        onClick={() => column.toggleSorting(sorted === 'asc')}
      >
        <span>{title}</span>
        {sorted === 'desc' ? (
          <ArrowDownIcon className='ms-2 h-4 w-4' />
        ) : sorted === 'asc' ? (
          <ArrowUpIcon className='ms-2 h-4 w-4' />
        ) : (
          <CaretSortIcon className='ms-2 h-4 w-4' />
        )}
      </Button>
    </div>
  )
}
