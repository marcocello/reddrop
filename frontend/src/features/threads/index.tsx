import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  type ColumnDef,
  type ColumnOrderState,
  type ColumnFiltersState,
  flexRender,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  type FilterFn,
  type SortingState,
  useReactTable,
  type VisibilityState,
} from '@tanstack/react-table'
import { MoreHorizontal, RefreshCw } from 'lucide-react'
import { DataTableColumnHeader, DataTablePagination, DataTableToolbar } from '@/components/data-table'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Label } from '@/components/ui/label'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import { reddropApi, type Job, type ThreadItem } from '@/lib/reddrop-api'

const ACTIONS_COLUMN_ID = 'actions'
const JOB_COLUMN_ID = 'job'
const AUTO_REFRESH_INTERVAL_MS = 2000

const arrayFilterFn: FilterFn<ThreadItem> = (row, columnId, value) => {
  const selected = Array.isArray(value) ? (value as string[]) : []
  if (selected.length === 0) {
    return true
  }
  return selected.includes(String(row.getValue(columnId)))
}

function formatWhen(value?: string | number | null): string {
  if (!value) return '-'
  let normalizedValue: string | number = value
  if (typeof value === 'number') {
    normalizedValue = value < 1_000_000_000_000 ? value * 1000 : value
  } else {
    const numeric = Number(value)
    if (!Number.isNaN(numeric) && Number.isFinite(numeric)) {
      normalizedValue = numeric < 1_000_000_000_000 ? numeric * 1000 : numeric
    }
  }
  const parsed = new Date(normalizedValue)
  if (Number.isNaN(parsed.getTime())) return String(value)
  return parsed.toLocaleString()
}

function formatThreadStatus(thread: ThreadItem): string {
  if (thread.user_has_commented) return 'Sent'
  if (thread.has_reply) return 'Draft'
  return 'No reply'
}

function shortId(value: string): string {
  const normalized = value.trim()
  if (!normalized) return '-'
  return normalized.slice(0, 8)
}

function formatJobCell(row: ThreadItem): string {
  return `${shortId(row.job_id)} (${row.job_name})`
}

function threadReplyKey(thread: ThreadItem): string {
  return `${thread.job_id}:${thread.subreddit}:${thread.thread_id}:reply`
}

function threadSendKey(thread: ThreadItem): string {
  return `${thread.job_id}:${thread.subreddit}:${thread.thread_id}:send`
}

export function ThreadsPage() {
  const [threads, setThreads] = useState<ThreadItem[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const [selectedThread, setSelectedThread] = useState<ThreadItem | null>(null)
  const [replyInput, setReplyInput] = useState('')
  const [sheetOpen, setSheetOpen] = useState(false)
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'semantic_similarity', desc: true },
  ])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [globalFilter, setGlobalFilter] = useState('')
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>([])
  const [draggingColumnId, setDraggingColumnId] = useState<string | null>(null)

  const loadThreads = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const payload = await reddropApi.listThreads({
        only_open: false,
        limit: 500,
        offset: 0,
      })
      setThreads(payload.threads)
      setSelectedThread((prev) => {
        if (!prev) return null
        return payload.threads.find(
          (item) =>
            item.job_id === prev.job_id &&
            item.subreddit === prev.subreddit &&
            item.thread_id === prev.thread_id
        ) ?? null
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load threads'
      toast.error(message)
    } finally {
      setLoading(false)
    }
  }, [])

  const loadJobs = useCallback(async (): Promise<void> => {
    try {
      const payload = await reddropApi.listJobs()
      setJobs(payload.jobs)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load jobs'
      toast.error(message)
    }
  }, [])

  useEffect(() => {
    void loadThreads()
    void loadJobs()
  }, [loadJobs, loadThreads])

  useEffect(() => {
    const timer = window.setInterval(() => {
      void loadThreads()
    }, AUTO_REFRESH_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [loadThreads])

  useEffect(() => {
    setReplyInput(selectedThread?.reply ?? '')
  }, [selectedThread])

  const resolvePersonaForThread = useCallback((thread: ThreadItem): string | null => {
    const job = jobs.find((item) => item.id === thread.job_id)
    const persona = job?.personas?.find((item) => item.trim().length > 0)?.trim() ?? null
    if (persona) {
      return persona
    }
    const jobRef = job?.name ?? thread.job_name
    toast.error(`No persona configured for job ${jobRef}. Edit the job and assign at least one persona.`)
    return null
  }, [jobs])

  const performReply = useCallback(async (thread: ThreadItem): Promise<void> => {
    const persona = resolvePersonaForThread(thread)
    if (!persona) {
      return
    }
    const replyKey = threadReplyKey(thread)
    setBusyKey(replyKey)
    try {
      const replyDraft = replyInput.trim()
      const updated = await reddropApi.createThreadReply(
        thread.job_id,
        thread.subreddit,
        thread.thread_id,
        persona,
        replyDraft.length > 0 ? replyDraft : undefined
      )
      toast.success(`Reply generated for ${thread.thread_id}`)
      setSelectedThread(updated)
      setReplyInput(updated.reply ?? '')
      await loadThreads()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to generate reply'
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [loadThreads, replyInput, resolvePersonaForThread])

  const performSend = useCallback(async (thread: ThreadItem): Promise<void> => {
    const sendKey = threadSendKey(thread)
    setBusyKey(sendKey)
    try {
      const updated = await reddropApi.sendThreadReply(
        thread.job_id,
        thread.subreddit,
        thread.thread_id
      )
      toast.success(`Reply sent for ${thread.thread_id}`)
      setSelectedThread(updated)
      await loadThreads()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to send reply'
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [loadThreads])

  const jobFilterOptions = useMemo(() => {
    const values = new Set<string>()
    threads.forEach((item) => values.add(formatJobCell(item)))
    return Array.from(values)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ label: value, value }))
  }, [threads])

  const subredditFilterOptions = useMemo(() => {
    const values = new Set<string>()
    threads.forEach((item) => values.add(item.subreddit))
    return Array.from(values)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ label: `r/${value}`, value }))
  }, [threads])

  const columns = useMemo<ColumnDef<ThreadItem>[]>(
    () => [
      {
        id: JOB_COLUMN_ID,
        accessorFn: (row) => formatJobCell(row),
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Job' />,
        cell: ({ row }) => (
          <div className='max-w-[11rem]'>
            <span className='block font-mono text-xs font-medium leading-tight'>
              {shortId(row.original.job_id)}
            </span>
            <span className='block truncate text-xs text-muted-foreground leading-tight' title={row.original.job_name}>
              ({row.original.job_name})
            </span>
          </div>
        ),
      },
      {
        accessorKey: 'title',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Thread' />,
        enableHiding: false,
        cell: ({ row }) => (
          <div className='max-w-[28rem]'>
            <span className='block max-w-[28rem] line-clamp-2 font-medium leading-snug' title={row.original.title}>
              {row.original.title}
            </span>
          </div>
        ),
      },
      {
        accessorKey: 'subreddit',
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Subreddit' />,
        cell: ({ row }) => `r/${row.original.subreddit}`,
      },
      {
        id: 'status',
        accessorFn: (row) => (row.user_has_commented ? 'sent' : row.has_reply ? 'draft' : 'none'),
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Status' />,
        cell: ({ row }) => {
          if (row.original.user_has_commented) {
            return <Badge className='bg-green-600 text-white'>Sent</Badge>
          }
          if (row.original.has_reply) {
            return <Badge className='bg-blue-600 text-white'>Draft</Badge>
          }
          return <Badge variant='secondary'>No reply</Badge>
        },
      },
      {
        accessorKey: 'semantic_similarity',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Similarity' />,
        cell: ({ row }) => row.original.semantic_similarity.toFixed(3),
      },
      {
        accessorKey: 'score',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Score' />,
      },
      {
        accessorKey: 'num_comments',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Comments' />,
      },
      {
        accessorKey: 'created_utc',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Created' />,
        cell: ({ row }) => formatWhen(row.original.created_utc),
      },
      {
        accessorKey: 'reply',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Reply' />,
        cell: ({ row }) => (
          <span className='block max-w-[22rem] truncate text-sm text-muted-foreground' title={row.original.reply}>
            {row.original.reply || '-'}
          </span>
        ),
      },
      {
        id: ACTIONS_COLUMN_ID,
        enableSorting: false,
        enableHiding: false,
        cell: ({ row }) => {
          const thread = row.original
          return (
            <div className='flex justify-end'>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant='ghost'
                    size='icon'
                    className='size-8'
                    onClick={(event) => event.stopPropagation()}
                  >
                    <MoreHorizontal className='size-4' />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align='end' onClick={(event) => event.stopPropagation()}>
                  <DropdownMenuItem onSelect={() => void performReply(thread)}>Reply</DropdownMenuItem>
                  <DropdownMenuItem
                    disabled={!thread.has_reply || thread.user_has_commented}
                    onSelect={() => void performSend(thread)}
                  >
                    Send
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )
        },
      },
    ],
    [performReply, performSend]
  )

  const orderedColumnIds = useMemo(
    () =>
      columns
        .map((column) => {
          if (typeof column.id === 'string' && column.id.length > 0) return column.id
          if ('accessorKey' in column && typeof column.accessorKey === 'string' && column.accessorKey.length > 0) {
            return column.accessorKey
          }
          return ''
        })
        .filter((value): value is string => value.length > 0),
    [columns]
  )

  useEffect(() => {
    setColumnOrder((previous) => {
      if (previous.length === 0) return orderedColumnIds
      const previousSet = new Set(previous)
      let next = previous.filter((id) => orderedColumnIds.includes(id))
      orderedColumnIds.forEach((id) => {
        if (!previousSet.has(id)) next.push(id)
      })
      next = next.filter((id) => id !== JOB_COLUMN_ID)
      next.unshift(JOB_COLUMN_ID)
      return next
    })
  }, [orderedColumnIds])

  const moveColumn = useCallback((fromId: string, toId: string): void => {
    if (
      fromId === toId ||
      fromId === ACTIONS_COLUMN_ID ||
      toId === ACTIONS_COLUMN_ID ||
      fromId === JOB_COLUMN_ID ||
      toId === JOB_COLUMN_ID
    ) {
      return
    }
    setColumnOrder((previous) => {
      const next = previous.length > 0 ? [...previous] : [...orderedColumnIds]
      const fromIndex = next.indexOf(fromId)
      const toIndex = next.indexOf(toId)
      if (fromIndex < 0 || toIndex < 0) return next
      const [moved] = next.splice(fromIndex, 1)
      next.splice(toIndex, 0, moved)
      const jobIndex = next.indexOf(JOB_COLUMN_ID)
      if (jobIndex > 0) {
        next.unshift(next.splice(jobIndex, 1)[0])
      }
      return next
    })
  }, [orderedColumnIds])

  const table = useReactTable({
    data: threads,
    columns,
    state: {
      sorting,
      columnFilters,
      globalFilter,
      columnVisibility,
      columnOrder,
    },
    enableGlobalFilter: true,
    globalFilterFn: (row, _columnId, filterValue) => {
      const query = String(filterValue ?? '').trim().toLowerCase()
      if (!query) return true
      const item = row.original
      return (
        item.title.toLowerCase().includes(query) ||
        item.job_id.toLowerCase().includes(query) ||
        item.job_name.toLowerCase().includes(query) ||
        item.subreddit.toLowerCase().includes(query) ||
        item.url.toLowerCase().includes(query)
      )
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    onColumnVisibilityChange: setColumnVisibility,
    onColumnOrderChange: setColumnOrder,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
    initialState: {
      pagination: { pageSize: 10 },
    },
  })

  const selectedReplyBusy = selectedThread ? busyKey === threadReplyKey(selectedThread) : false
  const selectedSendBusy = selectedThread ? busyKey === threadSendKey(selectedThread) : false

  return (
    <>
      <Header fixed>
        <div className='ms-auto flex items-center gap-2'>
          <Button
            variant='outline'
            size='sm'
            onClick={() => void loadThreads()}
            disabled={loading}
            aria-label='Refresh threads'
            title='Refresh threads'
          >
            <RefreshCw className='size-4 animate-spin [animation-duration:2s]' />
            Refresh
          </Button>
        </div>
      </Header>

      <Main className='flex flex-1 flex-col gap-4 sm:gap-6'>
        <div>
          <h2 className='text-2xl font-bold tracking-tight'>Threads</h2>
        </div>

        <DataTableToolbar
          table={table}
          searchPlaceholder='Search by thread, job id, subreddit or URL...'
          filters={[
            {
              columnId: 'status',
              title: 'Status',
              options: [
                { label: 'No reply', value: 'none' },
                { label: 'Draft', value: 'draft' },
                { label: 'Sent', value: 'sent' },
              ],
            },
            ...(jobFilterOptions.length > 0
              ? [{ columnId: 'job', title: 'Job', options: jobFilterOptions }]
              : []),
            ...(subredditFilterOptions.length > 0
              ? [{ columnId: 'subreddit', title: 'Subreddit', options: subredditFilterOptions }]
              : []),
          ]}
        />

        <div className='space-y-4'>
          <div className='rounded-md border'>
            <Table>
              <TableHeader>
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => (
                      <TableHead
                        key={header.id}
                        className={cn(
                          header.column.id === draggingColumnId && 'opacity-50',
                          header.column.id !== ACTIONS_COLUMN_ID && 'cursor-move',
                          header.column.id === ACTIONS_COLUMN_ID &&
                            'sticky right-0 z-20 bg-background text-right'
                        )}
                        draggable={header.column.id !== ACTIONS_COLUMN_ID}
                        onDragStart={(event) => {
                          if (header.column.id === ACTIONS_COLUMN_ID) return
                          setDraggingColumnId(header.column.id)
                          event.dataTransfer.effectAllowed = 'move'
                          event.dataTransfer.setData('text/plain', header.column.id)
                        }}
                        onDragOver={(event) => {
                          if (!draggingColumnId || header.column.id === ACTIONS_COLUMN_ID) return
                          if (draggingColumnId === header.column.id) return
                          event.preventDefault()
                          event.dataTransfer.dropEffect = 'move'
                        }}
                        onDrop={(event) => {
                          if (header.column.id === ACTIONS_COLUMN_ID) return
                          event.preventDefault()
                          const source = draggingColumnId ?? event.dataTransfer.getData('text/plain')
                          if (!source) return
                          moveColumn(source, header.column.id)
                          setDraggingColumnId(null)
                        }}
                        onDragEnd={() => setDraggingColumnId(null)}
                      >
                        {header.isPlaceholder
                          ? null
                          : flexRender(header.column.columnDef.header, header.getContext())}
                      </TableHead>
                    ))}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody>
                {table.getRowModel().rows.length ? (
                  table.getRowModel().rows.map((row) => (
                    <TableRow
                      key={row.id}
                      className='cursor-pointer'
                      onClick={() => {
                        setSelectedThread(row.original)
                        setSheetOpen(true)
                      }}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <TableCell
                          key={cell.id}
                          className={cn(
                            cell.column.id === ACTIONS_COLUMN_ID &&
                              'sticky right-0 z-10 bg-background'
                          )}
                        >
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={table.getVisibleLeafColumns().length} className='h-24 text-center text-muted-foreground'>
                      No threads found.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>

          <DataTablePagination table={table} />
        </div>
      </Main>

      <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
        <SheetContent side='right' className='sm:max-w-lg p-0'>
          <div className='flex h-full flex-col'>
            <SheetHeader className='px-4 pt-4'>
              <SheetTitle>{selectedThread?.title ?? 'Thread details'}</SheetTitle>
              <SheetDescription>
                {selectedThread ? `${selectedThread.job_name} · r/${selectedThread.subreddit}` : ''}
              </SheetDescription>
            </SheetHeader>

            {selectedThread && (
              <div className='flex-1 space-y-3 overflow-y-auto px-4 py-4 text-sm'>
                <div className='space-y-2'>
                  <Badge variant={selectedThread.user_has_commented ? 'default' : 'secondary'}>
                    {formatThreadStatus(selectedThread)}
                  </Badge>
                  <p className='text-xs text-muted-foreground'>
                    {`${selectedThread.semantic_similarity.toFixed(3)} similarity · ${selectedThread.score} score · ${selectedThread.num_comments} comments · ${formatWhen(selectedThread.created_utc)}`}
                  </p>
                </div>

                <p><span className='font-medium'>Thread ID:</span> {selectedThread.thread_id}</p>
                <p>
                  <span className='font-medium'>URL:</span>{' '}
                  <a
                    href={selectedThread.url}
                    target='_blank'
                    rel='noreferrer'
                    className='block max-w-full break-all underline'
                    style={{
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                    }}
                  >
                    {selectedThread.url}
                  </a>
                </p>
                <div className='space-y-2'>
                  <Label htmlFor='selected-thread-body'>Thread Body</Label>
                  <Textarea
                    id='selected-thread-body'
                    value={selectedThread.selftext || ''}
                    readOnly
                    className='min-h-[10rem] max-h-[10rem] resize-none'
                  />
                </div>
                <div className='space-y-2'>
                  <Label htmlFor='selected-thread-reply'>Reply Input</Label>
                  <Textarea
                    id='selected-thread-reply'
                    value={replyInput}
                    onChange={(event) => setReplyInput(event.target.value)}
                    className='min-h-[8rem] resize-y'
                    placeholder='Write or adjust the reply draft...'
                  />
                </div>
                <div className='flex items-center gap-2'>
                  <Button
                    size='sm'
                    variant='outline'
                    disabled={selectedReplyBusy}
                    onClick={() => void performReply(selectedThread)}
                  >
                    Reply now
                  </Button>
                  <Button
                    size='sm'
                    disabled={
                      !selectedThread.has_reply ||
                      selectedThread.user_has_commented ||
                      selectedSendBusy
                    }
                    onClick={() => void performSend(selectedThread)}
                  >
                    Send now
                  </Button>
                </div>
              </div>
            )}
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
