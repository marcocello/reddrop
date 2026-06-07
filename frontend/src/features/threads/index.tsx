import { useCallback, useEffect, useMemo, useState } from 'react'
import { faLinkedinIn, faRedditAlien } from '@fortawesome/free-brands-svg-icons'
import type { IconDefinition } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
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
import { ChevronDown, Loader2, MoreHorizontal, RefreshCw } from 'lucide-react'
import { DataTableColumnHeader, DataTablePagination, DataTableToolbar } from '@/components/data-table'
import { ConfirmDialog } from '@/components/confirm-dialog'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { SelectionActionBar } from '@/components/selection-action-bar'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import { reddropApi, type Job, type ThreadItem } from '@/lib/reddrop-api'

const ACTIONS_COLUMN_ID = 'actions'
const SELECT_COLUMN_ID = 'select'
const ID_COLUMN_ID = 'id'
const JOB_COLUMN_ID = 'job'
const AUTO_REFRESH_OPTIONS = [
  { value: 0, label: 'Off', menuLabel: 'Auto-refresh off' },
  { value: 30, label: 'Every 30s', menuLabel: 'Every 30s' },
  { value: 60, label: 'Every 1m', menuLabel: 'Every 1m' },
  { value: 300, label: 'Every 5m', menuLabel: 'Every 5m' },
  { value: 900, label: 'Every 15m', menuLabel: 'Every 15m' },
] as const

const arrayFilterFn: FilterFn<ThreadItem> = (row, columnId, value) => {
  const selected = Array.isArray(value) ? (value as string[]) : []
  if (selected.length === 0) {
    return true
  }
  return selected.includes(String(row.getValue(columnId)))
}

function formatWhen(value?: string | number | null): string {
  if (!value) return 'ND'
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) return trimmed
  }
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
  if (Number.isNaN(parsed.getTime())) return 'ND'
  return parsed.toISOString().replace('T', ' ').slice(0, 16)
}

function formatThreadCreatedWhen(thread: ThreadItem): string {
  return formatWhen(thread.published_at ?? thread.created_utc)
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

function formatSocial(value?: ThreadItem['social']): string {
  if (value === 'linkedin') return 'LinkedIn'
  if (value === 'reddit') return 'Reddit'
  return '-'
}

function socialIcon(platform: ThreadItem['social']): IconDefinition {
  return platform === 'linkedin' ? faLinkedinIn : faRedditAlien
}

function threadReplyKey(thread: ThreadItem): string {
  return `${thread.job_id}:${thread.subreddit}:${thread.thread_id}:reply`
}

function threadSendKey(thread: ThreadItem): string {
  return `${thread.job_id}:${thread.subreddit}:${thread.thread_id}:send`
}

function threadDeleteKey(thread: ThreadItem): string {
  return `${thread.job_id}:${thread.subreddit}:${thread.thread_id}:delete`
}

export function ThreadsPage() {
  const [threads, setThreads] = useState<ThreadItem[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [autoRefreshSeconds, setAutoRefreshSeconds] = useState<number>(30)
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const [selectedConversationIds, setSelectedConversationIds] = useState<Set<string>>(new Set())
  const [selectedThread, setSelectedThread] = useState<ThreadItem | null>(null)
  const [pendingDeleteThreads, setPendingDeleteThreads] = useState<ThreadItem[]>([])
  const [replyInput, setReplyInput] = useState('')
  const [replyAutosaveState, setReplyAutosaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [sheetOpen, setSheetOpen] = useState(false)
  const [sheetTab, setSheetTab] = useState<'overview' | 'compose' | 'activity'>('overview')
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
      setSelectedConversationIds((previous) => {
        const available = new Set(payload.threads.map((item) => item.conversation_id))
        const next = new Set<string>()
        previous.forEach((id) => {
          if (available.has(id)) next.add(id)
        })
        return next
      })
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
    if (autoRefreshSeconds <= 0) {
      return
    }
    const timer = window.setInterval(() => {
      void loadThreads()
    }, autoRefreshSeconds * 1000)
    return () => window.clearInterval(timer)
  }, [autoRefreshSeconds, loadThreads])

  const autoRefreshLabel = useMemo(() => {
    const selected = AUTO_REFRESH_OPTIONS.find((option) => option.value === autoRefreshSeconds)
    return selected?.label ?? 'Off'
  }, [autoRefreshSeconds])

  useEffect(() => {
    setReplyInput(selectedThread?.reply ?? '')
    setReplyAutosaveState('idle')
  }, [selectedThread?.conversation_id])

  const openThreadSheet = useCallback((thread: ThreadItem): void => {
    setSelectedThread(thread)
    setSheetTab('overview')
    setSheetOpen(true)
  }, [])

  const resolvePersonaForThread = useCallback((thread: ThreadItem, options?: { silent?: boolean }): string | null => {
    const silent = options?.silent === true
    const linkedPersona = thread.reply_personas?.find((item) => item.trim().length > 0)?.trim() ?? null
    if (linkedPersona) {
      return linkedPersona
    }

    const job = jobs.find((item) => item.id === thread.job_id)
    const persona = job?.personas?.find((item) => item.trim().length > 0)?.trim() ?? null
    if (persona) {
      return persona
    }
    const linkedReplyName = thread.reply_job_name?.trim()
    if (linkedReplyName) {
      if (!silent) {
        toast.error(`No persona configured for reply job ${linkedReplyName}. Edit that Reply job and assign at least one persona.`)
      }
      return null
    }
    const jobRef = job?.name ?? thread.job_name
    if (!silent) {
      toast.error(`No persona configured for job ${jobRef}. Edit the job and assign at least one persona.`)
    }
    return null
  }, [jobs])

  const persistReplyDraft = useCallback(async (
    thread: ThreadItem,
    draftText: string,
    options?: { notifyOnError?: boolean }
  ): Promise<ThreadItem | null> => {
    const persona = resolvePersonaForThread(thread, { silent: true })
    if (!persona) return null
    const normalizedReply = draftText.trim()
    if (!normalizedReply) return null

    setReplyAutosaveState('saving')
    try {
      const updated = await reddropApi.createThreadReply(
        thread.reply_job_id ?? thread.job_id,
        thread.subreddit,
        thread.thread_id,
        persona,
        normalizedReply
      )
      setSelectedThread((previous) => {
        if (!previous) return previous
        return previous.conversation_id === updated.conversation_id ? updated : previous
      })
      setThreads((previous) => previous.map((item) => (
        item.conversation_id === updated.conversation_id ? updated : item
      )))
      setReplyAutosaveState('saved')
      return updated
    } catch (error) {
      setReplyAutosaveState('error')
      if (options?.notifyOnError) {
        const message = error instanceof Error ? error.message : 'Failed to save reply draft'
        toast.error(message)
      }
      return null
    }
  }, [resolvePersonaForThread])

  const performReply = useCallback(async (thread: ThreadItem): Promise<void> => {
    const persona = resolvePersonaForThread(thread)
    if (!persona) {
      return
    }
    const replyKey = threadReplyKey(thread)
    setBusyKey(replyKey)
    try {
      const updated = await reddropApi.createThreadReply(
        thread.reply_job_id ?? thread.job_id,
        thread.subreddit,
        thread.thread_id,
        persona
      )
      toast.success(`Reply generated for ${thread.thread_id}`)
      setSelectedThread(updated)
      setReplyInput(updated.reply ?? '')
      setReplyAutosaveState('idle')
      await loadThreads()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to generate reply'
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [loadThreads, resolvePersonaForThread])

  const performSend = useCallback(async (thread: ThreadItem): Promise<void> => {
    const sendKey = threadSendKey(thread)
    setBusyKey(sendKey)
    try {
      let currentThread = thread
      const draftReply = replyInput.trim()
      const persistedReply = (currentThread.reply ?? '').trim()
      if (draftReply.length > 0 && draftReply !== persistedReply) {
        const saved = await persistReplyDraft(currentThread, draftReply, { notifyOnError: true })
        if (!saved) return
        currentThread = saved
      }
      const updated = await reddropApi.sendThreadReply(
        currentThread.reply_job_id ?? currentThread.job_id,
        currentThread.subreddit,
        currentThread.thread_id
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
  }, [loadThreads, persistReplyDraft, replyInput])

  const deleteThreads = useCallback(async (targets: ThreadItem[]): Promise<void> => {
    if (targets.length === 0) return
    const bulkKey = 'threads:bulk-delete'
    const isBulk = targets.length > 1
    setBusyKey(isBulk ? bulkKey : threadDeleteKey(targets[0]))
    try {
      const settled = await Promise.allSettled(
        targets.map((thread) =>
          reddropApi.deleteThread(
            thread.reply_job_id ?? thread.job_id,
            thread.subreddit,
            thread.thread_id
          )
        )
      )

      const succeededIds = new Set<string>()
      let failedCount = 0
      settled.forEach((item, index) => {
        if (item.status === 'fulfilled') {
          succeededIds.add(targets[index].conversation_id)
        } else {
          failedCount += 1
        }
      })

      if (succeededIds.size > 0) {
        if (selectedThread && succeededIds.has(selectedThread.conversation_id)) {
          setSelectedThread(null)
          setSheetOpen(false)
        }
        setSelectedConversationIds((previous) => {
          const next = new Set(previous)
          succeededIds.forEach((id) => next.delete(id))
          return next
        })
        await loadThreads()
        toast.success(
          succeededIds.size === 1
            ? `Deleted conversation ${targets.find((thread) => succeededIds.has(thread.conversation_id))?.thread_id ?? ''}`
            : `Deleted ${succeededIds.size} conversations`
        )
      }

      if (failedCount > 0) {
        toast.error(
          failedCount === 1
            ? 'Failed to delete 1 conversation'
            : `Failed to delete ${failedCount} conversations`
        )
      }
    } finally {
      setBusyKey(null)
    }
  }, [loadThreads, selectedThread])

  const requestDeleteThread = useCallback((thread: ThreadItem): void => {
    setPendingDeleteThreads([thread])
  }, [])

  const requestDeleteSelectedThreads = useCallback((): void => {
    const selectedThreads = threads.filter((item) => selectedConversationIds.has(item.conversation_id))
    if (selectedThreads.length === 0) return
    setPendingDeleteThreads(selectedThreads)
  }, [selectedConversationIds, threads])

  const confirmDeleteThread = useCallback(async (): Promise<void> => {
    if (pendingDeleteThreads.length === 0) return
    await deleteThreads(pendingDeleteThreads)
    setPendingDeleteThreads([])
  }, [deleteThreads, pendingDeleteThreads])

  const jobFilterOptions = useMemo(() => {
    const values = new Set<string>()
    threads.forEach((item) => values.add(item.job_name))
    return Array.from(values)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ label: value, value }))
  }, [threads])

  const socialFilterOptions = useMemo(() => {
    const values = new Set<string>()
    threads.forEach((item) => values.add(formatSocial(item.social)))
    return Array.from(values)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ label: value, value }))
  }, [threads])

  const selectedThreads = useMemo(
    () => threads.filter((item) => selectedConversationIds.has(item.conversation_id)),
    [selectedConversationIds, threads]
  )

  const columns = useMemo<ColumnDef<ThreadItem>[]>(
    () => [
      {
        id: SELECT_COLUMN_ID,
        enableSorting: false,
        enableHiding: false,
        header: ({ table }) => {
          const visibleRows = table.getRowModel().rows
          const visibleIds = visibleRows.map((row) => row.original.conversation_id)
          const selectedVisible = visibleIds.filter((id) => selectedConversationIds.has(id)).length
          const allVisibleSelected = visibleIds.length > 0 && selectedVisible === visibleIds.length
          const partiallySelected = selectedVisible > 0 && !allVisibleSelected

          return (
            <div className='flex items-center justify-center' onClick={(event) => event.stopPropagation()}>
              <Checkbox
                aria-label='Select all visible conversations'
                checked={allVisibleSelected ? true : partiallySelected ? 'indeterminate' : false}
                onCheckedChange={(checked) => {
                  const shouldSelect = checked === true
                  setSelectedConversationIds((previous) => {
                    const next = new Set(previous)
                    visibleIds.forEach((id) => {
                      if (shouldSelect) next.add(id)
                      else next.delete(id)
                    })
                    return next
                  })
                }}
              />
            </div>
          )
        },
        cell: ({ row }) => {
          const isSelected = selectedConversationIds.has(row.original.conversation_id)
          return (
            <div className='flex items-center justify-center' onClick={(event) => event.stopPropagation()}>
              <Checkbox
                aria-label={`Select conversation ${row.original.thread_id}`}
                checked={isSelected}
                onCheckedChange={(checked) => {
                  const shouldSelect = checked === true
                  setSelectedConversationIds((previous) => {
                    const next = new Set(previous)
                    if (shouldSelect) next.add(row.original.conversation_id)
                    else next.delete(row.original.conversation_id)
                    return next
                  })
                }}
              />
            </div>
          )
        },
      },
      {
        id: ID_COLUMN_ID,
        accessorFn: (row) => row.conversation_id,
        header: ({ column }) => <DataTableColumnHeader column={column} title='ID' />,
        enableHiding: false,
        cell: ({ row }) => (
          <span className='block font-mono text-xs font-medium leading-tight' title={row.original.conversation_id}>
            {shortId(row.original.conversation_id)}
          </span>
        ),
      },
      {
        accessorKey: 'title',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Title' />,
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
        id: 'social',
        accessorFn: (row) => formatSocial(row.social),
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Social' />,
        cell: ({ row }) => (
          <Badge className='bg-muted text-muted-foreground'>
            <span className='inline-flex items-center gap-1.5'>
              <FontAwesomeIcon icon={socialIcon(row.original.social)} className='size-3.5' />
              {formatSocial(row.original.social)}
            </span>
          </Badge>
        ),
      },
      {
        id: JOB_COLUMN_ID,
        accessorFn: (row) => row.job_name,
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Job' />,
        cell: ({ row }) => (
          <div className='leading-tight'>
            <span className='block max-w-[14rem] truncate text-sm font-medium' title={row.original.job_name}>
              {row.original.job_name}
            </span>
            <span className='block font-mono text-xs text-muted-foreground' title={row.original.job_id}>
              {shortId(row.original.job_id)}
            </span>
          </div>
        ),
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
        accessorKey: 'num_comments',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Comments' />,
      },
      {
        accessorKey: 'created_utc',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Created' />,
        cell: ({ row }) => formatThreadCreatedWhen(row.original),
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
                  <DropdownMenuItem onSelect={() => void performReply(thread)}>Generate Reply</DropdownMenuItem>
                  <DropdownMenuItem
                    disabled={!thread.has_reply || thread.user_has_commented}
                    onSelect={() => void performSend(thread)}
                  >
                    Send Reply
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2 text-destructive' onSelect={() => requestDeleteThread(thread)}>
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )
        },
      },
    ],
    [performReply, performSend, requestDeleteThread, selectedConversationIds]
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
      next = next.filter((id) => id !== ID_COLUMN_ID && id !== SELECT_COLUMN_ID)
      next.unshift(ID_COLUMN_ID)
      next.unshift(SELECT_COLUMN_ID)
      return next
    })
  }, [orderedColumnIds])

  const moveColumn = useCallback((fromId: string, toId: string): void => {
    if (
      fromId === toId ||
      fromId === ACTIONS_COLUMN_ID ||
      toId === ACTIONS_COLUMN_ID ||
      fromId === ID_COLUMN_ID ||
      toId === ID_COLUMN_ID ||
      fromId === SELECT_COLUMN_ID ||
      toId === SELECT_COLUMN_ID
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
      const selectIndex = next.indexOf(SELECT_COLUMN_ID)
      if (selectIndex >= 0) next.splice(selectIndex, 1)
      const idIndex = next.indexOf(ID_COLUMN_ID)
      if (idIndex >= 0) next.splice(idIndex, 1)
      next.unshift(ID_COLUMN_ID)
      next.unshift(SELECT_COLUMN_ID)
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
  const pendingDeleteBusy =
    pendingDeleteThreads.length > 1
      ? busyKey === 'threads:bulk-delete'
      : pendingDeleteThreads.length === 1 && busyKey === threadDeleteKey(pendingDeleteThreads[0])
  const threadActivityTimeline = useMemo(() => {
    if (!selectedThread) return []
    return [
      {
        label: 'Conversation discovered',
        detail: formatThreadCreatedWhen(selectedThread),
      },
      ...(selectedThread.search_updated_at
        ? [{
            label: 'Artifact updated',
            detail: formatWhen(selectedThread.search_updated_at),
          }]
        : []),
      {
        label: 'Reply generated',
        detail: selectedThread.has_reply ? 'Yes' : 'No',
      },
      {
        label: 'Reply sent',
        detail: selectedThread.user_has_commented ? 'Yes' : 'No',
      },
    ]
  }, [selectedThread])

  useEffect(() => {
    if (!sheetOpen || sheetTab !== 'compose' || !selectedThread) return
    if (busyKey === threadReplyKey(selectedThread) || busyKey === threadSendKey(selectedThread)) return
    const draftReply = replyInput.trim()
    const persistedReply = (selectedThread.reply ?? '').trim()
    if (!draftReply || draftReply === persistedReply) {
      if (replyAutosaveState !== 'saving') {
        setReplyAutosaveState('idle')
      }
      return
    }
    const autosaveTimer = window.setTimeout(() => {
      void persistReplyDraft(selectedThread, replyInput, { notifyOnError: false })
    }, 700)
    return () => window.clearTimeout(autosaveTimer)
  }, [busyKey, persistReplyDraft, replyAutosaveState, replyInput, selectedThread, sheetOpen, sheetTab])

  return (
    <>
      <Header fixed>
        <div className='ms-auto flex items-center gap-2'>
          <div className='inline-flex h-8 items-center overflow-hidden rounded-md border'>
            <Button
              variant='ghost'
              size='icon'
              className='h-full w-8 rounded-none border-r'
              onClick={() => void loadThreads()}
              disabled={loading}
              aria-label='Manual refresh threads'
              title='Manual refresh threads'
            >
              <RefreshCw className={cn('size-4', autoRefreshSeconds > 0 && 'animate-spin [animation-duration:2s]')} />
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant='ghost' size='sm' aria-label='Auto-refresh threads' className='h-full rounded-none gap-2 px-3'>
                  <ChevronDown className='size-4' />
                  <span>{autoRefreshLabel}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align='end'>
                {AUTO_REFRESH_OPTIONS.map((option) => (
                  <DropdownMenuItem key={option.value} onSelect={() => setAutoRefreshSeconds(option.value)}>
                    <span className='mr-3 inline-flex w-4 justify-center text-foreground'>
                      {autoRefreshSeconds === option.value ? '●' : ''}
                    </span>
                    {option.menuLabel}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </Header>

      <Main className='flex flex-1 flex-col gap-4 sm:gap-6'>
        <div>
          <h2 className='text-2xl font-bold tracking-tight'>Conversations</h2>
        </div>

        <DataTableToolbar
          table={table}
          searchPlaceholder='Search by thread, job id, subreddit or URL...'
          filters={[
            ...(socialFilterOptions.length > 0
              ? [{ columnId: 'social', title: 'Social', options: socialFilterOptions }]
              : []),
            ...(jobFilterOptions.length > 0
              ? [{ columnId: 'job', title: 'Job', options: jobFilterOptions }]
              : []),
            {
              columnId: 'status',
              title: 'Status',
              options: [
                { label: 'No reply', value: 'none' },
                { label: 'Draft', value: 'draft' },
                { label: 'Sent', value: 'sent' },
              ],
            },
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
                          header.column.id !== ACTIONS_COLUMN_ID &&
                            header.column.id !== SELECT_COLUMN_ID &&
                            'cursor-move',
                          header.column.id === ACTIONS_COLUMN_ID &&
                            'sticky right-0 z-20 bg-background text-right'
                        )}
                        draggable={
                          header.column.id !== ACTIONS_COLUMN_ID &&
                          header.column.id !== SELECT_COLUMN_ID
                        }
                        onDragStart={(event) => {
                          if (header.column.id === ACTIONS_COLUMN_ID || header.column.id === SELECT_COLUMN_ID) return
                          setDraggingColumnId(header.column.id)
                          event.dataTransfer.effectAllowed = 'move'
                          event.dataTransfer.setData('text/plain', header.column.id)
                        }}
                        onDragOver={(event) => {
                          if (!draggingColumnId || header.column.id === ACTIONS_COLUMN_ID || header.column.id === SELECT_COLUMN_ID) return
                          if (draggingColumnId === header.column.id) return
                          event.preventDefault()
                          event.dataTransfer.dropEffect = 'move'
                        }}
                        onDrop={(event) => {
                          if (header.column.id === ACTIONS_COLUMN_ID || header.column.id === SELECT_COLUMN_ID) return
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
                      onClick={() => openThreadSheet(row.original)}
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

      <SelectionActionBar
        selectedCount={selectedThreads.length}
        entityLabel='conversation'
        onClear={() => setSelectedConversationIds(new Set())}
        onDelete={() => requestDeleteSelectedThreads()}
        deleteLabel='Delete selected'
        deleteDisabled={busyKey !== null}
      />

      <ConfirmDialog
        open={pendingDeleteThreads.length > 0}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteThreads([])
        }}
        title='Delete conversation'
        desc={
          pendingDeleteThreads.length === 1
            ? `This action will permanently delete "${pendingDeleteThreads[0].title}".`
            : pendingDeleteThreads.length > 1
              ? `This action will permanently delete ${pendingDeleteThreads.length} conversations.`
              : 'This action will permanently delete this conversation.'
        }
        confirmText='Delete'
        destructive
        isLoading={pendingDeleteBusy}
        handleConfirm={() => void confirmDeleteThread()}
      />

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
              <Tabs
                value={sheetTab}
                onValueChange={(value) => setSheetTab(value as 'overview' | 'compose' | 'activity')}
                className='flex-1 overflow-hidden px-4 py-4'
              >
                <TabsList className='w-full justify-start'>
                  <TabsTrigger value='overview'>Overview</TabsTrigger>
                  <TabsTrigger value='compose'>Compose</TabsTrigger>
                  <TabsTrigger value='activity'>Activity</TabsTrigger>
                </TabsList>

                <TabsContent value='overview' className='min-h-0 overflow-y-auto pb-4 text-sm'>
                  <div className='space-y-4'>
                    <section className='rounded-lg border bg-muted/20 p-3 space-y-2'>
                      <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                        Post
                      </h3>
                      <Label htmlFor='selected-thread-body'>Thread Body</Label>
                      <Textarea
                        id='selected-thread-body'
                        value={selectedThread.selftext || ''}
                        readOnly
                        className='min-h-[12rem] max-h-[12rem] resize-none'
                      />
                    </section>

                    <section className='rounded-lg border bg-muted/20 p-3 space-y-2'>
                      <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                        Summary
                      </h3>
                      <Badge variant={selectedThread.user_has_commented ? 'default' : 'secondary'}>
                        {formatThreadStatus(selectedThread)}
                      </Badge>
                      <p className='text-xs text-muted-foreground'>
                        {`${selectedThread.semantic_similarity.toFixed(3)} similarity · ${selectedThread.score} score · ${selectedThread.num_comments} comments · ${formatThreadCreatedWhen(selectedThread)}`}
                      </p>
                      <div className='flex items-center gap-2'>
                        <Button size='sm' variant='outline' onClick={() => setSheetTab('compose')}>
                          Open compose
                        </Button>
                        <Button size='sm' variant='ghost' onClick={() => setSheetTab('activity')}>
                          Open activity
                        </Button>
                      </div>
                    </section>

                    <section className='rounded-lg border bg-muted/20 p-3 space-y-2'>
                      <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                        Link
                      </h3>
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
                    </section>
                  </div>
                </TabsContent>

                <TabsContent value='compose' className='min-h-0 overflow-y-auto pb-4 text-sm'>
                  <section className='rounded-lg border bg-muted/20 p-3 space-y-2'>
                    <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                      Compose reply
                    </h3>
                    <Label htmlFor='selected-thread-reply'>Reply Input</Label>
                    <Textarea
                      id='selected-thread-reply'
                      value={replyInput}
                      onChange={(event) => setReplyInput(event.target.value)}
                      className='min-h-[8rem] resize-y'
                      placeholder='Write or adjust the reply draft...'
                    />
                    {replyAutosaveState === 'saving' ? (
                      <p className='inline-flex items-center gap-1.5 text-xs text-muted-foreground'>
                        <Loader2 className='size-3.5 animate-spin' />
                        Autosaving...
                      </p>
                    ) : replyAutosaveState === 'saved' ? (
                      <p className='text-xs text-muted-foreground'>Saved automatically</p>
                    ) : replyAutosaveState === 'error' ? (
                      <p className='text-xs text-destructive'>Autosave failed</p>
                    ) : null}
                    <div className='flex items-center gap-2'>
                      <Button
                        size='sm'
                        variant='outline'
                        disabled={selectedReplyBusy}
                        onClick={() => void performReply(selectedThread)}
                      >
                        {selectedReplyBusy ? <Loader2 className='size-4 animate-spin' /> : null}
                        {selectedReplyBusy ? 'Generating Reply...' : 'Generate Reply'}
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
                        {selectedSendBusy ? <Loader2 className='size-4 animate-spin' /> : null}
                        {selectedSendBusy ? 'Sending Reply...' : 'Send Reply'}
                      </Button>
                    </div>
                  </section>
                </TabsContent>

                <TabsContent value='activity' className='min-h-0 overflow-y-auto pb-4 text-sm'>
                  <section className='rounded-lg border bg-muted/20 p-3 space-y-2'>
                    <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                      Activity timeline
                    </h3>
                    <div className='space-y-2'>
                      {threadActivityTimeline.map((event, index) => (
                        <p key={`${event.label}-${index}`} className='grid grid-cols-[12rem_1fr] gap-2 text-sm'>
                          <span className='font-medium text-muted-foreground'>{event.label}</span>
                          <span>{event.detail}</span>
                        </p>
                      ))}
                    </div>
                  </section>
                </TabsContent>
              </Tabs>
            )}
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
