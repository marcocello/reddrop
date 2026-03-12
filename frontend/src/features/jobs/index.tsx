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
import { Copy, Eye, MoreHorizontal, Play, Power, PowerOff, RefreshCw, Square, Trash2 } from 'lucide-react'
import { ConfirmDialog } from '@/components/confirm-dialog'
import { DataTableColumnHeader, DataTablePagination, DataTableToolbar } from '@/components/data-table'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader as ModalDialogHeader,
  DialogTitle as ModalDialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
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
import {
  reddropApi,
  type Job,
  type PersonaProfile,
  type UpsertJobPayload,
} from '@/lib/reddrop-api'
import { AutoRefreshMenu } from '@/components/auto-refresh-menu'

const ACTIONS_COLUMN_ID = 'actions'

function buildInitialForm(): UpsertJobPayload {
  return {
    name: '',
    topic: '',
    active: false,
    time_filter: 'week',
    subreddit_limit: 5,
    threads_limit: 10,
    replies_per_iteration: 3,
    max_runtime_minutes: 1440,
    personas: [],
  }
}

const arrayFilterFn: FilterFn<Job> = (row, columnId, value) => {
  const selected = Array.isArray(value) ? (value as string[]) : []
  if (selected.length === 0) {
    return true
  }
  return selected.includes(String(row.getValue(columnId)))
}

function formatWhen(value?: string | null): string {
  if (!value) return '-'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

function formatStatusLabel(status: string | undefined): string {
  const value = (status ?? 'idle').trim()
  if (!value) return 'Idle'
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function statusBadgeClass(status: string | undefined): string {
  if (status === 'searching') return 'bg-blue-600 text-white'
  if (status === 'replying') return 'bg-cyan-600 text-white'
  if (status === 'sending') return 'bg-emerald-600 text-white'
  if (status === 'idle') return 'bg-secondary text-secondary-foreground'
  if (status === 'failed') return 'bg-red-600 text-white'
  if (status === 'stopping') return 'bg-amber-500 text-black'
  if (status === 'inactive') return 'bg-muted text-muted-foreground'
  return 'bg-muted text-muted-foreground'
}

function activeBadgeClass(active: boolean): string {
  return active ? 'bg-emerald-600 text-white' : 'bg-muted text-muted-foreground'
}

function shortId(value: string): string {
  const normalized = value.trim()
  if (!normalized) return '-'
  return normalized.slice(0, 8)
}

function buildDuplicateJobName(baseName: string, existingNames: Set<string>): string {
  const normalized = baseName.trim().replace(/\s+/g, '-') || 'job'
  let attempt = `${normalized}-copy`
  let index = 2
  while (existingNames.has(attempt.toLowerCase())) {
    attempt = `${normalized}-copy-${index}`
    index += 1
  }
  return attempt
}

export function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const [sorting, setSorting] = useState<SortingState>([{ id: 'name', desc: false }])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [globalFilter, setGlobalFilter] = useState('')
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>([])
  const [draggingColumnId, setDraggingColumnId] = useState<string | null>(null)

  const [personaOptions, setPersonaOptions] = useState<PersonaProfile[]>([])

  const [sheetOpen, setSheetOpen] = useState(false)
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [pendingDeleteJob, setPendingDeleteJob] = useState<Job | null>(null)
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [form, setForm] = useState<UpsertJobPayload>(buildInitialForm)
  const [saving, setSaving] = useState(false)
  const [autoRefreshSeconds, setAutoRefreshSeconds] = useState(0)

  const load = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const payload = await reddropApi.listJobs()
      setJobs(payload.jobs)
      setSelectedJob((prev) => {
        if (!prev) return null
        return payload.jobs.find((item) => item.id === prev.id) ?? null
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load jobs'
      toast.error(message)
    } finally {
      setLoading(false)
    }
  }, [])

  const loadPersonas = useCallback(async (): Promise<void> => {
    try {
      const payload = await reddropApi.listPersonas()
      setPersonaOptions(payload.personas)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load personas'
      toast.error(message)
    }
  }, [])

  useEffect(() => {
    void load()
    void loadPersonas()
  }, [load, loadPersonas])

  useEffect(() => {
    if (autoRefreshSeconds <= 0) return
    const timer = window.setInterval(() => {
      void load()
    }, autoRefreshSeconds * 1000)
    return () => window.clearInterval(timer)
  }, [autoRefreshSeconds, load])

  const statusFilterOptions = useMemo(() => {
    const statuses = new Set<string>()
    jobs.forEach((job) => statuses.add(job.runtime?.status ?? 'idle'))
    return Array.from(statuses)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ label: formatStatusLabel(value), value }))
  }, [jobs])

  const activeFilterOptions = useMemo(() => {
    const states = new Set<string>()
    jobs.forEach((job) => states.add(job.active ? 'active' : 'inactive'))
    return Array.from(states)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ label: formatStatusLabel(value), value }))
  }, [jobs])

  const openCreateDialog = useCallback((): void => {
    setSelectedJob(null)
    setForm(buildInitialForm())
    setSheetOpen(false)
    setCreateDialogOpen(true)
  }, [])

  const openJobSheet = useCallback((job: Job): void => {
    setSelectedJob(job)
    setSheetOpen(true)
  }, [])

  const togglePersona = useCallback((personaName: string, checked: boolean | 'indeterminate'): void => {
    setForm((prev) => {
      if (checked === true) {
        if (prev.personas.includes(personaName)) {
          return prev
        }
        return { ...prev, personas: [...prev.personas, personaName] }
      }
      return {
        ...prev,
        personas: prev.personas.filter((item) => item !== personaName),
      }
    })
  }, [])

  const createJob = useCallback(async (): Promise<void> => {
    setSaving(true)
    try {
      await reddropApi.createJob(form)
      toast.success(`Created job ${form.name}`)
      await load()
      setCreateDialogOpen(false)
      setSelectedJob(null)
      setForm(buildInitialForm())
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create job'
      toast.error(message)
    } finally {
      setSaving(false)
    }
  }, [form, load])

  const runAction = useCallback(async (job: Job, action: 'start' | 'stop'): Promise<void> => {
    const key = `${job.id}:${action}`
    setBusyKey(key)
    try {
      if (action === 'start') await reddropApi.startJob(job.id)
      if (action === 'stop') await reddropApi.stopJob(job.id)
      toast.success(`Job ${job.name}: ${action} run requested`)
      await load()
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to ${action} ${job.name}`
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [load])

  const activationAction = useCallback(async (job: Job, action: 'activate' | 'deactivate'): Promise<void> => {
    const key = `${job.id}:${action}`
    setBusyKey(key)
    try {
      if (action === 'activate') await reddropApi.activateJob(job.id)
      if (action === 'deactivate') await reddropApi.deactivateJob(job.id)
      toast.success(`Job ${job.name}: ${action} requested`)
      await load()
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to ${action} ${job.name}`
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [load])

  const deleteJob = useCallback(async (job: Job): Promise<void> => {
    const key = `${job.id}:delete`
    setBusyKey(key)
    try {
      await reddropApi.deleteJob(job.id)
      toast.success(`Deleted job ${job.name}`)
      await load()
      setSelectedJob((prev) => (prev?.id === job.id ? null : prev))
      if (selectedJob?.id === job.id) {
        setSheetOpen(false)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to delete ${job.name}`
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [load, selectedJob?.id])

  const openDuplicateJobDialog = useCallback((job: Job): void => {
    const existingNames = new Set(jobs.map((item) => item.name.trim().toLowerCase()))
    const duplicateName = buildDuplicateJobName(job.name, existingNames)
    setSelectedJob(null)
    setSheetOpen(false)
    setForm({
      name: duplicateName,
      topic: job.topic,
      active: job.active,
      time_filter: job.time_filter,
      subreddit_limit: job.subreddit_limit,
      threads_limit: job.threads_limit,
      replies_per_iteration: job.replies_per_iteration,
      max_runtime_minutes: job.max_runtime_minutes,
      personas: [...job.personas],
    })
    setCreateDialogOpen(true)
  }, [jobs])

  const requestDeleteJob = useCallback((job: Job): void => {
    setPendingDeleteJob(job)
  }, [])

  const confirmDeleteJob = useCallback(async (): Promise<void> => {
    if (!pendingDeleteJob) return
    await deleteJob(pendingDeleteJob)
    setPendingDeleteJob(null)
  }, [deleteJob, pendingDeleteJob])

  const columns = useMemo<ColumnDef<Job>[]>(
    () => [
      {
        accessorKey: 'id',
        header: ({ column }) => <DataTableColumnHeader column={column} title='ID' />,
        cell: ({ row }) => (
          <span className='font-mono text-xs' title={row.original.id}>
            {shortId(row.original.id)}
          </span>
        ),
      },
      {
        accessorKey: 'name',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Name' />,
        enableHiding: false,
        cell: ({ row }) => <span className='font-medium'>{row.original.name}</span>,
      },
      {
        id: 'active',
        accessorFn: (row) => (row.active ? 'active' : 'inactive'),
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Active' />,
        cell: ({ row }) => (
          <Badge className={activeBadgeClass(row.original.active)}>
            {row.original.active ? 'Active' : 'Inactive'}
          </Badge>
        ),
      },
      {
        id: 'status',
        accessorFn: (row) => row.runtime?.status ?? 'idle',
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Status' />,
        cell: ({ row }) => (
          <Badge className={statusBadgeClass(row.original.runtime?.status)}>
            {formatStatusLabel(row.original.runtime?.status)}
          </Badge>
        ),
      },
      {
        accessorKey: 'topic',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Topic' />,
        cell: ({ row }) => (
          <span className='block max-w-[24rem] truncate' title={row.original.topic}>
            {row.original.topic}
          </span>
        ),
      },
      {
        accessorKey: 'last_run_status',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Last Run' />,
        cell: ({ row }) => formatStatusLabel(row.original.last_run_status ?? 'never'),
      },
      {
        accessorKey: 'successful_runs',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Successful Runs' />,
        cell: ({ row }) => Number(row.original.successful_runs ?? 0),
      },
      {
        accessorKey: 'threads_found_total',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Threads Found' />,
        cell: ({ row }) => Number(row.original.threads_found_total ?? 0),
      },
      {
        accessorKey: 'threads_replied_total',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Threads Replied' />,
        cell: ({ row }) => Number(row.original.threads_replied_total ?? 0),
      },
      {
        id: ACTIONS_COLUMN_ID,
        enableSorting: false,
        enableHiding: false,
        cell: ({ row }) => {
          const job = row.original
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
                  <DropdownMenuItem className='gap-2' onSelect={() => openJobSheet(job)}>
                    <Eye className='size-4' />
                    View details
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2' onSelect={() => openDuplicateJobDialog(job)}>
                    <Copy className='size-4' />
                    Duplicate
                  </DropdownMenuItem>
                  {job.active ? (
                    <DropdownMenuItem className='gap-2' onSelect={() => void activationAction(job, 'deactivate')}>
                      <PowerOff className='size-4' />
                      Deactivate
                    </DropdownMenuItem>
                  ) : (
                    <DropdownMenuItem className='gap-2' onSelect={() => void activationAction(job, 'activate')}>
                      <Power className='size-4' />
                      Activate
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuItem className='gap-2' onSelect={() => void runAction(job, 'start')}>
                    <Play className='size-4' />
                    Start run
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2' onSelect={() => void runAction(job, 'stop')}>
                    <Square className='size-4' />
                    Stop run
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2 text-destructive' onSelect={() => requestDeleteJob(job)}>
                    <Trash2 className='size-4' />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )
        },
      },
    ],
    [activationAction, openDuplicateJobDialog, openJobSheet, requestDeleteJob, runAction]
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
      const next = previous.filter((id) => orderedColumnIds.includes(id))
      orderedColumnIds.forEach((id) => {
        if (!previousSet.has(id)) next.push(id)
      })
      return next
    })
  }, [orderedColumnIds])

  const moveColumn = useCallback((fromId: string, toId: string): void => {
    if (fromId === toId || fromId === ACTIONS_COLUMN_ID || toId === ACTIONS_COLUMN_ID) return
    setColumnOrder((previous) => {
      const next = previous.length > 0 ? [...previous] : [...orderedColumnIds]
      const fromIndex = next.indexOf(fromId)
      const toIndex = next.indexOf(toId)
      if (fromIndex < 0 || toIndex < 0) return next
      const [moved] = next.splice(fromIndex, 1)
      next.splice(toIndex, 0, moved)
      return next
    })
  }, [orderedColumnIds])

  const table = useReactTable({
    data: jobs,
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
      const job = row.original
      return (
        job.id.toLowerCase().includes(query) ||
        job.name.toLowerCase().includes(query) ||
        job.topic.toLowerCase().includes(query) ||
        (job.active ? 'active' : 'inactive').includes(query) ||
        (job.runtime?.status ?? 'idle').toLowerCase().includes(query) ||
        (job.last_run_status ?? 'never').toLowerCase().includes(query)
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

  const selectedStartKey = selectedJob ? `${selectedJob.id}:start` : ''
  const selectedStopKey = selectedJob ? `${selectedJob.id}:stop` : ''
  const selectedActivateKey = selectedJob ? `${selectedJob.id}:activate` : ''
  const selectedDeactivateKey = selectedJob ? `${selectedJob.id}:deactivate` : ''
  const selectedDeleteKey = selectedJob ? `${selectedJob.id}:delete` : ''
  const selectedLogs = selectedJob?.runtime?.logs ?? []

  const renderJobFormFields = (prefix: string) => (
    <>
      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-job-name`}>Name</Label>
        <Input
          id={`${prefix}-job-name`}
          value={form.name}
          onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
          placeholder='growth'
          required
        />
      </div>

      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-job-active`}>Active</Label>
        <Select
          value={form.active ? 'active' : 'inactive'}
          onValueChange={(value) =>
            setForm((prev) => ({ ...prev, active: value === 'active' }))
          }
        >
          <SelectTrigger id={`${prefix}-job-active`} className='w-full'>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value='active'>Active</SelectItem>
            <SelectItem value='inactive'>Inactive</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-job-topic`}>Topic</Label>
        <Textarea
          id={`${prefix}-job-topic`}
          value={form.topic}
          onChange={(event) => setForm((prev) => ({ ...prev, topic: event.target.value }))}
          placeholder='AI launch strategy'
          rows={4}
          required
        />
      </div>

      <div className='grid gap-3 sm:grid-cols-3'>
        <div className='space-y-2'>
          <Label htmlFor={`${prefix}-job-time-filter`}>Time filter</Label>
          <Select
            value={form.time_filter}
            onValueChange={(value) => setForm((prev) => ({ ...prev, time_filter: value }))}
          >
            <SelectTrigger id={`${prefix}-job-time-filter`} className='w-full'>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value='hour'>hour</SelectItem>
              <SelectItem value='day'>day</SelectItem>
              <SelectItem value='week'>week</SelectItem>
              <SelectItem value='month'>month</SelectItem>
              <SelectItem value='year'>year</SelectItem>
              <SelectItem value='all'>all</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className='space-y-2'>
          <Label htmlFor={`${prefix}-subreddit-limit`}>Subreddit limit</Label>
          <Input
            id={`${prefix}-subreddit-limit`}
            type='number'
            min={1}
            value={form.subreddit_limit}
            onChange={(event) =>
              setForm((prev) => ({
                ...prev,
                subreddit_limit: Number(event.target.value || 1),
              }))
            }
          />
        </div>

        <div className='space-y-2'>
          <Label htmlFor={`${prefix}-threads-limit`}>Threads limit</Label>
          <Input
            id={`${prefix}-threads-limit`}
            type='number'
            min={1}
            value={form.threads_limit}
            onChange={(event) =>
              setForm((prev) => ({
                ...prev,
                threads_limit: Number(event.target.value || 1),
              }))
            }
          />
        </div>
      </div>

      <div className='grid gap-3 sm:grid-cols-2'>
        <div className='space-y-2'>
          <Label htmlFor={`${prefix}-replies-per-iteration`}>Replies per iteration</Label>
          <Input
            id={`${prefix}-replies-per-iteration`}
            type='number'
            min={1}
            value={form.replies_per_iteration}
            onChange={(event) =>
              setForm((prev) => ({
                ...prev,
                replies_per_iteration: Number(event.target.value || 1),
              }))
            }
          />
        </div>

        <div className='space-y-2'>
          <Label htmlFor={`${prefix}-max-runtime-minutes`}>Run interval minutes</Label>
          <Input
            id={`${prefix}-max-runtime-minutes`}
            type='number'
            min={1}
            value={form.max_runtime_minutes}
            onChange={(event) =>
              setForm((prev) => ({
                ...prev,
                max_runtime_minutes: Number(event.target.value || 1),
              }))
            }
          />
        </div>
      </div>

      <div className='space-y-2'>
        <Label id={`${prefix}-personas-label`}>Select personas</Label>
        {personaOptions.length > 0 ? (
          <div
            role='group'
            aria-labelledby={`${prefix}-personas-label`}
            className='grid gap-2 rounded-md border p-3 sm:grid-cols-2'
          >
            {personaOptions.map((persona) => (
              <label
                key={persona.name}
                htmlFor={`${prefix}-persona-${persona.name}`}
                className='flex cursor-pointer items-start gap-3 rounded-md border border-transparent p-2 hover:border-border'
              >
                <Checkbox
                  id={`${prefix}-persona-${persona.name}`}
                  checked={form.personas.includes(persona.name)}
                  onCheckedChange={(checked) => togglePersona(persona.name, checked)}
                />
                <span className='text-sm'>{persona.name}</span>
              </label>
            ))}
          </div>
        ) : (
          <p className='text-sm text-muted-foreground'>
            No personas found. Open the Personas page to add one.
          </p>
        )}
      </div>
    </>
  )

  return (
    <>
      <Header fixed>
        <div className='ms-auto flex items-center gap-2'>
          <Button size='sm' onClick={openCreateDialog}>
            Create job
          </Button>
          <AutoRefreshMenu
            valueSeconds={autoRefreshSeconds}
            onValueChange={setAutoRefreshSeconds}
          />
          <Button
            variant='outline'
            size='icon'
            className='size-8'
            onClick={() => void load()}
            disabled={loading}
            aria-label='Refresh jobs'
            title='Refresh jobs'
          >
            <RefreshCw className='size-4' />
          </Button>
        </div>
      </Header>

      <Main className='flex flex-1 flex-col gap-4 sm:gap-6'>
        <div>
          <h2 className='text-2xl font-bold tracking-tight'>Jobs</h2>
        </div>

        <div className='space-y-4'>
          <DataTableToolbar
            table={table}
            searchPlaceholder='Search by id, name, topic or status...'
            filters={[
              ...(activeFilterOptions.length > 0
                ? [{ columnId: 'active', title: 'Active', options: activeFilterOptions }]
                : []),
              ...(statusFilterOptions.length > 0
                ? [{ columnId: 'status', title: 'Status', options: statusFilterOptions }]
                : []),
            ]}
          />

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
                      onClick={() => openJobSheet(row.original)}
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
                      No jobs configured.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>

          <DataTablePagination table={table} />
        </div>
      </Main>

      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className='sm:max-w-2xl'>
          <ModalDialogHeader>
            <ModalDialogTitle>Create job</ModalDialogTitle>
            <DialogDescription>
              Configure discovery and reply settings for a new job.
            </DialogDescription>
          </ModalDialogHeader>

          <div className='max-h-[70vh] space-y-4 overflow-y-auto pr-1'>
            {renderJobFormFields('dialog')}
          </div>

          <DialogFooter>
            <Button
              type='button'
              variant='outline'
              onClick={() => setCreateDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button type='button' onClick={() => void createJob()} disabled={saving}>
              {saving ? 'Saving...' : 'Create job'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={pendingDeleteJob !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteJob(null)
        }}
        title='Delete job'
        desc={
          pendingDeleteJob
            ? `This action will permanently delete "${pendingDeleteJob.name}".`
            : 'This action will permanently delete this job.'
        }
        confirmText='Delete'
        destructive
        isLoading={pendingDeleteJob !== null && busyKey === `${pendingDeleteJob.id}:delete`}
        handleConfirm={() => void confirmDeleteJob()}
      />

      <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
        <SheetContent side='right' className='sm:max-w-xl p-0'>
          <div className='flex h-full flex-col'>
            <SheetHeader className='px-4 pt-4'>
              <SheetTitle>{selectedJob ? `Job · ${selectedJob.name}` : 'Job details'}</SheetTitle>
              <SheetDescription>
                Runtime controls and job details.
              </SheetDescription>
            </SheetHeader>

            <div className='sticky top-0 z-10 mt-3 flex items-center gap-2 border-y bg-background px-4 py-3'>
              {selectedJob ? (
                <>
                  <Button
                    size='sm'
                    variant='outline'
                    onClick={() =>
                      void activationAction(selectedJob, selectedJob.active ? 'deactivate' : 'activate')
                    }
                    disabled={
                      busyKey === selectedActivateKey ||
                      busyKey === selectedDeactivateKey
                    }
                  >
                    {selectedJob.active ? 'Deactivate' : 'Activate'}
                  </Button>
                  <Button
                    size='sm'
                    variant='outline'
                    onClick={() => void runAction(selectedJob, 'start')}
                    disabled={busyKey === selectedStartKey}
                  >
                    <Play className='size-4' />
                    Start run
                  </Button>
                  <Button
                    size='sm'
                    variant='outline'
                    onClick={() => void runAction(selectedJob, 'stop')}
                    disabled={busyKey === selectedStopKey}
                  >
                    <Square className='size-4' />
                    Stop run
                  </Button>
                  <Button
                    size='sm'
                    variant='outline'
                    onClick={() => requestDeleteJob(selectedJob)}
                    disabled={busyKey === selectedDeleteKey}
                  >
                    <Trash2 className='size-4' />
                    Delete
                  </Button>
                </>
              ) : null}
            </div>

            <div className='flex-1 space-y-4 overflow-y-auto px-4 py-4'>
              {selectedJob ? (
                <div className='grid gap-3 text-sm'>
                  <p><span className='font-medium'>ID:</span> <span className='font-mono'>{selectedJob.id}</span></p>
                  <p><span className='font-medium'>Name:</span> {selectedJob.name}</p>
                  <p><span className='font-medium'>Topic:</span> {selectedJob.topic}</p>
                  <p><span className='font-medium'>Active:</span> {selectedJob.active ? 'Yes' : 'No'}</p>
                  <p><span className='font-medium'>Current status:</span> {formatStatusLabel(selectedJob.runtime?.status)}</p>
                  <p><span className='font-medium'>Last run status:</span> {formatStatusLabel(selectedJob.last_run_status ?? 'never')}</p>
                  <p><span className='font-medium'>Successful runs:</span> {selectedJob.successful_runs ?? 0}</p>
                  <p><span className='font-medium'>Threads found:</span> {selectedJob.threads_found_total ?? 0}</p>
                  <p><span className='font-medium'>Threads replied:</span> {selectedJob.threads_replied_total ?? 0}</p>
                  <p><span className='font-medium'>Time filter:</span> {selectedJob.time_filter}</p>
                  <p><span className='font-medium'>Subreddit limit:</span> {selectedJob.subreddit_limit}</p>
                  <p><span className='font-medium'>Threads limit:</span> {selectedJob.threads_limit}</p>
                  <p><span className='font-medium'>Replies per iteration:</span> {selectedJob.replies_per_iteration}</p>
                  <p><span className='font-medium'>Run interval minutes:</span> {selectedJob.max_runtime_minutes}</p>
                  <p><span className='font-medium'>Personas:</span> {selectedJob.personas.length > 0 ? selectedJob.personas.join(', ') : '-'}</p>
                </div>
              ) : null}
              <div className='space-y-2'>
                <Label>Runtime logs</Label>
                <div className='max-h-56 overflow-y-auto rounded-md border bg-muted/20 p-3'>
                  {selectedLogs.length > 0 ? (
                    <div className='space-y-2'>
                      {selectedLogs.map((entry, index) => (
                        <p key={`${entry.at}-${index}`} className='font-mono text-xs leading-5 text-muted-foreground'>
                          [{formatWhen(entry.at)}] {entry.message}
                        </p>
                      ))}
                    </div>
                  ) : (
                    <p className='text-sm text-muted-foreground'>No runtime logs yet.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
