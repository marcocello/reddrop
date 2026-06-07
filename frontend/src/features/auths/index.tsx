import { useCallback, useEffect, useMemo, useState } from 'react'
import { faLinkedinIn, faRedditAlien } from '@fortawesome/free-brands-svg-icons'
import type { IconDefinition } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  type ColumnDef,
  type ColumnFiltersState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  type RowSelectionState,
  type SortingState,
  useReactTable,
  type VisibilityState,
} from '@tanstack/react-table'
import { CheckCircle2, CircleX, Copy, MoreHorizontal, Pencil, Play, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { ConfirmDialog } from '@/components/confirm-dialog'
import { DataTableColumnHeader, DataTablePagination, DataTableToolbar } from '@/components/data-table'
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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
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
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import { reddropApi, type AuthProfile, type Job, type SocialPlatform } from '@/lib/reddrop-api'

const ACTIONS_COLUMN_ID = 'actions'
const SELECT_COLUMN_ID = 'select'

function buildDefaultAuth(): AuthProfile {
  return {
    id: '',
    name: '',
    social: 'reddit',
    status: undefined,
    reddit_client_id: '',
    reddit_client_secret: '',
    reddit_user_agent: 'Reddrop:v1.0',
    reddit_username: '',
    reddit_password: '',
    linkedin_username: '',
    linkedin_email: '',
    linkedin_password: '',
    linkedin_headless: 'true',
    linkedin_user_data_dir: '',
    linkedin_wait_seconds: '120',
  }
}

function createAuthId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  const random = Math.random().toString(16).slice(2, 10).padEnd(8, '0')
  return `00000000-0000-4000-8000-${random}`
}

function shortId(value: string): string {
  const normalized = value.trim()
  if (!normalized) return '-'
  return normalized.slice(0, 8)
}

function normalizeToken(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function buildLinkedinProfileDir(label: string, authId: string): string {
  const namePart = normalizeToken(label).slice(0, 24) || 'linkedin'
  const idPart = authId.slice(0, 8) || 'auth'
  return `.reddrop/auth/linkedin/${namePart}-${idPart}`
}

function normalizeAuth(payload: AuthProfile): AuthProfile {
  const id = payload.id.trim() || createAuthId()
  const social: SocialPlatform = payload.social === 'linkedin' ? 'linkedin' : 'reddit'
  const redditUsername = payload.reddit_username.trim()
  const generatedName = social === 'reddit'
    ? (redditUsername || `reddit-${id.slice(0, 8)}`)
    : `linkedin-${id.slice(0, 8)}`
  return {
    ...payload,
    id,
    name: generatedName,
    social,
    status: payload.status,
    reddit_client_id: payload.reddit_client_id.trim(),
    reddit_client_secret: payload.reddit_client_secret.trim(),
    reddit_user_agent: payload.reddit_user_agent.trim() || 'Reddrop:v1.0',
    reddit_username: redditUsername,
    reddit_password: payload.reddit_password.trim(),
    linkedin_username: payload.linkedin_username.trim(),
    linkedin_email: payload.linkedin_email.trim(),
    linkedin_password: payload.linkedin_password.trim(),
    linkedin_headless: 'true',
    linkedin_user_data_dir: social === 'linkedin'
      ? buildLinkedinProfileDir(generatedName, id)
      : payload.linkedin_user_data_dir.trim(),
    linkedin_wait_seconds: '120',
  }
}

function socialBadgeClass(_value: SocialPlatform): string {
  return 'bg-muted text-muted-foreground'
}

function formatSocial(value: SocialPlatform): string {
  return value === 'linkedin' ? 'LinkedIn' : 'Reddit'
}

function socialIcon(platform: SocialPlatform): IconDefinition {
  return platform === 'linkedin' ? faLinkedinIn : faRedditAlien
}

function authUsername(item: AuthProfile): string {
  if (item.social === 'reddit') {
    return item.reddit_username.trim() || '-'
  }
  if (item.social === 'linkedin') {
    return item.linkedin_username.trim() || '-'
  }
  return '-'
}

function authLabel(item: AuthProfile): string {
  const username = authUsername(item)
  if (username !== '-') return username
  return shortId(item.id)
}

function statusBadgeClass(status: string | undefined): string {
  if (status === 'ok') return 'bg-emerald-600 text-white'
  return 'bg-red-600 text-white'
}

export function AuthsPage() {
  const [auths, setAuths] = useState<AuthProfile[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const [sorting, setSorting] = useState<SortingState>([{ id: 'id', desc: false }])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [globalFilter, setGlobalFilter] = useState('')
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({})

  const [sheetOpen, setSheetOpen] = useState(false)
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [pendingDeleteAuth, setPendingDeleteAuth] = useState<{ auth: AuthProfile; relatedJobs: Job[] } | null>(null)
  const [pendingDeleteAuths, setPendingDeleteAuths] = useState<{ auths: AuthProfile[]; relatedJobs: Job[] } | null>(null)
  const [selectedAuth, setSelectedAuth] = useState<AuthProfile | null>(null)
  const [form, setForm] = useState<AuthProfile>(buildDefaultAuth)

  const loadAuths = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const [authsPayload, jobsPayload] = await Promise.all([
        reddropApi.listAuths(),
        reddropApi.listJobs(),
      ])
      setAuths(authsPayload.auths)
      setJobs(jobsPayload.jobs)
      setSelectedAuth((prev) => {
        if (!prev) return null
        return authsPayload.auths.find((item) => item.id === prev.id) ?? null
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load auths'
      toast.error(message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadAuths()
  }, [loadAuths])

  const persistAuths = useCallback(async (next: AuthProfile[], successMessage: string): Promise<void> => {
    setSaving(true)
    try {
      const payload = await reddropApi.saveAuths({ auths: next })
      setAuths(payload.auths)
      toast.success(successMessage)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save auths'
      toast.error(message)
    } finally {
      setSaving(false)
      setBusyKey(null)
    }
  }, [])

  const openCreateDialog = useCallback((): void => {
    setSelectedAuth(null)
    setForm(buildDefaultAuth())
    setSheetOpen(false)
    setCreateDialogOpen(true)
  }, [])

  const openAuthSheet = useCallback((auth: AuthProfile): void => {
    setSelectedAuth(auth)
    setForm(auth)
    setSheetOpen(true)
  }, [])

  const saveFromSheet = useCallback(async (): Promise<void> => {
    const normalized = normalizeAuth(form)
    if (normalized.social === 'reddit' && !normalized.reddit_client_id.trim()) {
      toast.error('Client ID is required for Reddit auth.')
      return
    }
    if (normalized.social === 'reddit' && !normalized.reddit_client_secret.trim()) {
      toast.error('Client secret is required for Reddit auth.')
      return
    }
    if (normalized.social === 'reddit' && !normalized.reddit_username.trim()) {
      toast.error('Username is required for Reddit auth.')
      return
    }
    if (normalized.social === 'reddit' && !normalized.reddit_password.trim()) {
      toast.error('Password is required for Reddit auth.')
      return
    }
    if (!normalized.id) {
      toast.error('Auth ID must include letters or numbers.')
      return
    }

    const editingId = selectedAuth?.id.trim().toLowerCase() ?? null
    const targetId = normalized.id.toLowerCase()
    const duplicate = auths.some((item) => {
      const current = item.id.trim().toLowerCase()
      if (editingId !== null && current === editingId) {
        return false
      }
      return current === targetId
    })
    if (duplicate) {
      toast.error(`Auth ID already exists: ${normalized.id}`)
      return
    }

    const next = selectedAuth === null
      ? [...auths, normalized]
      : auths.map((item) => (item.id === selectedAuth.id ? normalized : item))

    if (selectedAuth === null && normalized.social === 'linkedin') {
      setSaving(true)
      let persisted = false
      try {
        const payload = await reddropApi.saveAuths({ auths: next })
        setAuths(payload.auths)
        persisted = true
        await reddropApi.runLinkedinAuthLogin(normalized.id)
        await loadAuths()
        toast.success(`Created auth ${authLabel(normalized)}`)
        toast.success(`Run LinkedIn login for ${authLabel(normalized)}`)
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to run LinkedIn login'
        if (persisted) {
          try {
            const rollback = await reddropApi.saveAuths({ auths })
            setAuths(rollback.auths)
            toast.error('LinkedIn login failed. Auth was not created.')
          } catch (rollbackError) {
            const rollbackMessage = rollbackError instanceof Error ? rollbackError.message : 'Failed to rollback auth creation'
            toast.error(`LinkedIn login failed and rollback failed: ${rollbackMessage}`)
          }
        } else {
          toast.error(message)
        }
        return
      } finally {
        setSaving(false)
        setBusyKey(null)
      }
      setSelectedAuth(null)
      setSheetOpen(false)
      setCreateDialogOpen(false)
      return
    }

    await persistAuths(
      next,
      selectedAuth === null
        ? `Created auth ${authLabel(normalized)}`
        : `Updated auth ${authLabel(normalized)}`
    )
    setSelectedAuth(selectedAuth === null ? null : normalized)
    setSheetOpen(false)
    setCreateDialogOpen(false)
  }, [auths, form, loadAuths, persistAuths, selectedAuth])

  const relatedJobsForAuth = useCallback((authId: string): Job[] => {
    const normalized = authId.trim()
    if (!normalized) return []
    const relatedById = new Map<string, Job>()
    const direct = jobs.filter((item) => (item.auth_id ?? '').trim() === normalized)
    const seedIds = new Set(direct.map((item) => item.id))
    for (const item of direct) {
      relatedById.set(item.id, item)
    }
    for (const item of jobs) {
      if (item.job_type !== 'reply') continue
      if (!item.source_job_id) continue
      if (seedIds.has(item.source_job_id)) {
        relatedById.set(item.id, item)
      }
    }
    return Array.from(relatedById.values())
  }, [jobs])

  const deleteAuth = useCallback(async (auth: AuthProfile, cascade: boolean): Promise<void> => {
    const key = `${auth.id}:delete`
    setBusyKey(key)
    try {
      if (cascade) await reddropApi.deleteAuthCascade(auth.id)
      else await reddropApi.deleteAuth(auth.id)
      toast.success(`Deleted auth ${authLabel(auth)}`)
      await loadAuths()
      if (selectedAuth?.id === auth.id) {
        setSelectedAuth(null)
        setSheetOpen(false)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to delete ${authLabel(auth)}`
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [loadAuths, selectedAuth?.id])

  const requestDeleteSelectedAuths = useCallback((): void => {
    const selectedIds = new Set(
      Object.entries(rowSelection)
        .filter(([, selected]) => Boolean(selected))
        .map(([id]) => id)
    )
    const selected = auths.filter((item) => selectedIds.has(item.id))
    if (selected.length === 0) return
    const relatedById = new Map<string, Job>()
    for (const auth of selected) {
      for (const related of relatedJobsForAuth(auth.id)) {
        relatedById.set(related.id, related)
      }
    }
    setPendingDeleteAuths({
      auths: selected,
      relatedJobs: Array.from(relatedById.values()),
    })
  }, [auths, relatedJobsForAuth, rowSelection])

  const deleteSelectedAuths = useCallback(async (): Promise<void> => {
    if (!pendingDeleteAuths) return
    setBusyKey('bulk-delete')
    let deletedCount = 0
    let firstError = ''
    for (const auth of pendingDeleteAuths.auths) {
      try {
        const related = relatedJobsForAuth(auth.id)
        if (related.length > 0) await reddropApi.deleteAuthCascade(auth.id)
        else await reddropApi.deleteAuth(auth.id)
        deletedCount += 1
      } catch (error) {
        if (!firstError) {
          firstError = error instanceof Error ? error.message : `Failed to delete ${authLabel(auth)}`
        }
      }
    }
    setPendingDeleteAuths(null)
    setRowSelection({})
    await loadAuths()
    if (deletedCount > 0) {
      toast.success(`Deleted ${deletedCount} auth object${deletedCount === 1 ? '' : 's'}`)
    }
    if (firstError) {
      toast.error(firstError)
    }
    setBusyKey(null)
  }, [loadAuths, pendingDeleteAuths, relatedJobsForAuth])

  const duplicateAuth = useCallback(async (auth: AuthProfile): Promise<void> => {
    const key = `${auth.id}:duplicate`
    setBusyKey(key)
    const duplicateId = createAuthId()
    const normalized = normalizeAuth({
      ...auth,
      id: duplicateId,
      reddit_username: auth.social === 'reddit' ? `${auth.reddit_username.trim() || 'user'}-copy` : auth.reddit_username,
    })
    const next = [
      ...auths,
      normalized,
    ]
    await persistAuths(next, `Duplicated auth ${authLabel(auth)} as ${authLabel(normalized)}`)
  }, [auths, persistAuths])

  const runLinkedinLogin = useCallback(async (auth: AuthProfile): Promise<void> => {
    if (auth.social !== 'linkedin') {
      toast.error('LinkedIn login is available only for LinkedIn auth objects.')
      return
    }
    const key = `${auth.id}:login`
    setBusyKey(key)
    try {
      await reddropApi.runLinkedinAuthLogin(auth.id)
      await loadAuths()
      toast.success(`Run LinkedIn login for ${authLabel(auth)}`)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to run LinkedIn login'
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [loadAuths])

  const requestDeleteAuth = useCallback((auth: AuthProfile): void => {
    setPendingDeleteAuth({
      auth,
      relatedJobs: relatedJobsForAuth(auth.id),
    })
  }, [relatedJobsForAuth])

  const confirmDeleteAuth = useCallback(async (): Promise<void> => {
    if (!pendingDeleteAuth) return
    await deleteAuth(pendingDeleteAuth.auth, pendingDeleteAuth.relatedJobs.length > 0)
    setPendingDeleteAuth(null)
  }, [deleteAuth, pendingDeleteAuth])

  const confirmDeleteSelectedAuths = useCallback(async (): Promise<void> => {
    await deleteSelectedAuths()
  }, [deleteSelectedAuths])

  const columns = useMemo<ColumnDef<AuthProfile>[]>(
    () => [
      {
        id: SELECT_COLUMN_ID,
        enableSorting: false,
        enableHiding: false,
        header: ({ table }) => (
          <Checkbox
            aria-label='Select all auths'
            checked={
              table.getIsAllPageRowsSelected() ||
              (table.getIsSomePageRowsSelected() && 'indeterminate')
            }
            onCheckedChange={(value) => table.toggleAllPageRowsSelected(Boolean(value))}
          />
        ),
        cell: ({ row }) => (
          <Checkbox
            aria-label={`Select auth ${row.original.id}`}
            checked={row.getIsSelected()}
            onCheckedChange={(value) => row.toggleSelected(Boolean(value))}
            onClick={(event) => event.stopPropagation()}
          />
        ),
      },
      {
        accessorKey: 'id',
        header: ({ column }) => <DataTableColumnHeader column={column} title='ID' />,
        cell: ({ row }) => <span className='font-mono text-xs'>{shortId(row.original.id)}</span>,
      },
      {
        id: 'username',
        accessorFn: (row) => authUsername(row),
        header: ({ column }) => <DataTableColumnHeader column={column} title='Username' />,
        enableHiding: false,
        cell: ({ row }) => <span className='font-medium'>{authUsername(row.original)}</span>,
      },
      {
        accessorKey: 'social',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Social' />,
        cell: ({ row }) => (
          <Badge className={socialBadgeClass(row.original.social)}>
            <span className='inline-flex items-center gap-1.5'>
              <FontAwesomeIcon icon={socialIcon(row.original.social)} className='size-3.5' />
              {formatSocial(row.original.social)}
            </span>
          </Badge>
        ),
      },
      {
        id: 'status',
        accessorFn: (row) => row.status ?? 'invalid',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Status' />,
        cell: ({ row }) => (
          <Badge className={statusBadgeClass(row.original.status)}>
            {row.original.status === 'ok' ? <CheckCircle2 className='mr-1 size-3' /> : <CircleX className='mr-1 size-3' />}
            {row.original.status === 'ok' ? 'OK' : 'Invalid'}
          </Badge>
        ),
      },
      {
        id: ACTIONS_COLUMN_ID,
        enableSorting: false,
        enableHiding: false,
        cell: ({ row }) => {
          const auth = row.original
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
                  <DropdownMenuItem className='gap-2' onSelect={() => openAuthSheet(auth)}>
                    <Pencil className='size-4' />
                    Edit
                  </DropdownMenuItem>
                  {auth.social === 'linkedin' ? (
                    <DropdownMenuItem className='gap-2' onSelect={() => void runLinkedinLogin(auth)}>
                      <Play className='size-4' />
                      Run LinkedIn login
                    </DropdownMenuItem>
                  ) : null}
                  <DropdownMenuItem className='gap-2' onSelect={() => void duplicateAuth(auth)}>
                    <Copy className='size-4' />
                    Duplicate
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2 text-destructive' onSelect={() => requestDeleteAuth(auth)}>
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
    [duplicateAuth, openAuthSheet, requestDeleteAuth, runLinkedinLogin]
  )

  const table = useReactTable({
    data: auths,
    columns,
    state: {
      sorting,
      columnFilters,
      globalFilter,
      columnVisibility,
      rowSelection,
    },
    enableRowSelection: true,
    enableGlobalFilter: true,
    globalFilterFn: (row, _columnId, filterValue) => {
      const query = String(filterValue ?? '').trim().toLowerCase()
      if (!query) return true
      const item = row.original
      return (
        item.id.toLowerCase().includes(query) ||
        authUsername(item).toLowerCase().includes(query) ||
        item.social.toLowerCase().includes(query) ||
        (item.status ?? 'invalid').toLowerCase().includes(query)
      )
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    getRowId: (row) => row.id,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: {
      pagination: { pageSize: 10 },
    },
  })

  const renderAuthFormFields = (prefix: string) => (
    <div className='space-y-4'>
      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-social`}>Social</Label>
        <Select
          value={form.social}
          onValueChange={(value) => setForm((prev) => ({ ...prev, social: value as SocialPlatform }))}
        >
          <SelectTrigger id={`${prefix}-social`}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value='reddit'>
              <span className='inline-flex items-center gap-2'>
                <FontAwesomeIcon icon={socialIcon('reddit')} className='size-3.5' />
                Reddit
              </span>
            </SelectItem>
            <SelectItem value='linkedin'>
              <span className='inline-flex items-center gap-2'>
                <FontAwesomeIcon icon={socialIcon('linkedin')} className='size-3.5' />
                LinkedIn
              </span>
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      {form.social === 'reddit' ? (
        <div className='space-y-3'>
          <p className='text-sm font-medium'>Reddit credentials</p>
          <div className='grid gap-3 md:grid-cols-2'>
            <div className='space-y-2'>
              <Label htmlFor={`${prefix}-reddit-client-id`}>Client ID</Label>
              <Input
                id={`${prefix}-reddit-client-id`}
                value={form.reddit_client_id}
                onChange={(event) => setForm((prev) => ({ ...prev, reddit_client_id: event.target.value }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor={`${prefix}-reddit-client-secret`}>Client secret</Label>
              <Input
                id={`${prefix}-reddit-client-secret`}
                type='password'
                value={form.reddit_client_secret}
                onChange={(event) => setForm((prev) => ({ ...prev, reddit_client_secret: event.target.value }))}
              />
            </div>
            <div className='space-y-2 md:col-span-2'>
              <Label htmlFor={`${prefix}-reddit-user-agent`}>User agent</Label>
              <Input
                id={`${prefix}-reddit-user-agent`}
                value={form.reddit_user_agent}
                onChange={(event) => setForm((prev) => ({ ...prev, reddit_user_agent: event.target.value }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor={`${prefix}-reddit-username`}>Username</Label>
              <Input
                id={`${prefix}-reddit-username`}
                value={form.reddit_username}
                onChange={(event) => setForm((prev) => ({ ...prev, reddit_username: event.target.value }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor={`${prefix}-reddit-password`}>Password</Label>
              <Input
                id={`${prefix}-reddit-password`}
                type='password'
                value={form.reddit_password}
                onChange={(event) => setForm((prev) => ({ ...prev, reddit_password: event.target.value }))}
              />
            </div>
            <div className='md:col-span-2'>
              <p className='text-xs text-muted-foreground'>
                Client ID, client secret, username, and password are required for Reddit validation.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className='space-y-3'>
          <p className='text-sm font-medium'>LinkedIn credentials</p>
          <div className='grid gap-3'>
            <div className='md:col-span-2'>
              <p className='text-xs text-muted-foreground'>
                No username/password required. On create, a browser opens so you can save a LinkedIn session.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )

  return (
    <>
      <Header fixed>
        <div className='ms-auto flex items-center gap-2'>
          <Button size='sm' variant='outline' onClick={() => void loadAuths()} disabled={loading || saving}>
            <RefreshCw className='size-4' />
            Refresh
          </Button>
          <Button size='sm' onClick={openCreateDialog}>
            <Plus className='size-4' />
            Create auth
          </Button>
        </div>
      </Header>

      <Main className='flex flex-1 flex-col gap-4 sm:gap-6'>
        <div>
          <h2 className='text-2xl font-bold tracking-tight'>Auth</h2>
        </div>

        <DataTableToolbar
          table={table}
          searchPlaceholder='Search by auth id, username, social or status...'
        />

        <div className='space-y-4'>
          <div className='rounded-md border'>
            <Table>
              <TableHeader>
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => (
                      <TableHead key={header.id}>
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
                      onClick={() => openAuthSheet(row.original)}
                      className='cursor-pointer'
                      data-state={row.getIsSelected() && 'selected'}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <TableCell
                          key={cell.id}
                          className={cn(
                            cell.column.id === ACTIONS_COLUMN_ID ? 'w-[60px]' : undefined,
                            cell.column.id === 'id' ? 'w-[96px]' : undefined,
                          )}
                        >
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={columns.length} className='h-24 text-center text-muted-foreground'>
                      No auth objects found.
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
        selectedCount={table.getSelectedRowModel().rows.length}
        entityLabel='auth'
        onClear={() => setRowSelection({})}
        onDelete={() => requestDeleteSelectedAuths()}
        deleteLabel='Delete selected'
        deleteDisabled={busyKey === 'bulk-delete'}
      />

      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className='sm:max-w-2xl'>
          <ModalDialogHeader>
            <ModalDialogTitle>Create auth</ModalDialogTitle>
            <DialogDescription>Create an auth object for Reddit or LinkedIn.</DialogDescription>
          </ModalDialogHeader>
          {renderAuthFormFields('create-auth')}
          <DialogFooter>
            <Button variant='outline' onClick={() => setCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={() => void saveFromSheet()} disabled={saving}>
              {saving ? 'Saving...' : 'Create auth'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
        <SheetContent side='right' className='sm:max-w-xl p-0'>
          <div className='flex h-full flex-col'>
            <SheetHeader className='px-4 pt-4'>
              <SheetTitle>{selectedAuth ? `Auth · ${authLabel(selectedAuth)}` : 'Auth details'}</SheetTitle>
              <SheetDescription>Manage this auth object and linked credentials.</SheetDescription>
            </SheetHeader>

            <div className='sticky top-0 z-10 mt-3 flex items-center gap-2 border-y bg-background px-4 py-3'>
              {selectedAuth?.social === 'linkedin' ? (
                <Button
                  size='sm'
                  variant='outline'
                  onClick={() => selectedAuth && void runLinkedinLogin(selectedAuth)}
                  disabled={busyKey === `${selectedAuth.id}:login`}
                >
                  <Play className='size-4' />
                  Run LinkedIn login
                </Button>
              ) : null}
              {selectedAuth ? (
                <Button
                  size='sm'
                  variant='outline'
                  onClick={() => requestDeleteAuth(selectedAuth)}
                  disabled={busyKey === `${selectedAuth.id}:delete`}
                >
                  <Trash2 className='size-4' />
                  Delete
                </Button>
              ) : null}
              <Button className='ms-auto' size='sm' onClick={() => void saveFromSheet()} disabled={saving}>
                {saving ? 'Saving...' : 'Save changes'}
              </Button>
            </div>

            <div className='flex-1 space-y-4 overflow-y-auto px-4 py-4'>
              {renderAuthFormFields('edit-auth')}
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <ConfirmDialog
        open={Boolean(pendingDeleteAuth)}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteAuth(null)
        }}
        title='Delete auth'
        desc={
          pendingDeleteAuth
            ? pendingDeleteAuth.relatedJobs.length > 0
              ? (
                  <>
                    <p>Delete {authLabel(pendingDeleteAuth.auth)} and related jobs?</p>
                    <p className='mt-2 text-xs text-muted-foreground'>
                      Related jobs: {pendingDeleteAuth.relatedJobs.map((item) => item.name).join(', ')}
                    </p>
                  </>
                )
              : `Delete ${authLabel(pendingDeleteAuth.auth)}? This cannot be undone.`
            : ''
        }
        confirmText='Delete'
        destructive
        isLoading={pendingDeleteAuth ? busyKey === `${pendingDeleteAuth.auth.id}:delete` : false}
        handleConfirm={() => void confirmDeleteAuth()}
      />

      <ConfirmDialog
        open={Boolean(pendingDeleteAuths)}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteAuths(null)
        }}
        title='Delete selected auths'
        desc={
          pendingDeleteAuths
            ? pendingDeleteAuths.relatedJobs.length > 0
              ? (
                  <>
                    <p>
                      This will delete {pendingDeleteAuths.auths.length} selected auth objects and related jobs.
                    </p>
                    <p className='mt-2 text-xs text-muted-foreground'>
                      Related jobs: {pendingDeleteAuths.relatedJobs.map((item) => item.name).join(', ')}
                    </p>
                  </>
                )
              : `This action will permanently delete ${pendingDeleteAuths.auths.length} selected auth object${pendingDeleteAuths.auths.length === 1 ? '' : 's'}.`
            : ''
        }
        confirmText='Delete'
        destructive
        isLoading={busyKey === 'bulk-delete'}
        handleConfirm={() => void confirmDeleteSelectedAuths()}
      />
    </>
  )
}
