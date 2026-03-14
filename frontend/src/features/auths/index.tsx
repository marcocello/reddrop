import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  type ColumnDef,
  type ColumnFiltersState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
  type VisibilityState,
} from '@tanstack/react-table'
import { Copy, MoreHorizontal, Pencil, Play, Trash2 } from 'lucide-react'
import { ConfirmDialog } from '@/components/confirm-dialog'
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
import { reddropApi, type AuthProfile, type SocialPlatform } from '@/lib/reddrop-api'

const ACTIONS_COLUMN_ID = 'actions'

function buildDefaultAuth(): AuthProfile {
  return {
    id: '',
    name: '',
    social: 'reddit',
    reddit_client_id: '',
    reddit_client_secret: '',
    reddit_user_agent: 'Reddrop:v1.0',
    reddit_username: '',
    reddit_password: '',
    linkedin_email: '',
    linkedin_password: '',
    linkedin_headless: 'true',
    linkedin_user_data_dir: '~/.linkedin-playwright',
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

function normalizeToken(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function buildLinkedinProfileDir(name: string, authId: string): string {
  const namePart = normalizeToken(name).slice(0, 24) || 'linkedin'
  const idPart = authId.slice(0, 8) || 'auth'
  return `~/.reddrop/linkedin/${namePart}-${idPart}`
}

function normalizeAuth(payload: AuthProfile): AuthProfile {
  const name = payload.name.trim()
  const id = payload.id.trim() || createAuthId()
  const social: SocialPlatform = payload.social === 'linkedin' ? 'linkedin' : 'reddit'
  return {
    ...payload,
    id,
    name,
    social,
    reddit_client_id: payload.reddit_client_id.trim(),
    reddit_client_secret: payload.reddit_client_secret.trim(),
    reddit_user_agent: payload.reddit_user_agent.trim() || 'Reddrop:v1.0',
    reddit_username: payload.reddit_username.trim(),
    reddit_password: payload.reddit_password.trim(),
    linkedin_email: payload.linkedin_email.trim(),
    linkedin_password: payload.linkedin_password.trim(),
    linkedin_headless: payload.linkedin_headless.trim() || 'true',
    linkedin_user_data_dir: social === 'linkedin'
      ? buildLinkedinProfileDir(name, id)
      : payload.linkedin_user_data_dir.trim() || '~/.linkedin-playwright',
    linkedin_wait_seconds: payload.linkedin_wait_seconds.trim() || '120',
  }
}

function socialBadgeClass(value: SocialPlatform): string {
  return value === 'linkedin'
    ? 'bg-sky-600 text-white'
    : 'bg-orange-600 text-white'
}

function summaryText(item: AuthProfile): string {
  if (item.social === 'linkedin') {
    return item.linkedin_user_data_dir || 'Browser session profile'
  }
  return item.reddit_username || 'No Reddit username'
}

export function AuthsPage() {
  const [auths, setAuths] = useState<AuthProfile[]>([])
  const [saving, setSaving] = useState(false)
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const [sorting, setSorting] = useState<SortingState>([{ id: 'name', desc: false }])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [globalFilter, setGlobalFilter] = useState('')
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})

  const [sheetOpen, setSheetOpen] = useState(false)
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [pendingDeleteAuth, setPendingDeleteAuth] = useState<AuthProfile | null>(null)
  const [selectedAuth, setSelectedAuth] = useState<AuthProfile | null>(null)
  const [form, setForm] = useState<AuthProfile>(buildDefaultAuth)

  const loadAuths = useCallback(async (): Promise<void> => {
    try {
      const payload = await reddropApi.listAuths()
      setAuths(payload.auths)
      setSelectedAuth((prev) => {
        if (!prev) return null
        return payload.auths.find((item) => item.id === prev.id) ?? null
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load auths'
      toast.error(message)
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
    if (!normalized.name) {
      toast.error('Name is required.')
      return
    }
    if (!normalized.id) {
      toast.error('Name must include letters or numbers.')
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
        toast.success(`Created auth ${normalized.name}`)
        toast.success(`Run LinkedIn login for ${normalized.name}`)
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
        ? `Created auth ${normalized.name}`
        : `Updated auth ${normalized.name}`
    )
    setSelectedAuth(selectedAuth === null ? null : normalized)
    setSheetOpen(false)
    setCreateDialogOpen(false)
  }, [auths, form, persistAuths, selectedAuth])

  const deleteAuth = useCallback(async (auth: AuthProfile): Promise<void> => {
    const key = `${auth.id}:delete`
    setBusyKey(key)
    const next = auths.filter((item) => item.id !== auth.id)
    await persistAuths(next, `Deleted auth ${auth.name}`)
    if (selectedAuth?.id === auth.id) {
      setSelectedAuth(null)
      setSheetOpen(false)
    }
  }, [auths, persistAuths, selectedAuth?.id])

  const duplicateAuth = useCallback(async (auth: AuthProfile): Promise<void> => {
    const key = `${auth.id}:duplicate`
    setBusyKey(key)
    const duplicateId = createAuthId()
    const normalized = normalizeAuth({
      ...auth,
      id: duplicateId,
      name: `${auth.name} Copy`,
    })
    const next = [
      ...auths,
      normalized,
    ]
    await persistAuths(next, `Duplicated auth ${auth.name} as ${normalized.name}`)
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
      toast.success(`Run LinkedIn login for ${auth.name}`)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to run LinkedIn login'
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [])

  const requestDeleteAuth = useCallback((auth: AuthProfile): void => {
    setPendingDeleteAuth(auth)
  }, [])

  const confirmDeleteAuth = useCallback(async (): Promise<void> => {
    if (!pendingDeleteAuth) return
    await deleteAuth(pendingDeleteAuth)
    setPendingDeleteAuth(null)
  }, [deleteAuth, pendingDeleteAuth])

  const columns = useMemo<ColumnDef<AuthProfile>[]>(
    () => [
      {
        accessorKey: 'name',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Name' />,
        enableHiding: false,
        cell: ({ row }) => <span className='font-medium'>{row.original.name}</span>,
      },
      {
        accessorKey: 'social',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Social' />,
        cell: ({ row }) => (
          <Badge className={socialBadgeClass(row.original.social)}>
            {row.original.social}
          </Badge>
        ),
      },
      {
        id: 'summary',
        accessorFn: (row) => summaryText(row),
        header: ({ column }) => <DataTableColumnHeader column={column} title='Summary' />,
        cell: ({ row }) => (
          <span className='block max-w-[28rem] truncate' title={summaryText(row.original)}>
            {summaryText(row.original)}
          </span>
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
    },
    enableGlobalFilter: true,
    globalFilterFn: (row, _columnId, filterValue) => {
      const query = String(filterValue ?? '').trim().toLowerCase()
      if (!query) return true
      const item = row.original
      return (
        item.name.toLowerCase().includes(query) ||
        item.social.toLowerCase().includes(query) ||
        summaryText(item).toLowerCase().includes(query)
      )
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    onColumnVisibilityChange: setColumnVisibility,
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
        <Label htmlFor={`${prefix}-name`}>Name</Label>
        <Input
          id={`${prefix}-name`}
          value={form.name}
          onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
          placeholder='Reddit Main'
          required
        />
      </div>

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
            <SelectItem value='reddit'>reddit</SelectItem>
            <SelectItem value='linkedin'>linkedin</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {form.social === 'reddit' ? (
        <div className='space-y-3'>
          <p className='text-sm font-medium'>Reddit credentials</p>
          <div className='grid gap-3 md:grid-cols-2'>
            <div className='space-y-2'>
              <Label htmlFor={`${prefix}-reddit-username`}>Username</Label>
              <Input
                id={`${prefix}-reddit-username`}
                value={form.reddit_username}
                onChange={(event) => setForm((prev) => ({ ...prev, reddit_username: event.target.value }))}
              />
            </div>
            <div className='space-y-2 md:col-span-2'>
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
                Reddit app credentials can stay in environment variables; this auth only requires username/password.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className='space-y-3'>
          <p className='text-sm font-medium'>LinkedIn credentials</p>
          <div className='grid gap-3 md:grid-cols-2'>
            <div className='space-y-2'>
              <Label htmlFor={`${prefix}-linkedin-headless`}>Headless</Label>
              <Input
                id={`${prefix}-linkedin-headless`}
                value={form.linkedin_headless}
                onChange={(event) => setForm((prev) => ({ ...prev, linkedin_headless: event.target.value }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor={`${prefix}-linkedin-wait-seconds`}>Wait seconds</Label>
              <Input
                id={`${prefix}-linkedin-wait-seconds`}
                value={form.linkedin_wait_seconds}
                onChange={(event) => setForm((prev) => ({ ...prev, linkedin_wait_seconds: event.target.value }))}
              />
            </div>
            <div className='space-y-2 md:col-span-2'>
              <Label>User data dir</Label>
              <p className='rounded-md border bg-muted/30 px-3 py-2 font-mono text-xs'>
                {buildLinkedinProfileDir(form.name, form.id || 'generated-id')}
              </p>
            </div>
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
          <Button size='sm' onClick={openCreateDialog}>
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
          searchPlaceholder='Search by auth id, name, social...'
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
                        <TableCell key={cell.id} className={cn(cell.column.id === ACTIONS_COLUMN_ID ? 'w-[60px]' : undefined)}>
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

      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className='sm:max-w-2xl'>
          <ModalDialogHeader>
            <ModalDialogTitle>Create auth</ModalDialogTitle>
            <DialogDescription>Create an auth object for reddit or linkedin.</DialogDescription>
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
              <SheetTitle>{selectedAuth ? `Auth · ${selectedAuth.name}` : 'Auth details'}</SheetTitle>
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
        desc={pendingDeleteAuth ? `Delete ${pendingDeleteAuth.name}? This cannot be undone.` : ''}
        confirmText='Delete'
        destructive
        isLoading={pendingDeleteAuth ? busyKey === `${pendingDeleteAuth.id}:delete` : false}
        handleConfirm={() => void confirmDeleteAuth()}
      />
    </>
  )
}
