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
  type SortingState,
  useReactTable,
  type VisibilityState,
} from '@tanstack/react-table'
import { Copy, MoreHorizontal, Pencil, RefreshCw, Trash2 } from 'lucide-react'
import { ConfirmDialog } from '@/components/confirm-dialog'
import { DataTableColumnHeader, DataTablePagination, DataTableToolbar } from '@/components/data-table'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
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
import { reddropApi, type PersonaProfile } from '@/lib/reddrop-api'

const ACTIONS_COLUMN_ID = 'actions'

function buildDefaultPersona(): PersonaProfile {
  return {
    name: '',
    description: '',
    objective: '',
  }
}

function normalizePersona(payload: PersonaProfile): PersonaProfile {
  return {
    name: payload.name.trim(),
    description: payload.description.trim(),
    objective: payload.objective.trim(),
  }
}

function buildDuplicatePersonaName(baseName: string, existingNames: Set<string>): string {
  const normalized = baseName.trim().replace(/\s+/g, '-') || 'persona'
  let attempt = `${normalized}-copy`
  let index = 2
  while (existingNames.has(attempt.toLowerCase())) {
    attempt = `${normalized}-copy-${index}`
    index += 1
  }
  return attempt
}

export function PersonasPage() {
  const [personas, setPersonas] = useState<PersonaProfile[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const [sorting, setSorting] = useState<SortingState>([{ id: 'name', desc: false }])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [globalFilter, setGlobalFilter] = useState('')
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>([])
  const [draggingColumnId, setDraggingColumnId] = useState<string | null>(null)

  const [sheetOpen, setSheetOpen] = useState(false)
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [pendingDeletePersona, setPendingDeletePersona] = useState<PersonaProfile | null>(null)
  const [selectedPersona, setSelectedPersona] = useState<PersonaProfile | null>(null)
  const [form, setForm] = useState<PersonaProfile>(buildDefaultPersona)

  const loadPersonas = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const payload = await reddropApi.listPersonas()
      setPersonas(payload.personas)
      setSelectedPersona((prev) => {
        if (!prev) return null
        return payload.personas.find((item) => item.name === prev.name) ?? null
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load personas'
      toast.error(message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadPersonas()
  }, [loadPersonas])

  const persistPersonas = useCallback(async (next: PersonaProfile[], successMessage: string): Promise<void> => {
    setSaving(true)
    try {
      const payload = await reddropApi.savePersonas({ personas: next })
      setPersonas(payload.personas)
      toast.success(successMessage)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save personas'
      toast.error(message)
    } finally {
      setSaving(false)
      setBusyKey(null)
    }
  }, [])

  const openCreateDialog = useCallback((): void => {
    setSelectedPersona(null)
    setForm(buildDefaultPersona())
    setSheetOpen(false)
    setCreateDialogOpen(true)
  }, [])

  const openPersonaSheet = useCallback((persona: PersonaProfile): void => {
    setSelectedPersona(persona)
    setForm(persona)
    setSheetOpen(true)
  }, [])

  const saveFromSheet = useCallback(async (): Promise<void> => {
    const normalized = normalizePersona(form)
    if (!normalized.name || !normalized.description || !normalized.objective) {
      toast.error('Name, description, and objective are required.')
      return
    }

    const editingName = selectedPersona?.name.trim().toLowerCase() ?? null
    const targetName = normalized.name.toLowerCase()
    const duplicate = personas.some((item) => {
      const current = item.name.trim().toLowerCase()
      if (editingName !== null && current === editingName) {
        return false
      }
      return current === targetName
    })

    if (duplicate) {
      toast.error(`Persona already exists: ${normalized.name}`)
      return
    }

    const next = selectedPersona === null
      ? [...personas, normalized]
      : personas.map((item) => (item.name === selectedPersona.name ? normalized : item))

    await persistPersonas(
      next,
      selectedPersona === null
        ? `Created persona ${normalized.name}`
        : `Updated persona ${normalized.name}`
    )
    setSelectedPersona(selectedPersona === null ? null : normalized)
    setSheetOpen(false)
    setCreateDialogOpen(false)
  }, [form, personas, persistPersonas, selectedPersona])

  const deletePersona = useCallback(async (persona: PersonaProfile): Promise<void> => {
    const key = `${persona.name}:delete`
    setBusyKey(key)
    const next = personas.filter((item) => item.name !== persona.name)
    await persistPersonas(next, `Deleted persona ${persona.name}`)
    if (selectedPersona?.name === persona.name) {
      setSelectedPersona(null)
      setSheetOpen(false)
    }
  }, [personas, persistPersonas, selectedPersona?.name])

  const duplicatePersona = useCallback(async (persona: PersonaProfile): Promise<void> => {
    const key = `${persona.name}:duplicate`
    setBusyKey(key)
    const existingNames = new Set(personas.map((item) => item.name.trim().toLowerCase()))
    const duplicateName = buildDuplicatePersonaName(persona.name, existingNames)
    const next = [
      ...personas,
      {
        ...persona,
        name: duplicateName,
      },
    ]
    await persistPersonas(next, `Duplicated persona ${persona.name} as ${duplicateName}`)
  }, [personas, persistPersonas])

  const requestDeletePersona = useCallback((persona: PersonaProfile): void => {
    setPendingDeletePersona(persona)
  }, [])

  const confirmDeletePersona = useCallback(async (): Promise<void> => {
    if (!pendingDeletePersona) return
    await deletePersona(pendingDeletePersona)
    setPendingDeletePersona(null)
  }, [deletePersona, pendingDeletePersona])

  const columns = useMemo<ColumnDef<PersonaProfile>[]>(
    () => [
      {
        accessorKey: 'name',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Name' />,
        enableHiding: false,
        cell: ({ row }) => <span className='font-medium'>{row.original.name}</span>,
      },
      {
        accessorKey: 'description',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Description' />,
        cell: ({ row }) => (
          <span className='block max-w-[30rem] truncate' title={row.original.description}>
            {row.original.description}
          </span>
        ),
      },
      {
        accessorKey: 'objective',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Objective' />,
        cell: ({ row }) => (
          <span className='block max-w-[30rem] truncate' title={row.original.objective}>
            {row.original.objective}
          </span>
        ),
      },
      {
        id: ACTIONS_COLUMN_ID,
        enableSorting: false,
        enableHiding: false,
        cell: ({ row }) => {
          const persona = row.original
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
                  <DropdownMenuItem className='gap-2' onSelect={() => openPersonaSheet(persona)}>
                    <Pencil className='size-4' />
                    Edit
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2' onSelect={() => void duplicatePersona(persona)}>
                    <Copy className='size-4' />
                    Duplicate
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2 text-destructive' onSelect={() => requestDeletePersona(persona)}>
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
    [duplicatePersona, openPersonaSheet, requestDeletePersona]
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
    data: personas,
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
      const persona = row.original
      return (
        persona.name.toLowerCase().includes(query) ||
        persona.description.toLowerCase().includes(query) ||
        persona.objective.toLowerCase().includes(query)
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

  const selectedDeleteKey = selectedPersona ? `${selectedPersona.name}:delete` : ''
  const renderPersonaFormFields = (prefix: string) => (
    <>
      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-persona-name`}>Name</Label>
        <Input
          id={`${prefix}-persona-name`}
          value={form.name}
          onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
          placeholder='marco'
          required
        />
      </div>

      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-persona-description`}>Description</Label>
        <Textarea
          id={`${prefix}-persona-description`}
          value={form.description}
          onChange={(event) => setForm((prev) => ({ ...prev, description: event.target.value }))}
          rows={6}
          required
        />
      </div>

      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-persona-objective`}>Objective</Label>
        <Textarea
          id={`${prefix}-persona-objective`}
          value={form.objective}
          onChange={(event) => setForm((prev) => ({ ...prev, objective: event.target.value }))}
          rows={6}
          required
        />
      </div>
    </>
  )

  return (
    <>
      <Header fixed>
        <div className='ms-auto flex items-center gap-2'>
          <Button size='sm' onClick={openCreateDialog}>
            Create persona
          </Button>
          <Button variant='outline' size='sm' onClick={() => void loadPersonas()} disabled={loading}>
            <RefreshCw className='size-4' />
            Refresh
          </Button>
        </div>
      </Header>

      <Main className='flex flex-1 flex-col gap-4 sm:gap-6'>
        <div>
          <h2 className='text-2xl font-bold tracking-tight'>Personas</h2>
        </div>

        <DataTableToolbar
          table={table}
          searchPlaceholder='Search by name, description or objective...'
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
                      onClick={() => openPersonaSheet(row.original)}
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
                      No personas configured.
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
            <ModalDialogTitle>Create persona</ModalDialogTitle>
            <DialogDescription>
              Persona profile used for reply generation.
            </DialogDescription>
          </ModalDialogHeader>

          <div className='max-h-[70vh] space-y-4 overflow-y-auto pr-1'>
            {renderPersonaFormFields('dialog')}
          </div>

          <DialogFooter>
            <Button
              type='button'
              variant='outline'
              onClick={() => setCreateDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button type='button' onClick={() => void saveFromSheet()} disabled={saving}>
              {saving ? 'Saving...' : 'Create persona'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={pendingDeletePersona !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDeletePersona(null)
        }}
        title='Delete persona'
        desc={
          pendingDeletePersona
            ? `This action will permanently delete "${pendingDeletePersona.name}".`
            : 'This action will permanently delete this persona.'
        }
        confirmText='Delete'
        destructive
        isLoading={pendingDeletePersona !== null && busyKey === `${pendingDeletePersona.name}:delete`}
        handleConfirm={() => void confirmDeletePersona()}
      />

      <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
        <SheetContent side='right' className='sm:max-w-xl p-0'>
          <div className='flex h-full flex-col'>
            <SheetHeader className='px-4 pt-4'>
              <SheetTitle>{selectedPersona ? `Persona · ${selectedPersona.name}` : 'Persona details'}</SheetTitle>
              <SheetDescription>
                Persona profile used for reply generation.
              </SheetDescription>
            </SheetHeader>

            <div className='sticky top-0 z-10 mt-3 flex items-center gap-2 border-y bg-background px-4 py-3'>
              {selectedPersona ? (
                <Button
                  size='sm'
                  variant='outline'
                  onClick={() => requestDeletePersona(selectedPersona)}
                  disabled={busyKey === selectedDeleteKey}
                >
                  <Trash2 className='size-4' />
                  Delete
                </Button>
              ) : null}
              <Button className='ms-auto' size='sm' onClick={() => void saveFromSheet()} disabled={saving}>
                {saving ? 'Saving...' : selectedPersona ? 'Save changes' : 'Create persona'}
              </Button>
            </div>

            <div className='flex-1 space-y-4 overflow-y-auto px-4 py-4'>
              {renderPersonaFormFields('sheet')}
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
