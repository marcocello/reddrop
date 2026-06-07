import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { reddropApi, type SettingsPayload } from '@/lib/reddrop-api'
import { toast } from 'sonner'

const initialState: SettingsPayload = {
  openrouter: {
    api_key: '',
    base_url: '',
    model: '',
    http_referer: '',
    x_title: '',
    timeout_seconds: '20',
  },
}

function toRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null) return null
  return value as Record<string, unknown>
}

function toStringValue(value: unknown, fallback: string): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number') return String(value)
  return fallback
}

function parseImportedSettings(content: string): SettingsPayload {
  const parsed = JSON.parse(content) as unknown
  const root = toRecord(parsed)
  if (!root) {
    throw new Error('Invalid settings file format')
  }
  const openrouter = toRecord(root.openrouter)
  if (!openrouter) {
    throw new Error('Settings file must include openrouter section')
  }
  return {
    openrouter: {
      api_key: toStringValue(openrouter.api_key, ''),
      base_url: toStringValue(openrouter.base_url, ''),
      model: toStringValue(openrouter.model, ''),
      http_referer: toStringValue(openrouter.http_referer, ''),
      x_title: toStringValue(openrouter.x_title, ''),
      timeout_seconds: toStringValue(openrouter.timeout_seconds, '20'),
    },
  }
}

export function SettingsPage() {
  const [settings, setSettings] = useState<SettingsPayload>(initialState)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [importing, setImporting] = useState(false)
  const importInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const payload = await reddropApi.listSettings()
        setSettings(payload)
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to load settings'
        toast.error(message)
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [])

  const save = async () => {
    setSaving(true)
    try {
      const payload = await reddropApi.saveSettings(settings)
      setSettings(payload)
      toast.success('Settings saved')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save settings'
      toast.error(message)
    } finally {
      setSaving(false)
    }
  }

  const openImportDialog = () => {
    importInputRef.current?.click()
  }

  const importSettings = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setImporting(true)
    try {
      const content = await file.text()
      const imported = parseImportedSettings(content)
      setSettings(imported)
      toast.success('Settings imported. Save settings to persist.')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to import settings'
      toast.error(message)
    } finally {
      event.target.value = ''
      setImporting(false)
    }
  }

  return (
    <>
      <Header fixed>
        <div className='ms-auto flex items-center gap-2'>
          <input
            ref={importInputRef}
            type='file'
            accept='application/json,.json'
            className='hidden'
            onChange={(event) => void importSettings(event)}
          />
          <Button variant='outline' onClick={openImportDialog} disabled={loading || saving || importing}>
            {importing ? 'Importing...' : 'Import settings'}
          </Button>
          <Button onClick={() => void save()} disabled={loading || saving || importing}>
            Save settings
          </Button>
        </div>
      </Header>

      <Main className='space-y-6'>
        <div>
          <h2 className='text-2xl font-bold tracking-tight'>Settings</h2>
        </div>

        <section className='space-y-3'>
          <h2 className='text-xl font-semibold'>OpenRouter configuration</h2>
          <div className='grid gap-3 md:grid-cols-2'>
            <div className='space-y-2'>
              <Label htmlFor='openrouter-api-key'>API Key</Label>
              <Input
                id='openrouter-api-key'
                value={settings.openrouter.api_key}
                onChange={(event) => setSettings((prev) => ({ ...prev, openrouter: { ...prev.openrouter, api_key: event.target.value } }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor='openrouter-base-url'>Base URL</Label>
              <Input
                id='openrouter-base-url'
                value={settings.openrouter.base_url}
                onChange={(event) => setSettings((prev) => ({ ...prev, openrouter: { ...prev.openrouter, base_url: event.target.value } }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor='openrouter-model'>Model</Label>
              <Input
                id='openrouter-model'
                value={settings.openrouter.model}
                onChange={(event) => setSettings((prev) => ({ ...prev, openrouter: { ...prev.openrouter, model: event.target.value } }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor='openrouter-http-referer'>HTTP Referer</Label>
              <Input
                id='openrouter-http-referer'
                value={settings.openrouter.http_referer}
                onChange={(event) => setSettings((prev) => ({ ...prev, openrouter: { ...prev.openrouter, http_referer: event.target.value } }))}
              />
            </div>
            <div className='space-y-2 md:col-span-2'>
              <Label htmlFor='openrouter-x-title'>X-Title</Label>
              <Input
                id='openrouter-x-title'
                value={settings.openrouter.x_title}
                onChange={(event) => setSettings((prev) => ({ ...prev, openrouter: { ...prev.openrouter, x_title: event.target.value } }))}
              />
            </div>
            <div className='space-y-2 md:col-span-2'>
              <Label htmlFor='openrouter-timeout-seconds'>Timeout (seconds)</Label>
              <Input
                id='openrouter-timeout-seconds'
                value={settings.openrouter.timeout_seconds}
                onChange={(event) =>
                  setSettings((prev) => ({
                    ...prev,
                    openrouter: { ...prev.openrouter, timeout_seconds: event.target.value },
                  }))
                }
              />
            </div>
          </div>
        </section>

      </Main>
    </>
  )
}
