import { useEffect, useState } from 'react'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { reddropApi, type SettingsPayload } from '@/lib/reddrop-api'
import { toast } from 'sonner'

const initialState: SettingsPayload = {
  reddit: {
    client_id: '',
    client_secret: '',
    user_agent: '',
    username: '',
    password: '',
  },
  openrouter: {
    api_key: '',
    base_url: '',
    model: '',
    http_referer: '',
    x_title: '',
    timeout_seconds: '20',
  },
}

export function SettingsPage() {
  const [settings, setSettings] = useState<SettingsPayload>(initialState)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

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

  return (
    <>
      <Header fixed>
        <div className='ms-auto flex items-center gap-2'>
          <Button onClick={() => void save()} disabled={loading || saving}>
            Save settings
          </Button>
        </div>
      </Header>

      <Main className='space-y-6'>
        <section className='space-y-3'>
          <h2 className='text-xl font-semibold'>Reddit configuration</h2>
          <div className='grid gap-3 md:grid-cols-2'>
            <div className='space-y-2'>
              <Label htmlFor='reddit-client-id'>Client ID</Label>
              <Input
                id='reddit-client-id'
                value={settings.reddit.client_id}
                onChange={(event) => setSettings((prev) => ({ ...prev, reddit: { ...prev.reddit, client_id: event.target.value } }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor='reddit-client-secret'>Client Secret</Label>
              <Input
                id='reddit-client-secret'
                value={settings.reddit.client_secret}
                onChange={(event) => setSettings((prev) => ({ ...prev, reddit: { ...prev.reddit, client_secret: event.target.value } }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor='reddit-user-agent'>User Agent</Label>
              <Input
                id='reddit-user-agent'
                value={settings.reddit.user_agent}
                onChange={(event) => setSettings((prev) => ({ ...prev, reddit: { ...prev.reddit, user_agent: event.target.value } }))}
              />
            </div>
            <div className='space-y-2'>
              <Label htmlFor='reddit-username'>Username</Label>
              <Input
                id='reddit-username'
                value={settings.reddit.username}
                onChange={(event) => setSettings((prev) => ({ ...prev, reddit: { ...prev.reddit, username: event.target.value } }))}
              />
            </div>
            <div className='space-y-2 md:col-span-2'>
              <Label htmlFor='reddit-password'>Password</Label>
              <Input
                id='reddit-password'
                type='password'
                value={settings.reddit.password}
                onChange={(event) => setSettings((prev) => ({ ...prev, reddit: { ...prev.reddit, password: event.target.value } }))}
              />
            </div>
          </div>
        </section>

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
