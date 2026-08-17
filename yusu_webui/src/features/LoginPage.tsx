import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { ZapIcon } from 'lucide-react'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import { getHealth } from '@/api/yusu'
import { useSettingsStore } from '@/stores/settings'
import { SiteInfo } from '@/lib/constants'
import ThemeProvider from '@/components/ThemeProvider'

export default function LoginPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const baseUrl = useSettingsStore.use.baseUrl()
  const apiKey = useSettingsStore.use.apiKey()
  const setBaseUrl = useSettingsStore.use.setBaseUrl()
  const setApiKey = useSettingsStore.use.setApiKey()

  const [url, setUrl] = useState(baseUrl || 'http://127.0.0.1:8920')
  const [key, setKey] = useState(apiKey ?? '')
  const [testing, setTesting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [connectedInfo, setConnectedInfo] = useState<string | null>(null)

  const handleConnect = async () => {
    setTesting(true)
    setError(null)
    setConnectedInfo(null)
    try {
      const health = await getHealth(url, key || null)
      setBaseUrl(url)
      setApiKey(key || null)
      setConnectedInfo(`${health.service} v${health.version}`)
      setTimeout(() => navigate('/'), 400)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setTesting(false)
    }
  }

  return (
    <ThemeProvider>
      <div className="flex min-h-screen items-center justify-center bg-background p-4">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <div className="mb-2 flex items-center justify-center gap-2">
              <ZapIcon className="size-6 text-emerald-400" aria-hidden="true" />
              <span className="text-xl font-bold">{SiteInfo.name}</span>
            </div>
            <CardTitle className="text-lg">{t('login.title')}</CardTitle>
            <CardDescription>{t('login.description')}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="grid gap-2">
              <label htmlFor="base-url" className="text-sm font-medium">
                {t('login.baseUrl')}
              </label>
              <Input
                id="base-url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="http://127.0.0.1:8920"
              />
            </div>
            <div className="grid gap-2">
              <label htmlFor="api-key" className="text-sm font-medium">
                {t('login.apiKey')}
              </label>
              <Input
                id="api-key"
                type="password"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder={t('login.apiKeyHint')}
              />
            </div>
            {error && (
              <Alert variant="destructive">
                <AlertTitle>{t('login.connectionFailed')}</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            {connectedInfo && (
              <Alert>
                <AlertTitle>{t('login.connectionSuccess')}</AlertTitle>
                <AlertDescription>{connectedInfo}</AlertDescription>
              </Alert>
            )}
            <Button onClick={handleConnect} disabled={testing}>
              {testing ? t('login.connecting') : t('login.connect')}
            </Button>
          </CardContent>
        </Card>
      </div>
    </ThemeProvider>
  )
}