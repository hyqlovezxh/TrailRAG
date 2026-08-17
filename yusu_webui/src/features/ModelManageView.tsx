import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import {
  CheckCircle2Icon,
  PlusIcon,
  RefreshCwIcon,
  ServerIcon,
  Trash2Icon,
  XCircleIcon
} from 'lucide-react'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Badge from '@/components/ui/Badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger
} from '@/components/ui/AlertDialog'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import Checkbox from '@/components/ui/Checkbox'
import {
  createModelProvider,
  deleteModelProvider,
  fetchRemoteModels,
  getDefaultModels,
  getModelStatus,
  getModelsV2,
  listModelProviders,
  refreshModelCache,
  updateDefaultModels,
  updateModelProvider,
  type ModelDefaults,
  type ModelProvider,
  type ModelSpecInfo,
  type RemoteModel
} from '@/api/yusu'
import { cn } from '@/lib/utils'

const CAPABILITY_OPTIONS = ['chat', 'embedding', 'rerank'] as const

interface ProviderFormState {
  provider_id: string
  display_name: string
  base_url: string
  embedding_base_url: string
  rerank_base_url: string
  models_endpoint: string
  embedding_models_endpoint: string
  rerank_models_endpoint: string
  api_key_env: string
  capabilities: string[]
  enabled_models: string[]
  is_enabled: boolean
}

const emptyForm = (): ProviderFormState => ({
  provider_id: '',
  display_name: '',
  base_url: '',
  embedding_base_url: '',
  rerank_base_url: '',
  models_endpoint: '/models',
  embedding_models_endpoint: '/embeddings/models',
  rerank_models_endpoint: '',
  api_key_env: '',
  capabilities: ['chat'],
  enabled_models: [],
  is_enabled: true
})

const formFromProvider = (p: ModelProvider): ProviderFormState => ({
  provider_id: p.provider_id,
  display_name: p.display_name,
  base_url: p.base_url,
  embedding_base_url: p.embedding_base_url ?? '',
  rerank_base_url: p.rerank_base_url ?? '',
  models_endpoint: p.models_endpoint,
  embedding_models_endpoint: p.embedding_models_endpoint ?? '',
  rerank_models_endpoint: p.rerank_models_endpoint ?? '',
  api_key_env: p.api_key_env,
  capabilities: [...p.capabilities],
  enabled_models: [...p.enabled_models],
  is_enabled: p.is_enabled
})

const specLabel = (spec: ModelSpecInfo): string =>
  spec.display_name && spec.display_name !== spec.model_id ? `${spec.display_name} (${spec.spec})` : spec.spec

function ProviderFormDialog({
  open,
  onOpenChange,
  initial,
  onSaved
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  initial: ProviderFormState | null
  onSaved: () => void
}) {
  const { t } = useTranslation()
  const [form, setForm] = useState<ProviderFormState>(emptyForm())
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (open) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset form on dialog open
      setForm(initial ? { ...initial } : emptyForm())
    }
  }, [open, initial])

  const set = <K extends keyof ProviderFormState>(key: K, value: ProviderFormState[K]) =>
    setForm((prev) => ({ ...prev, [key]: value }))

  const toggleCapability = (cap: string, checked: boolean) => {
    setForm((prev) => {
      const next = checked ? [...prev.capabilities, cap] : prev.capabilities.filter((c) => c !== cap)
      return { ...prev, capabilities: next }
    })
  }

  const handleSubmit = async () => {
    setBusy(true)
    try {
      const payload = {
        display_name: form.display_name,
        base_url: form.base_url,
        embedding_base_url: form.embedding_base_url || null,
        rerank_base_url: form.rerank_base_url || null,
        models_endpoint: form.models_endpoint,
        embedding_models_endpoint: form.embedding_models_endpoint || null,
        rerank_models_endpoint: form.rerank_models_endpoint || null,
        api_key_env: form.api_key_env || null,
        capabilities: form.capabilities,
        enabled_models: form.enabled_models,
        is_enabled: form.is_enabled
      }
      if (initial) {
        await updateModelProvider(initial.provider_id, payload)
        toast.success(t('modelManage.updateSuccess'))
      } else {
        await createModelProvider({ ...payload, provider_id: form.provider_id.trim() })
        toast.success(t('modelManage.createSuccess'))
      }
      onOpenChange(false)
      onSaved()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{initial ? t('modelManage.editTitle') : t('modelManage.addTitle')}</DialogTitle>
          <DialogDescription>{t('modelManage.formHint')}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-3">
          {!initial && (
            <Input
              value={form.provider_id}
              onChange={(e) => set('provider_id', e.target.value)}
              placeholder="provider_id"
            />
          )}
          <Input
            value={form.display_name}
            onChange={(e) => set('display_name', e.target.value)}
            placeholder={t('modelManage.displayName')}
          />
          <Input
            value={form.base_url}
            onChange={(e) => set('base_url', e.target.value)}
            placeholder="https://api.example.com/v1"
          />
          <div className="grid grid-cols-2 gap-2">
            <Input
              value={form.embedding_base_url}
              onChange={(e) => set('embedding_base_url', e.target.value)}
              placeholder={t('modelManage.embeddingBaseUrl')}
            />
            <Input
              value={form.rerank_base_url}
              onChange={(e) => set('rerank_base_url', e.target.value)}
              placeholder={t('modelManage.rerankBaseUrl')}
            />
            <Input
              value={form.models_endpoint}
              onChange={(e) => set('models_endpoint', e.target.value)}
              placeholder={t('modelManage.modelsEndpoint')}
            />
            <Input
              value={form.embedding_models_endpoint}
              onChange={(e) => set('embedding_models_endpoint', e.target.value)}
              placeholder={t('modelManage.embeddingEndpoint')}
            />
            <Input
              value={form.rerank_models_endpoint}
              onChange={(e) => set('rerank_models_endpoint', e.target.value)}
              placeholder={t('modelManage.rerankEndpoint')}
            />
          </div>
          <Input
            value={form.api_key_env}
            onChange={(e) => set('api_key_env', e.target.value)}
            placeholder="OPENAI_API_KEY"
          />
          <p className="text-muted-foreground text-xs">{t('modelManage.apiKeyEnvHint')}</p>
          <div className="flex items-center gap-4">
            {CAPABILITY_OPTIONS.map((cap) => (
              <label key={cap} className="flex cursor-pointer items-center gap-1.5 text-sm">
                <Checkbox
                  checked={form.capabilities.includes(cap)}
                  onCheckedChange={(checked) => toggleCapability(cap, checked === true)}
                />
                {cap}
              </label>
            ))}
          </div>
          <label className="flex cursor-pointer items-center gap-1.5 text-sm">
            <Checkbox
              checked={form.is_enabled}
              onCheckedChange={(checked) => set('is_enabled', checked === true)}
            />
            {t('modelManage.enabled')}
          </label>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleSubmit} disabled={busy || !form.display_name.trim() || !form.base_url.trim()}>
            {busy ? t('common.loading') : t('common.confirm')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function RemoteModelsDialog({
  provider,
  open,
  onOpenChange,
  onUpdated
}: {
  provider: ModelProvider
  open: boolean
  onOpenChange: (open: boolean) => void
  onUpdated: () => void
}) {
  const { t } = useTranslation()
  const [models, setModels] = useState<RemoteModel[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [enabledIds, setEnabledIds] = useState<Set<string>>(new Set())
  const [busyId, setBusyId] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    // eslint-disable-next-line react-hooks/set-state-in-effect -- init dialog state on open
    setLoading(true)
    setError(null)
    setEnabledIds(new Set(provider.enabled_models))
    fetchRemoteModels(provider.provider_id)
      .then(({ data }) => setModels(data))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [open, provider])

  const toggleModel = async (model: RemoteModel, checked: boolean) => {
    setBusyId(model.id)
    try {
      const type = model.type === 'embedding' ? 'embedding' : model.type === 'rerank' ? 'rerank' : 'chat'
      const caps = new Set(provider.capabilities)
      const enabled = new Set(enabledIds)
      if (checked) {
        caps.add(type)
        enabled.add(model.id)
      } else {
        enabled.delete(model.id)
      }
      const { data } = await updateModelProvider(provider.provider_id, {
        capabilities: [...caps],
        enabled_models: [...enabled]
      })
      setEnabledIds(new Set(data.enabled_models))
      onUpdated()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyId(null)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {t('modelManage.remoteModels')} · {provider.display_name}
          </DialogTitle>
          <DialogDescription>{t('modelManage.remoteHint')}</DialogDescription>
        </DialogHeader>
        {loading ? (
          <p className="text-muted-foreground py-8 text-center text-sm">{t('modelManage.fetching')}</p>
        ) : error ? (
          <p className="text-destructive py-8 text-center text-sm">
            {t('modelManage.fetchError', { message: error })}
          </p>
        ) : models.length === 0 ? (
          <p className="text-muted-foreground py-8 text-center text-sm">{t('modelManage.noRemoteModels')}</p>
        ) : (
          <ScrollArea className="max-h-[50vh]">
            <div className="grid gap-1.5">
              {models.map((model) => (
                <label
                  key={model.id}
                  className={cn(
                    'hover:bg-accent flex cursor-pointer items-start gap-2 rounded-md border p-2',
                    busyId === model.id && 'opacity-60'
                  )}
                >
                  <Checkbox
                    className="mt-0.5"
                    checked={enabledIds.has(model.id)}
                    onCheckedChange={(checked) => toggleModel(model, checked === true)}
                    disabled={busyId !== null}
                  />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{model.display_name || model.id}</p>
                    <p className="text-muted-foreground truncate text-xs">{model.id}</p>
                    <p className="text-muted-foreground text-xs">
                      {model.type ?? 'chat'}
                      {model.context_length ? ` · ${model.context_length} ctx` : ''}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </ScrollArea>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function DefaultModelSelects({
  defaults,
  chatModels,
  embeddingModels,
  rerankModels,
  onChange
}: {
  defaults: ModelDefaults
  chatModels: ModelSpecInfo[]
  embeddingModels: ModelSpecInfo[]
  rerankModels: ModelSpecInfo[]
  onChange: (defaults: ModelDefaults) => void
}) {
  const { t } = useTranslation()
  const [saving, setSaving] = useState(false)

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateDefaultModels({
        default_chat_model_spec: defaults.default_chat_model_spec,
        default_embedding_model_spec: defaults.default_embedding_model_spec,
        default_rerank_model_spec: defaults.default_rerank_model_spec
      })
      toast.success(t('modelManage.saveSuccess'))
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const renderSelect = (label: string, value: string | null, options: ModelSpecInfo[], key: keyof ModelDefaults) => (
    <div className="grid gap-1.5">
      <span className="text-sm font-medium">{label}</span>
      <Select value={value ?? ''} onValueChange={(v) => onChange({ ...defaults, [key]: v || null })}>
        <SelectTrigger>
          <SelectValue placeholder={t('modelManage.none')} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="">{t('modelManage.none')}</SelectItem>
          {options.map((spec) => (
            <SelectItem key={spec.spec} value={spec.spec}>
              {specLabel(spec)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{t('modelManage.defaults')}</CardTitle>
        <CardDescription>{t('modelManage.defaultsHint')}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="grid gap-4 md:grid-cols-3">
          {renderSelect(t('modelManage.defaultChat'), defaults.default_chat_model_spec, chatModels, 'default_chat_model_spec')}
          {renderSelect(
            t('modelManage.defaultEmbedding'),
            defaults.default_embedding_model_spec,
            embeddingModels,
            'default_embedding_model_spec'
          )}
          {renderSelect(t('modelManage.defaultRerank'), defaults.default_rerank_model_spec, rerankModels, 'default_rerank_model_spec')}
        </div>
        <div>
          <Button onClick={handleSave} disabled={saving} size="sm">
            {saving ? t('common.loading') : t('modelManage.saveDefaults')}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export default function ModelManageView() {
  const { t } = useTranslation()
  const [providers, setProviders] = useState<ModelProvider[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [defaults, setDefaults] = useState<ModelDefaults>({
    default_chat_model_spec: null,
    default_embedding_model_spec: null,
    default_rerank_model_spec: null
  })
  const [chatModels, setChatModels] = useState<ModelSpecInfo[]>([])
  const [embeddingModels, setEmbeddingModels] = useState<ModelSpecInfo[]>([])
  const [rerankModels, setRerankModels] = useState<ModelSpecInfo[]>([])
  const [formOpen, setFormOpen] = useState(false)
  const [remoteOpen, setRemoteOpen] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ status: string; message: string } | null>(null)

  const selected = useMemo(
    () => providers.find((p) => p.provider_id === selectedId) ?? null,
    [providers, selectedId]
  )

  const loadAll = useCallback(async () => {
    try {
      const [providersRes, defaultsRes, chatRes, embedRes, rerankRes] = await Promise.all([
        listModelProviders(),
        getDefaultModels(),
        getModelsV2('chat'),
        getModelsV2('embedding'),
        getModelsV2('rerank')
      ])
      setProviders(providersRes.data)
      setDefaults(defaultsRes.data)
      setChatModels(Object.values(chatRes.data).flatMap((g) => g.models))
      setEmbeddingModels(Object.values(embedRes.data).flatMap((g) => g.models))
      setRerankModels(Object.values(rerankRes.data).flatMap((g) => g.models))
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data load
    loadAll()
  }, [loadAll])

  useEffect(() => {
    if (selectedId && !providers.some((p) => p.provider_id === selectedId)) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- keep selection valid
      setSelectedId(providers[0]?.provider_id ?? null)
    }
  }, [providers, selectedId])

  const handleTest = async () => {
    if (!selected) return
    const candidate = Object.values(chatModels).find((m) => m.spec.startsWith(`${selected.provider_id}:`))
    if (!candidate) {
      toast.error(t('modelManage.noModelsForProvider'))
      return
    }
    setTesting(true)
    try {
      const { data } = await getModelStatus(candidate.spec)
      setTestResult(data)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setTesting(false)
    }
  }

  const handleRefreshCache = async () => {
    try {
      const result = await refreshModelCache()
      toast.success(result.message)
      await loadAll()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  const handleDelete = async (provider: ModelProvider) => {
    try {
      await deleteModelProvider(provider.provider_id)
      toast.success(t('modelManage.deleteSuccess'))
      await loadAll()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="flex h-full">
      <div className="flex w-72 shrink-0 flex-col border-r">
        <div className="flex items-center justify-between gap-2 p-3">
          <div>
            <p className="text-sm font-semibold">{t('modelManage.providers')}</p>
            <p className="text-muted-foreground text-xs">{t('modelManage.providerHint')}</p>
          </div>
          <Button size="icon" variant="outline" side="bottom" tooltip={t('modelManage.refresh')} onClick={handleRefreshCache}>
            <RefreshCwIcon className="size-4" aria-hidden="true" />
          </Button>
        </div>
        <div className="flex gap-2 px-3 pb-2">
          <Dialog open={formOpen} onOpenChange={setFormOpen}>
            <DialogTrigger asChild>
              <Button size="sm" className="w-full">
                <PlusIcon aria-hidden="true" />
                {t('modelManage.addProvider')}
              </Button>
            </DialogTrigger>
            <ProviderFormDialog open={formOpen} onOpenChange={setFormOpen} initial={null} onSaved={loadAll} />
          </Dialog>
        </div>
        <ScrollArea className="grow">
          <div className="grid gap-1.5 p-2">
            {providers.map((provider) => (
              <button
                key={provider.provider_id}
                onClick={() => setSelectedId(provider.provider_id)}
                className={cn(
                  'hover:bg-accent flex items-center justify-between rounded-md border p-2 text-left',
                  selectedId === provider.provider_id && 'border-emerald-400 bg-emerald-400/10'
                )}
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{provider.display_name}</p>
                  <p className="text-muted-foreground truncate text-xs">{provider.provider_id}</p>
                </div>
                {provider.credential_status === 'configured' ? (
                  <CheckCircle2Icon className="text-emerald-500 size-4 shrink-0" aria-label={t('modelManage.credentialConfigured')} />
                ) : (
                  <XCircleIcon className="text-destructive size-4 shrink-0" aria-label={t('modelManage.credentialMissing')} />
                )}
              </button>
            ))}
            {providers.length === 0 && (
              <p className="text-muted-foreground p-4 text-center text-sm">{t('modelManage.noProviders')}</p>
            )}
          </div>
        </ScrollArea>
      </div>

      <div className="flex min-w-0 grow flex-col gap-4 overflow-y-auto p-4">
        {selected ? (
          <>
            <Card>
              <CardHeader className="flex flex-row items-start justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <ServerIcon className="size-4" aria-hidden="true" />
                    {selected.display_name}
                    {selected.is_builtin && <Badge variant="secondary">{t('modelManage.builtin')}</Badge>}
                    {!selected.is_enabled && <Badge variant="destructive">{t('modelManage.disabled')}</Badge>}
                  </CardTitle>
                  <CardDescription className="mt-1 font-mono text-xs">{selected.provider_id}</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  {!selected.is_builtin && (
                    <>
                      <Dialog open={formOpen} onOpenChange={setFormOpen}>
                        <DialogTrigger asChild>
                          <Button size="sm" variant="outline" onClick={() => setSelectedId(selected.provider_id)}>
                            {t('modelManage.edit')}
                          </Button>
                        </DialogTrigger>
                        <ProviderFormDialog
                          open={formOpen}
                          onOpenChange={setFormOpen}
                          initial={formFromProvider(selected)}
                          onSaved={loadAll}
                        />
                      </Dialog>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button size="sm" variant="outline">
                            <Trash2Icon className="size-4" aria-hidden="true" />
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>{t('modelManage.deleteConfirmTitle')}</AlertDialogTitle>
                            <AlertDialogDescription>
                              {t('modelManage.deleteConfirm', { name: selected.display_name })}
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                            <AlertDialogAction onClick={() => handleDelete(selected)}>
                              {t('modelManage.delete')}
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </>
                  )}
                </div>
              </CardHeader>
              <CardContent className="grid gap-3 text-sm">
                <div className="grid grid-cols-2 gap-x-6 gap-y-2">
                  <div>
                    <span className="text-muted-foreground">{t('modelManage.providerType')}: </span>
                    {selected.provider_type}
                  </div>
                  <div>
                    <span className="text-muted-foreground">{t('modelManage.apiKeyEnv')}: </span>
                    <code className="rounded bg-muted px-1">{selected.api_key_env || '-'}</code>
                  </div>
                  <div className="col-span-2">
                    <span className="text-muted-foreground">{t('modelManage.baseUrl')}: </span>
                    <code className="rounded bg-muted px-1">{selected.base_url}</code>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-muted-foreground text-xs">{t('modelManage.capabilities')}:</span>
                  {selected.capabilities.map((cap) => (
                    <Badge key={cap} variant="outline">
                      {cap}
                    </Badge>
                  ))}
                </div>
                {testResult && (
                  <div
                    className={cn(
                      'rounded-md border p-2 text-xs',
                      testResult.status === 'available' ? 'border-emerald-400/40 bg-emerald-400/10' : 'border-destructive/40 bg-destructive/10'
                    )}
                  >
                    <span className="font-medium">
                      {testResult.status === 'available' ? t('modelManage.testAvailable') : t('modelManage.testFailed')}
                    </span>
                    : {testResult.message}
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <Button size="sm" variant="outline" onClick={() => setRemoteOpen(true)}>
                    <RefreshCwIcon className="size-4" aria-hidden="true" />
                    {t('modelManage.fetchRemote')}
                  </Button>
                  <Button size="sm" variant="outline" onClick={handleTest} disabled={testing}>
                    {testing ? t('modelManage.testing') : t('modelManage.testConnection')}
                  </Button>
                </div>
              </CardContent>
            </Card>
            <RemoteModelsDialog
              provider={selected}
              open={remoteOpen}
              onOpenChange={setRemoteOpen}
              onUpdated={loadAll}
            />
          </>
        ) : (
          <div className="text-muted-foreground flex grow items-center justify-center text-sm">
            {t('modelManage.noSelection')}
          </div>
        )}

        <DefaultModelSelects
          defaults={defaults}
          chatModels={chatModels}
          embeddingModels={embeddingModels}
          rerankModels={rerankModels}
          onChange={setDefaults}
        />
      </div>
    </div>
  )
}
