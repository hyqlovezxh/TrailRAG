import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { DatabaseIcon, FileTextIcon, NetworkIcon, PlayIcon, PlusIcon, RefreshCwIcon, RotateCcwIcon, SearchIcon, Trash2Icon } from 'lucide-react'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import Badge from '@/components/ui/Badge'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/Dialog'
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/AlertDialog'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table'
import { ScrollArea } from '@/components/ui/ScrollArea'
import Separator from '@/components/ui/Separator'
import {
  buildGraph,
  createDatabase,
  deleteDatabase,
  deleteFile,
  getDatabase,
  getGraphStatus,
  indexFile,
  listDatabases,
  parseFile,
  queryKb,
  refreshStats,
  resetGraph,
  uploadFile,
  type Chunk,
  type FileRecord,
  type GraphStatus,
  type KnowledgeBase
} from '@/api/yusu'
import { useSettingsStore } from '@/stores/settings'
import { acceptedFileExtensions } from '@/lib/constants'
import { cn } from '@/lib/utils'

const statusVariant = (status: FileRecord['status']): 'default' | 'secondary' | 'destructive' | 'outline' => {
  switch (status) {
    case 'indexed':
      return 'default'
    case 'parsed':
      return 'secondary'
    case 'error_parsing':
    case 'error_indexing':
      return 'destructive'
    default:
      return 'outline'
  }
}

function CreateDatabaseDialog({ onCreated }: { onCreated: (kb: KnowledgeBase) => void }) {
  const { t } = useTranslation()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)

  const handleCreate = async () => {
    if (!name.trim()) return
    setCreating(true)
    try {
      const kb = await createDatabase(name.trim(), description.trim())
      setOpen(false)
      setName('')
      setDescription('')
      toast.success(t('kb.createSuccess'))
      onCreated(kb)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setCreating(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm">
          <PlusIcon aria-hidden="true" />
          {t('kb.create')}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('kb.createTitle')}</DialogTitle>
          <DialogDescription>{t('kb.createDescription')}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4">
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={t('kb.namePlaceholder')}
            maxLength={64}
          />
          <Textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder={t('kb.descriptionPlaceholder')}
            rows={3}
          />
        </div>
        <DialogFooter>
          <Button onClick={handleCreate} disabled={creating || !name.trim()}>
            {creating ? t('common.loading') : t('common.confirm')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function FileActions({ kbId, file, onChanged }: { kbId: string; file: FileRecord; onChanged: () => void }) {
  const { t } = useTranslation()
  const [busy, setBusy] = useState(false)

  const run = async (action: () => Promise<unknown>, successMessage: string) => {
    setBusy(true)
    try {
      await action()
      toast.success(successMessage)
      onChanged()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const canParse = file.status === 'uploaded' || file.status === 'error_parsing'
  const canIndex = file.status === 'parsed' || file.status === 'error_indexing'

  return (
    <div className="flex items-center gap-1">
      {canParse && (
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() => run(() => parseFile(kbId, file.file_id), t('kb.parseSuccess'))}
        >
          {t('kb.parse')}
        </Button>
      )}
      {canIndex && (
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() => run(() => indexFile(kbId, file.file_id), t('kb.indexSuccess'))}
        >
          {t('kb.index')}
        </Button>
      )}
      {file.status === 'indexed' && (
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() => run(() => indexFile(kbId, file.file_id), t('kb.indexSuccess'))}
        >
          {t('kb.reindex')}
        </Button>
      )}
      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button size="sm" variant="ghost" disabled={busy}>
            <Trash2Icon className="text-destructive" aria-hidden="true" />
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('kb.deleteFileConfirmTitle')}</AlertDialogTitle>
            <AlertDialogDescription>
              {t('kb.deleteFileConfirm', { filename: file.filename })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => run(() => deleteFile(kbId, file.file_id), t('kb.deleteFileSuccess'))}
            >
              {t('common.confirm')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

function RetrievalTest({ kbId }: { kbId: string }) {
  const { t } = useTranslation()
  const queryParams = useSettingsStore.use.queryParams()
  const setQueryParams = useSettingsStore.use.setQueryParams()
  const [query, setQuery] = useState('')
  const [paramsText, setParamsText] = useState(JSON.stringify(queryParams, null, 2))
  const [results, setResults] = useState<Chunk[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return
    setSearching(true)
    setError(null)
    try {
      let params: Record<string, unknown>
      try {
        params = JSON.parse(paramsText)
      } catch {
        setError(t('kb.paramsInvalidJson'))
        return
      }
      setQueryParams(params)
      const output = await queryKb(kbId, query.trim(), params)
      setResults(output.results)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setResults(null)
    } finally {
      setSearching(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <SearchIcon className="size-4" aria-hidden="true" />
          {t('kb.retrievalTest')}
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder={t('kb.queryPlaceholder')}
        />
        <Textarea
          value={paramsText}
          onChange={(e) => setParamsText(e.target.value)}
          rows={4}
          className="font-mono text-xs"
          placeholder="{}"
        />
        <Button onClick={handleSearch} disabled={searching || !query.trim()} className="w-fit">
          {searching ? t('common.loading') : t('kb.search')}
        </Button>
        {error && <p className="text-destructive text-sm">{error}</p>}
        {results !== null && (
          <ScrollArea className="max-h-80 rounded-md border p-3">
            <div className="grid gap-3">
              {results.length === 0 && <p className="text-muted-foreground text-sm">{t('kb.noResults')}</p>}
              {results.map((chunk, index) => (
                <div key={chunk.id} className="text-sm">
                  <p className="mb-1 font-medium">
                    [{index + 1}] {chunk.file_id}
                  </p>
                  <p className="text-muted-foreground whitespace-pre-wrap">{chunk.content}</p>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  )
}

function GraphStatusCard({ kbId }: { kbId: string }) {
  const { t } = useTranslation()
  const [status, setStatus] = useState<GraphStatus | null>(null)
  const [building, setBuilding] = useState(false)

  useEffect(() => {
    let disposed = false
    const poll = async () => {
      try {
        const next = await getGraphStatus(kbId)
        if (!disposed) setStatus(next)
      } catch {
        // keep the last known status while polling
      }
    }
    void poll()
    const timer = setInterval(() => void poll(), 3000)
    return () => {
      disposed = true
      clearInterval(timer)
    }
  }, [kbId])

  const buildingNow = status?.build_task_status === 'running' || status?.build_task_status === 'pending'
  const progress = Math.round(
    status && status.total_chunks > 0 ? (status.indexed_chunks / status.total_chunks) * 100 : 0
  )

  const handleBuild = async () => {
    setBuilding(true)
    try {
      const result = await buildGraph(kbId)
      toast.success(t('graph.buildStarted'))
      setStatus(result.status)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setBuilding(false)
    }
  }

  const handleReset = async () => {
    try {
      await resetGraph(kbId)
      toast.success(t('graph.resetSuccess'))
      const next = await getGraphStatus(kbId)
      setStatus(next)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <NetworkIcon className="size-4" aria-hidden="true" />
          {t('graph.title')}
        </CardTitle>
        <CardDescription>{t('graph.statusHint')}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 text-sm">
        {status ? (
          <div className="grid grid-cols-2 gap-x-6 gap-y-2">
            <div>
              <span className="text-muted-foreground">{t('graph.entities')}: </span>
              <span className="font-medium">{status.entity_count}</span>
            </div>
            <div>
              <span className="text-muted-foreground">{t('graph.relations')}: </span>
              <span className="font-medium">{status.relation_count}</span>
            </div>
            <div>
              <span className="text-muted-foreground">{t('graph.chunks')}: </span>
              <span className="font-medium">
                {status.indexed_chunks}/{status.total_chunks}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">{t('graph.pending')}: </span>
              <span className="font-medium">{status.pending_chunks}</span>
            </div>
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">{t('graph.noStatus')}</p>
        )}
        <div>
          <div className="mb-1 flex justify-between text-xs">
            <span className="text-muted-foreground">{t('graph.buildProgress')}</span>
            <span>{progress}%</span>
          </div>
          <div className="bg-muted h-2 overflow-hidden rounded-full">
            <div
              className={cn('h-full rounded-full transition-all', buildingNow ? 'bg-emerald-400' : 'bg-emerald-400/40')}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={handleBuild} disabled={building || buildingNow}>
            <PlayIcon aria-hidden="true" />
            {building ? t('common.loading') : t('graph.build')}
          </Button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button size="sm" variant="outline">
                <RotateCcwIcon aria-hidden="true" />
                {t('graph.reset')}
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>{t('graph.resetConfirmTitle')}</AlertDialogTitle>
                <AlertDialogDescription>{t('graph.resetConfirm')}</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                <AlertDialogAction onClick={() => void handleReset()}>{t('common.confirm')}</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </CardContent>
    </Card>
  )
}

export default function KnowledgeBaseView() {
  const { t } = useTranslation()
  const currentKbId = useSettingsStore.use.currentKbId()
  const setCurrentKbId = useSettingsStore.use.setCurrentKbId()

  const [databases, setDatabases] = useState<KnowledgeBase[]>([])
  const [selected, setSelected] = useState<KnowledgeBase | null>(null)
  const [files, setFiles] = useState<FileRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const reloadDatabases = useCallback(async () => {
    try {
      const { databases } = await listDatabases()
      setDatabases(databases)
      if (currentKbId && !databases.some((db) => db.kb_id === currentKbId)) {
        setCurrentKbId(null)
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [currentKbId, setCurrentKbId])

  const reloadFiles = useCallback(async (kbId: string) => {
    try {
      const info = await getDatabase(kbId)
      setSelected(info)
      setFiles(info.files)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    void (async () => {
      try {
        const { databases } = await listDatabases()
        setDatabases(databases)
        if (currentKbId && !databases.some((db) => db.kb_id === currentKbId)) {
          setCurrentKbId(null)
        }
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e))
      } finally {
        setLoading(false)
      }
    })()
  }, [currentKbId, setCurrentKbId])

  useEffect(() => {
    void (async () => {
      if (!currentKbId) {
        setSelected(null)
        setFiles([])
        return
      }
      try {
        const info = await getDatabase(currentKbId)
        setSelected(info)
        setFiles(info.files)
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e))
      }
    })()
  }, [currentKbId, reloadFiles])

  const handleSelect = (kbId: string) => {
    setCurrentKbId(kbId === currentKbId ? null : kbId)
  }

  const handleCreated = (kb: KnowledgeBase) => {
    reloadDatabases()
    setCurrentKbId(kb.kb_id)
  }

  const handleDeleteKb = async () => {
    if (!selected) return
    try {
      await deleteDatabase(selected.kb_id)
      toast.success(t('kb.deleteSuccess'))
      setCurrentKbId(null)
      reloadDatabases()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  const handleUpload = async (file: File) => {
    if (!currentKbId) return
    setUploading(true)
    try {
      await uploadFile(currentKbId, file)
      toast.success(t('kb.uploadSuccess'))
      reloadFiles(currentKbId)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleRefreshStats = async () => {
    if (!selected) return
    try {
      await refreshStats(selected.kb_id)
      toast.success(t('kb.refreshStatsSuccess'))
      reloadFiles(selected.kb_id)
      reloadDatabases()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  return (
    <div className="flex h-full gap-4 p-4">
      <Card className="flex w-72 shrink-0 flex-col">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <DatabaseIcon className="size-4" aria-hidden="true" />
            {t('kb.databases')}
          </CardTitle>
          <CardDescription>{t('kb.databasesHint')}</CardDescription>
          <div className="flex items-center gap-2">
            <CreateDatabaseDialog onCreated={handleCreated} />
            <Button size="sm" variant="ghost" onClick={reloadDatabases} disabled={loading}>
              <RefreshCwIcon aria-hidden="true" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grow overflow-auto">
          <div className="grid gap-1">
            {databases.length === 0 && !loading && (
              <p className="text-muted-foreground text-sm">{t('kb.noDatabases')}</p>
            )}
            {databases.map((db) => (
              <button
                key={db.kb_id}
                onClick={() => handleSelect(db.kb_id)}
                className={cn(
                  'flex flex-col rounded-md border px-3 py-2 text-left transition-colors cursor-pointer',
                  currentKbId === db.kb_id
                    ? 'border-emerald-400 bg-emerald-400/10'
                    : 'hover:bg-accent'
                )}
              >
                <span className="text-sm font-medium">{db.name}</span>
                <span className="text-muted-foreground text-xs">
                  {db.stats?.file_count ?? 0} {t('kb.files')} · {db.stats?.chunk_count ?? 0} {t('kb.chunks')}
                </span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid min-w-0 grow grid-cols-1 gap-4 overflow-auto xl:grid-cols-[1fr_380px]">
        {selected ? (
          <>
            <Card>
              <CardHeader>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <CardTitle className="text-base">{selected.name}</CardTitle>
                    {selected.description && (
                      <CardDescription>{selected.description}</CardDescription>
                    )}
                  </div>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button size="sm" variant="ghost">
                        <Trash2Icon className="text-destructive" aria-hidden="true" />
                        {t('kb.delete')}
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>{t('kb.deleteConfirmTitle')}</AlertDialogTitle>
                        <AlertDialogDescription>
                          {t('kb.deleteConfirm', { name: selected.name })}
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                        <AlertDialogAction
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                          onClick={handleDeleteKb}
                        >
                          {t('common.confirm')}
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="flex items-center gap-2">
                  <Button size="sm" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                    <PlusIcon aria-hidden="true" />
                    {uploading ? t('common.loading') : t('kb.upload')}
                  </Button>
                  <Button size="sm" variant="outline" onClick={handleRefreshStats}>
                    <RefreshCwIcon aria-hidden="true" />
                    {t('kb.refreshStats')}
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept={acceptedFileExtensions.join(',')}
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) handleUpload(file)
                    }}
                  />
                </div>
                <Separator />
                {files.length === 0 ? (
                  <p className="text-muted-foreground text-sm">{t('kb.noFiles')}</p>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>{t('kb.fileName')}</TableHead>
                        <TableHead>{t('kb.status')}</TableHead>
                        <TableHead className="text-right">{t('kb.size')}</TableHead>
                        <TableHead className="text-right">{t('kb.chunks')}</TableHead>
                        <TableHead className="text-right">{t('kb.actions')}</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {files.map((file) => (
                        <TableRow key={file.file_id}>
                          <TableCell>
                            <span className="flex items-center gap-2">
                              <FileTextIcon className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                              <span className="max-w-56 truncate" title={file.filename}>
                                {file.filename}
                              </span>
                            </span>
                          </TableCell>
                          <TableCell>
                            <Badge variant={statusVariant(file.status)}>{file.status}</Badge>
                            {file.error && (
                              <span className="text-muted-foreground ml-2 text-xs" title={file.error}>
                                {file.error}
                              </span>
                            )}
                          </TableCell>
                          <TableCell className="text-right">{formatSize(file.size)}</TableCell>
                          <TableCell className="text-right">{file.chunk_count}</TableCell>
                          <TableCell className="text-right">
                            <FileActions kbId={selected.kb_id} file={file} onChanged={() => reloadFiles(selected.kb_id)} />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
            <RetrievalTest kbId={selected.kb_id} />
            <GraphStatusCard kbId={selected.kb_id} />
          </>
        ) : (
          <Card className="flex items-center justify-center">
            <CardContent className="text-muted-foreground py-12 text-center">
              {t('kb.selectKbHint')}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}