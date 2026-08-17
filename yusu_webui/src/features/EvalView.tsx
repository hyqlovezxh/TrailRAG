import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import {
  CheckCircle2Icon,
  DownloadIcon,
  FilePlus2Icon,
  ListChecksIcon,
  PlayIcon,
  RefreshCwIcon,
  Trash2Icon,
  XCircleIcon
} from 'lucide-react'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import Badge from '@/components/ui/Badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/AlertDialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table'
import {
  createEvalRun,
  deleteEvalDataset,
  deleteEvalRun,
  downloadEvalDataset,
  generateEvalDataset,
  getEvalRun,
  listDatabases,
  listEvalDatasets,
  listEvalRuns,
  uploadEvalDataset,
  type EvalDataset,
  type EvalRun,
  type EvalRunDetail,
  type KnowledgeBase
} from '@/api/yusu'
import { useSettingsStore } from '@/stores/settings'
import { cn } from '@/lib/utils'

const RUNNING_STATUSES = new Set(['pending', 'running'])

const metricValue = (metrics: Record<string, unknown>, key: string): number | null => {
  const value = metrics[key]
  return typeof value === 'number' ? value : null
}

const formatScore = (value: number | null): string => (value === null ? '—' : value.toFixed(3))

const METRIC_KEYS = ['recall@1', 'recall@3', 'recall@5', 'recall@10', 'f1@10', 'ndcg@10', 'answer_correctness'] as const

function MetricCards({ metrics, overallScore }: { metrics: Record<string, unknown>; overallScore: number | null }) {
  const { t } = useTranslation()
  const items = METRIC_KEYS.map((key) => ({ key, value: metricValue(metrics, key) }))
  const judgeStatus = typeof metrics.judge_status === 'string' ? metrics.judge_status : null
  return (
    <div className="grid gap-2">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        {items.map(({ key, value }) => (
          <div key={key} className="rounded-md border p-2">
            <p className="text-muted-foreground text-xs">{key}</p>
            <p className="text-lg font-semibold">{formatScore(value)}</p>
          </div>
        ))}
        <div className="rounded-md border p-2">
          <p className="text-muted-foreground text-xs">{t('eval.overall')}</p>
          <p className="text-lg font-semibold">{formatScore(overallScore ?? metricValue(metrics, 'overall_score'))}</p>
        </div>
      </div>
      {judgeStatus && (
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground">{t('eval.judgeStatus')}:</span>
          <Badge variant={judgeStatus === 'available' ? 'default' : judgeStatus === 'unavailable' ? 'destructive' : 'outline'}>
            {t(`eval.judge_${String(judgeStatus)}`)}
          </Badge>
        </div>
      )}
    </div>
  )
}

function DatasetForm({ kbId, onChanged }: { kbId: string; onChanged: () => void }) {
  const { t } = useTranslation()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [uploading, setUploading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [count, setCount] = useState('5')
  const [generationMode, setGenerationMode] = useState<'vector' | 'graph_enhanced'>('vector')
  const [neighborsCount, setNeighborsCount] = useState('3')
  const [graphExpandTopK, setGraphExpandTopK] = useState('1')
  const [concurrencyCount, setConcurrencyCount] = useState('4')

  const handleUpload = async (file: File) => {
    setUploading(true)
    try {
      await uploadEvalDataset(kbId, file, name.trim(), description.trim())
      toast.success(t('eval.uploadSuccess'))
      setName('')
      setDescription('')
      onChanged()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      await generateEvalDataset({
        kb_id: kbId,
        name: name.trim() || undefined,
        description: description.trim() || undefined,
        count: Math.max(1, Number(count) || 5),
        neighbors_count: Math.max(1, Number(neighborsCount) || 3),
        concurrency_count: Math.max(1, Number(concurrencyCount) || 4),
        generation_mode: generationMode,
        graph_expand_top_k: Math.max(0, Number(graphExpandTopK) || 1)
      })
      toast.success(t('eval.generateStarted'))
      onChanged()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <FilePlus2Icon className="size-4" aria-hidden="true" />
          {t('eval.datasetForm')}
        </CardTitle>
        <CardDescription>{t('eval.datasetFormHint')}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3">
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={t('eval.namePlaceholder')} />
        <Textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder={t('eval.descPlaceholder')} rows={2} />
        <div className="grid grid-cols-2 gap-2">
          <Input type="number" min={1} max={50} value={count} onChange={(e) => setCount(e.target.value)} placeholder={t('eval.count')} />
          <Select value={generationMode} onValueChange={(v) => setGenerationMode(v as 'vector' | 'graph_enhanced')}>
            <SelectTrigger>
              <SelectValue placeholder={t('eval.mode')} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="vector">{t('eval.modeVector')}</SelectItem>
              <SelectItem value="graph_enhanced">{t('eval.modeGraph')}</SelectItem>
            </SelectContent>
          </Select>
          <Input type="number" min={1} max={20} value={neighborsCount} onChange={(e) => setNeighborsCount(e.target.value)} placeholder={t('eval.neighbors')} />
          <Input type="number" min={0} max={10} value={graphExpandTopK} onChange={(e) => setGraphExpandTopK(e.target.value)} placeholder={t('eval.graphTopK')} />
          <Input type="number" min={1} max={16} value={concurrencyCount} onChange={(e) => setConcurrencyCount(e.target.value)} placeholder={t('eval.concurrency')} />
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={handleGenerate} disabled={generating}>
            <PlayIcon aria-hidden="true" />
            {generating ? t('common.loading') : t('eval.generate')}
          </Button>
          <Button size="sm" variant="outline" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
            {uploading ? t('common.loading') : t('eval.upload')}
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".jsonl,application/x-ndjson"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) void handleUpload(file)
            }}
          />
        </div>
      </CardContent>
    </Card>
  )
}

function DatasetList({
  datasets,
  selectedId,
  onSelect,
  onChanged
}: {
  datasets: EvalDataset[]
  selectedId: string | null
  onSelect: (datasetId: string | null) => void
  onChanged: () => void
}) {
  const { t } = useTranslation()

  const handleDownload = async (datasetId: string) => {
    try {
      await downloadEvalDataset(datasetId)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  const handleDelete = async (datasetId: string) => {
    try {
      await deleteEvalDataset(datasetId)
      toast.success(t('eval.datasetDeleted'))
      onChanged()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <ListChecksIcon className="size-4" aria-hidden="true" />
          {t('eval.datasets')}
        </CardTitle>
        <CardDescription>{t('eval.datasetsHint')}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-2">
        {datasets.length === 0 && <p className="text-muted-foreground text-sm">{t('eval.noDatasets')}</p>}
        {datasets.map((dataset) => {
          const status = String(dataset.build_metadata?.status ?? '')
          const generatingNow = RUNNING_STATUSES.has(status)
          return (
            <div
              key={dataset.dataset_id}
              className={cn(
                'grid gap-1 rounded-md border p-2',
                selectedId === dataset.dataset_id && 'border-emerald-400 bg-emerald-400/10'
              )}
            >
              <button
                onClick={() => onSelect(selectedId === dataset.dataset_id ? null : dataset.dataset_id)}
                className="cursor-pointer text-left"
              >
                <p className="truncate text-sm font-medium">{dataset.name || dataset.dataset_id}</p>
                <p className="text-muted-foreground text-xs">
                  {dataset.item_count} {t('eval.items')}
                  {dataset.has_gold_answers ? ' · gold answers' : ''}
                  {dataset.has_gold_chunks ? ' · gold chunks' : ''}
                </p>
              </button>
              <div className="flex items-center gap-2">
                {generatingNow && <Badge variant="secondary">{t('eval.generating')}</Badge>}
                {status === 'failed' && <Badge variant="destructive">{t('eval.failed')}</Badge>}
                <div className="grow" />
                <Button size="icon" variant="ghost" side="top" tooltip={t('eval.download')} onClick={() => void handleDownload(dataset.dataset_id)}>
                  <DownloadIcon className="size-4" aria-hidden="true" />
                </Button>
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button size="icon" variant="ghost" side="top" tooltip={t('eval.delete')}>
                      <Trash2Icon className="text-destructive size-4" aria-hidden="true" />
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>{t('eval.deleteDatasetTitle')}</AlertDialogTitle>
                      <AlertDialogDescription>
                        {t('eval.deleteDatasetConfirm', { name: dataset.name || dataset.dataset_id })}
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                      <AlertDialogAction onClick={() => void handleDelete(dataset.dataset_id)}>{t('common.confirm')}</AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

function RunDetail({ kbId, runId }: { kbId: string; runId: string }) {
  const { t } = useTranslation()
  const [detail, setDetail] = useState<EvalRunDetail | null>(null)
  const [page, setPage] = useState(1)
  const [errorOnly, setErrorOnly] = useState(false)
  const pageSize = 10

  const load = useCallback(async () => {
    try {
      const { data } = await getEvalRun(kbId, runId, page, pageSize, errorOnly)
      setDetail(data)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }, [kbId, runId, page, errorOnly])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial detail load
    void load()
  }, [load])

  useEffect(() => {
    if (!detail || detail.status !== 'running') return
    const timer = setInterval(() => void load(), 3000)
    return () => clearInterval(timer)
  }, [detail, load])

  const running = detail?.status === 'running'
  const pagination = detail?.pagination
  const total = pagination?.total ?? pagination?.total_items ?? 0
  const totalPages = pagination?.total_pages ?? Math.max(1, Math.ceil(total / pageSize))

  if (!detail) {
    return <Card><CardContent className="text-muted-foreground py-6 text-sm">{t('eval.noRunSelected')}</CardContent></Card>
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          {detail.name}
          {running && <Badge variant="secondary">{t('eval.running')}</Badge>}
          {detail.status === 'failed' && <Badge variant="destructive">{t('eval.failed')}</Badge>}
        </CardTitle>
        <CardDescription>
          {detail.completed_items}/{detail.total_items} · {t('eval.startedAt')} {detail.started_at ?? '—'}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <MetricCards metrics={detail.metrics} overallScore={detail.overall_score} />
        <div className="flex items-center gap-2 text-sm">
          <label className="flex cursor-pointer items-center gap-1.5">
            <input
              type="checkbox"
              checked={errorOnly}
              onChange={(e) => {
                setErrorOnly(e.target.checked)
                setPage(1)
              }}
            />
            {t('eval.errorOnly')}
          </label>
          <div className="grow" />
          <Button size="sm" variant="outline" onClick={() => void load()}>
            <RefreshCwIcon aria-hidden="true" />
          </Button>
        </div>
        <ScrollArea className="max-h-[32rem]">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10">#</TableHead>
                <TableHead>{t('eval.query')}</TableHead>
                <TableHead>{t('eval.goldAnswer')}</TableHead>
                <TableHead>{t('eval.generatedAnswer')}</TableHead>
                <TableHead className="text-right">{t('eval.score')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {detail.items.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-muted-foreground text-center">
                    {t('eval.noItems')}
                  </TableCell>
                </TableRow>
              )}
              {detail.items.map((item, index) => {
                const score = metricValue(item.metrics, 'score')
                const isError = (score ?? 1) <= 0.5
                return (
                  <TableRow key={`${detail.run_id}-${(page - 1) * pageSize + index}`} className={cn(isError && 'bg-destructive/5')}>
                    <TableCell className="text-muted-foreground">{(page - 1) * pageSize + index + 1}</TableCell>
                    <TableCell className="max-w-48">
                      <p className="line-clamp-2 text-xs" title={item.query}>{item.query}</p>
                    </TableCell>
                    <TableCell className="max-w-48">
                      <p className="line-clamp-2 text-xs whitespace-pre-wrap" title={item.gold_answer ?? ''}>{item.gold_answer ?? '—'}</p>
                    </TableCell>
                    <TableCell className="max-w-48">
                      <p className="line-clamp-2 text-xs whitespace-pre-wrap" title={item.generated_answer ?? ''}>{item.generated_answer ?? '—'}</p>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className={cn('text-xs font-medium', isError && 'text-destructive')}>{formatScore(score)}</span>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </ScrollArea>
        {total > 0 && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground text-xs">
              {t('eval.page')} {page}/{totalPages} · {total} {t('eval.items')}
            </span>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" disabled={!pagination?.has_prev || page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>
                {t('eval.prev')}
              </Button>
              <Button size="sm" variant="outline" disabled={!pagination?.has_next || page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                {t('eval.next')}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function EvalView() {
  const { t } = useTranslation()
  const currentKbId = useSettingsStore.use.currentKbId()
  const setCurrentKbId = useSettingsStore.use.setCurrentKbId()

  const [databases, setDatabases] = useState<KnowledgeBase[]>([])
  const [datasets, setDatasets] = useState<EvalDataset[]>([])
  const [runs, setRuns] = useState<EvalRun[]>([])
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [runningRun, setRunningRun] = useState(false)
  const [creatingRun, setCreatingRun] = useState(false)

  const reloadDatasets = useCallback(async (kbId: string) => {
    try {
      const { data } = await listEvalDatasets(kbId)
      setDatasets(data)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const reloadRuns = useCallback(
    async (kbId: string) => {
      try {
        const { data } = await listEvalRuns(kbId)
        setRuns(data)
        const active = data.find((run) => run.status === 'running')
        setRunningRun(Boolean(active))
        if (selectedRunId && !data.some((run) => run.run_id === selectedRunId)) {
          setSelectedRunId(null)
        }
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e))
      }
    },
    [selectedRunId]
  )

  useEffect(() => {
    void (async () => {
      try {
        const { databases } = await listDatabases()
        setDatabases(databases)
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e))
      }
    })()
  }, [])

  useEffect(() => {
    if (!currentKbId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset view state when KB changes
      setDatasets([])
      setRuns([])
      setSelectedDatasetId(null)
      setSelectedRunId(null)
      return
    }
    void reloadDatasets(currentKbId)
    void reloadRuns(currentKbId)
  }, [currentKbId, reloadDatasets, reloadRuns])

  // poll while any dataset generation task is active
  useEffect(() => {
    if (!currentKbId) return
    const generating = datasets.some((d) => RUNNING_STATUSES.has(String(d.build_metadata?.status ?? '')))
    if (!generating) return
    const timer = setInterval(() => void reloadDatasets(currentKbId), 3000)
    return () => clearInterval(timer)
  }, [currentKbId, datasets, reloadDatasets])

  // poll while any run is running
  useEffect(() => {
    if (!currentKbId || !runningRun) return
    const timer = setInterval(() => void reloadRuns(currentKbId), 3000)
    return () => clearInterval(timer)
  }, [currentKbId, runningRun, reloadRuns])

  const handleCreateRun = async () => {
    if (!currentKbId || !selectedDatasetId) return
    setCreatingRun(true)
    try {
      const { data } = await createEvalRun(currentKbId, selectedDatasetId)
      setSelectedRunId(data.run_id)
      toast.success(t('eval.runStarted'))
      await reloadRuns(currentKbId)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setCreatingRun(false)
    }
  }

  const handleDeleteRun = async (runId: string) => {
    if (!currentKbId) return
    try {
      await deleteEvalRun(currentKbId, runId)
      toast.success(t('eval.runDeleted'))
      await reloadRuns(currentKbId)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  const runStatusIcon = (status: string) =>
    status === 'completed' ? (
      <CheckCircle2Icon className="text-emerald-500 size-4 shrink-0" aria-hidden="true" />
    ) : status === 'running' ? (
      <RefreshCwIcon className="text-amber-500 size-4 shrink-0 animate-spin" aria-hidden="true" />
    ) : (
      <XCircleIcon className="text-destructive size-4 shrink-0" aria-hidden="true" />
    )

  return (
    <div className="flex h-full gap-4 p-4">
      <div className="flex w-96 shrink-0 flex-col gap-4 overflow-y-auto">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('eval.kb')}</CardTitle>
          </CardHeader>
          <CardContent>
            <Select
              value={currentKbId ?? 'none'}
              onValueChange={(value) => setCurrentKbId(value === 'none' ? null : value)}
            >
              <SelectTrigger>
                <SelectValue placeholder={t('chat.selectKb')} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">{t('chat.noKb')}</SelectItem>
                {databases.map((db) => (
                  <SelectItem key={db.kb_id} value={db.kb_id}>
                    {db.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardContent>
        </Card>
        {currentKbId && (
          <>
            <DatasetForm kbId={currentKbId} onChanged={() => void reloadDatasets(currentKbId)} />
            <DatasetList
              datasets={datasets}
              selectedId={selectedDatasetId}
              onSelect={setSelectedDatasetId}
              onChanged={() => void reloadDatasets(currentKbId)}
            />
          </>
        )}
      </div>

      <div className="grid min-w-0 grow grid-cols-1 gap-4 overflow-auto xl:grid-cols-[1fr_1.4fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              {t('eval.runs')}
              <Badge variant="outline">{runs.length}</Badge>
            </CardTitle>
            <CardDescription>{t('eval.runsHint')}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3">
            <Button onClick={handleCreateRun} disabled={!selectedDatasetId || creatingRun} className="w-fit">
              <PlayIcon aria-hidden="true" />
              {creatingRun ? t('common.loading') : t('eval.runOnDataset')}
            </Button>
            {!selectedDatasetId && <p className="text-muted-foreground text-xs">{t('eval.runHint')}</p>}
            <ScrollArea className="max-h-[28rem]">
              <div className="grid gap-1.5">
                {runs.length === 0 && <p className="text-muted-foreground text-sm">{t('eval.noRuns')}</p>}
                {runs.map((run) => (
                  <button
                    key={run.run_id}
                    onClick={() => setSelectedRunId(run.run_id)}
                    className={cn(
                      'hover:bg-accent grid cursor-pointer gap-1 rounded-md border p-2 text-left',
                      selectedRunId === run.run_id && 'border-emerald-400 bg-emerald-400/10'
                    )}
                  >
                    <span className="flex items-center gap-2 text-sm font-medium">
                      {runStatusIcon(run.status)}
                      <span className="truncate">{run.name}</span>
                    </span>
                    <span className="text-muted-foreground text-xs">
                      {run.completed_items}/{run.total_items} · {t('eval.overall')}: {formatScore(run.overall_score)}
                    </span>
                    <span className="flex items-center gap-2">
                      <span className="text-muted-foreground truncate text-xs">{run.dataset_id}</span>
                      <span className="grow" />
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <span
                            role="button"
                            tabIndex={0}
                            className="hover:text-destructive shrink-0 cursor-pointer"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <Trash2Icon className="size-3.5" aria-hidden="true" />
                          </span>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>{t('eval.deleteRunTitle')}</AlertDialogTitle>
                            <AlertDialogDescription>{t('eval.deleteRunConfirm', { name: run.name })}</AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
                            <AlertDialogAction onClick={() => void handleDeleteRun(run.run_id)}>{t('common.confirm')}</AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </span>
                  </button>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        {currentKbId && selectedRunId ? (
          <RunDetail kbId={currentKbId} runId={selectedRunId} />
        ) : (
          <Card>
            <CardContent className="text-muted-foreground flex h-full items-center justify-center py-12 text-sm">
              {t('eval.noRunSelected')}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
