import Graph from 'graphology'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import {
  FullScreenControl,
  SigmaContainer,
  useRegisterEvents,
  useSigma,
  ZoomControl,
  ControlsContainer
} from '@react-sigma/core'
import { XIcon } from 'lucide-react'
import { AlertCircleIcon, Loader2Icon, NetworkIcon, PlayIcon, RotateCcwIcon, SearchIcon } from 'lucide-react'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Badge from '@/components/ui/Badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/AlertDialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { ScrollArea } from '@/components/ui/ScrollArea'
import {
  buildGraph,
  getGraphFull,
  getGraphStatus,
  getGraphSubgraph,
  listDatabases,
  resetGraph,
  type GraphEdge,
  type GraphNode,
  type GraphStatus,
  type KnowledgeBase
} from '@/api/yusu'
import { useSettingsStore } from '@/stores/settings'
import { colorForType, graphToGraphology } from '@/lib/graphToGraphology'
import { cn } from '@/lib/utils'

const mergeNodes = (prev: GraphNode[], next: GraphNode[]): GraphNode[] => {
  const byId = new Map(prev.map((n) => [n.entity_id, n]))
  for (const node of next) {
    if (!byId.has(node.entity_id)) byId.set(node.entity_id, node)
  }
  return [...byId.values()]
}

const mergeEdges = (prev: GraphEdge[], next: GraphEdge[]): GraphEdge[] => {
  const key = (e: GraphEdge) => `${e.source_id}\u0000${e.target_id}\u0000${e.type ?? ''}`
  const byKey = new Map(prev.map((e) => [key(e), e]))
  for (const edge of next) {
    if (!byKey.has(key(edge))) byKey.set(key(edge), edge)
  }
  return [...byKey.values()]
}

type GraphSelection = { kind: 'node'; id: string } | { kind: 'edge'; id: string } | null

function GraphEvents({
  onExpand,
  onSelect
}: {
  onExpand: (entityId: string) => void
  onSelect: (selection: GraphSelection) => void
}) {
  const registerEvents = useRegisterEvents()
  const sigma = useSigma()
  const [draggedNode, setDraggedNode] = useState<string | null>(null)

  useEffect(() => {
    // Reactive node dragging (mirrors the YUSU webui implementation):
    // `downNode` starts a drag, `mousemovebody` follows the pointer on screen,
    // `mouseup` ends it. During a drag we also freeze sigma's camera via a
    // custom bounding box so the graph doesn't auto-pan while moving a node.
    registerEvents({
      downNode: (e) => {
        setDraggedNode(e.node)
        sigma.getGraph().setNodeAttribute(e.node, 'highlighted', true)
      },
      mousemovebody: (e) => {
        if (!draggedNode) return
        const pos = sigma.viewportToGraph(e)
        sigma.getGraph().setNodeAttribute(draggedNode, 'x', pos.x)
        sigma.getGraph().setNodeAttribute(draggedNode, 'y', pos.y)
        e.preventSigmaDefault()
        e.original.preventDefault()
        e.original.stopPropagation()
      },
      mouseup: () => {
        if (draggedNode) {
          setDraggedNode(null)
          sigma.getGraph().removeNodeAttribute(draggedNode, 'highlighted')
        }
      },
      mousedown: (e) => {
        const mouseEvent = e.original as MouseEvent
        if (mouseEvent.buttons !== 0 && !sigma.getCustomBBox()) {
          sigma.setCustomBBox(sigma.getBBox())
        }
      },
      // Single click selects a node/edge to show a description panel (the
      // description attribute comes from the real graph retrieval payload).
      clickNode: (e) => onSelect({ kind: 'node', id: e.node }),
      clickEdge: (e) => onSelect({ kind: 'edge', id: e.edge }),
      // Double-click still expands the neighborhood via graph retrieval.
      doubleClickNode: (e) => onExpand(e.node)
    })
  }, [registerEvents, sigma, draggedNode, onExpand, onSelect])

  return null
}

function GraphSelectionPanel({
  selection,
  graph,
  nodes,
  edges,
  onClose
}: {
  selection: Exclude<GraphSelection, null>
  graph: Graph
  nodes: GraphNode[]
  edges: GraphEdge[]
  onClose: () => void
}) {
  const { t } = useTranslation()

  let title: string
  let subtitle: string
  let description: string

  if (selection.kind === 'node') {
    const node = nodes.find((n) => n.entity_id === selection.id)
    title = node?.name || node?.label || selection.id
    subtitle = node?.type || node?.label || t('graph.node', '节点')
    description = node?.description ?? ''
  } else {
    let src = ''
    let tgt = ''
    if (graph.hasEdge(selection.id)) {
      const attr = graph.getEdgeAttributes(selection.id)
      src = (attr.source_id as string) ?? ''
      tgt = (attr.target_id as string) ?? ''
    }
    const edge = edges.find((e) => e.source_id === src && e.target_id === tgt)
    title = edge?.type || t('graph.relation', '关系')
    subtitle = edge ? `${src} → ${tgt}` : t('graph.edge', '连线')
    description = edge?.description ?? ''
  }

  return (
    <div className="bg-card top-2 right-2 z-10 w-72 rounded-lg border shadow-lg">
      <div className="flex items-start justify-between border-b px-3 py-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{title}</div>
          {subtitle && (
            <div className="text-muted-foreground truncate text-xs">{subtitle}</div>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-muted-foreground hover:text-foreground -m-1 shrink-0 rounded p-1"
          aria-label={t('common.close', '关闭')}
        >
          <XIcon className="size-4" aria-hidden="true" />
        </button>
      </div>
      <div className="max-h-56 overflow-y-auto p-3 text-sm">
        {description ? (
          <p className="text-foreground/90">{description}</p>
        ) : (
          <p className="text-muted-foreground">{t('graph.noDescription', '暂无描述')}</p>
        )}
      </div>
    </div>
  )
}

function StatusCard({ status }: { status: GraphStatus | null }) {
  const { t } = useTranslation()
  if (!status) {
    return (
      <Card>
        <CardContent className="text-muted-foreground py-6 text-sm">
          {t('graph.noStatus')}
        </CardContent>
      </Card>
    )
  }
  const building = status.build_task_status === 'running' || status.build_task_status === 'pending'
  const progress = Math.round(
    status.total_chunks > 0 ? (status.indexed_chunks / status.total_chunks) * 100 : 0
  )
  const label = (value: string | null) => (value ? t(`graph.${value}`) : t('graph.idle'))

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
          <div>
            <span className="text-muted-foreground">{t('graph.configured')}: </span>
            {status.configured ? (
              <Badge variant="default">{t('graph.yes')}</Badge>
            ) : (
              <Badge variant="outline">{t('graph.no')}</Badge>
            )}
          </div>
          <div>
            <span className="text-muted-foreground">{t('graph.buildTask')}: </span>
            <Badge variant={building ? 'secondary' : 'outline'}>{label(status.build_task_status)}</Badge>
          </div>
        </div>
        <div>
          <div className="mb-1 flex justify-between text-xs">
            <span className="text-muted-foreground">{t('graph.buildProgress')}</span>
            <span>{progress}%</span>
          </div>
          <div className="bg-muted h-2 overflow-hidden rounded-full">
            <div
              className={cn('h-full rounded-full transition-all', building ? 'bg-emerald-400' : 'bg-emerald-400/40')}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default function GraphView() {
  const { t } = useTranslation()
  const currentKbId = useSettingsStore.use.currentKbId()
  const setCurrentKbId = useSettingsStore.use.setCurrentKbId()

  const [databases, setDatabases] = useState<KnowledgeBase[]>([])
  const [status, setStatus] = useState<GraphStatus | null>(null)
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [seed, setSeed] = useState('')
  const [loadingGraph, setLoadingGraph] = useState(false)
  const [building, setBuilding] = useState(false)
  const [expandingId, setExpandingId] = useState<string | null>(null)
  const [selection, setSelection] = useState<GraphSelection>(null)

  // Stable graphology instance: react-sigma only reads the `graph` prop once at
  // mount, so we must keep the same instance and mutate it in place to refresh
  // the canvas (replacing the reference never re-renders Sigma).
  const graphRef = useRef<Graph | null>(null)
  if (graphRef.current === null) {
    graphRef.current = new Graph({ multi: true, type: 'directed' })
  }
  const loadedEntityCount = useRef<number>(-1)
  const loadingFullRef = useRef(false)

  // Declared BEFORE the polling useEffect below: that effect lists
  // `loadFullGraph` in its dependency array, and a `const` referenced in a
  // deps array is evaluated at hook-call time — before its declaration would
  // otherwise run — which throws a temporal-dead-zone ReferenceError
  // ("Cannot access 'loadFullGraph' before initialization") and blanks the
  // whole knowledge-graph panel.
  const loadFullGraph = useCallback(async (kbId: string, expectedCount?: number) => {
    if (loadingFullRef.current) return
    loadingFullRef.current = true
    setLoadingGraph(true)
    try {
      const full = await getGraphFull(kbId)
      setNodes(full.nodes)
      setEdges(full.edges)
      setSelection(null)
      loadedEntityCount.current = expectedCount ?? full.nodes.length
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingGraph(false)
      loadingFullRef.current = false
    }
  }, [])

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
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset status when KB changes
      setStatus(null)
      return
    }
    let disposed = false
    const poll = async () => {
      try {
        const next = await getGraphStatus(currentKbId)
        if (disposed) return
        setStatus(next)
        // Auto-load the built graph once it exists (and after a rebuild that
        // changes the entity count), so the canvas is never blank by default.
        if (next.entity_count > 0 && loadedEntityCount.current !== next.entity_count) {
          void loadFullGraph(currentKbId, next.entity_count)
        }
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
  }, [currentKbId, loadFullGraph])

  // Reset the canvas when the selected KB changes.
  useEffect(() => {
    graphRef.current?.clear()
    setNodes([])
    setEdges([])
    setSelection(null)
    loadedEntityCount.current = -1
  }, [currentKbId])

  // Mirror the latest nodes/edges into the stable graph instance.
  useEffect(() => {
    const g = graphRef.current
    if (!g) return
    g.clear()
    graphToGraphology({ nodes, edges }, g)
  }, [nodes, edges])

  const graph = graphRef.current!

  const loadSeed = async (raw: string) => {
    const kbId = currentKbId
    if (!kbId || !raw.trim()) return
    setLoadingGraph(true)
    try {
      const subgraph = await getGraphSubgraph(kbId, [raw.trim()], 3)
      if (subgraph.nodes.length === 0) {
        toast.error(t('graph.noMatch', { seed: raw.trim() }))
        return
      }
      setNodes(subgraph.nodes)
      setEdges(subgraph.edges)
      loadedEntityCount.current = -1
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingGraph(false)
    }
  }

  const handleExpand = useCallback(
    async (entityId: string) => {
      const kbId = currentKbId
      if (!kbId) return
      setExpandingId(entityId)
      try {
        const subgraph = await getGraphSubgraph(kbId, [entityId], 3)
        setNodes((prev) => mergeNodes(prev, subgraph.nodes))
        setEdges((prev) => mergeEdges(prev, subgraph.edges))
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e))
      } finally {
        setExpandingId(null)
      }
    },
    [currentKbId]
  )

  const handleBuild = async () => {
    if (!currentKbId) return
    setBuilding(true)
    try {
      const result = await buildGraph(currentKbId)
      toast.success(t('graph.buildStarted'))
      setStatus(result.status)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setBuilding(false)
    }
  }

  const handleReset = async () => {
    if (!currentKbId) return
    try {
      await resetGraph(currentKbId)
      toast.success(t('graph.resetSuccess'))
      setNodes([])
      setEdges([])
      setSelection(null)
      const next = await getGraphStatus(currentKbId)
      setStatus(next)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  const categoryColors = useMemo(() => {
    const map = new Map<string, string>()
    for (const node of nodes) {
      const category = node.label || node.type || 'Entity'
      if (!map.has(category)) map.set(category, colorForType(category))
    }
    return [...map.entries()]
  }, [nodes])

  return (
    <div className="flex h-full gap-4 p-4">
      <div className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('graph.kb')}</CardTitle>
            <CardDescription>{t('graph.kbHint')}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3">
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
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={handleBuild} disabled={!currentKbId || building || status?.build_task_status === 'running'}>
                <PlayIcon aria-hidden="true" />
                {building ? t('common.loading') : t('graph.build')}
              </Button>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button size="sm" variant="outline" disabled={!currentKbId}>
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

        <StatusCard status={status} />

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('graph.explore')}</CardTitle>
            <CardDescription>{t('graph.exploreHint')}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3">
            <div className="flex items-center gap-2">
              <Input
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && void loadSeed(seed)}
                placeholder={t('graph.seedPlaceholder')}
              />
              <Button size="icon" variant="outline" onClick={() => void loadSeed(seed)} disabled={!currentKbId || loadingGraph}>
                {loadingGraph ? <Loader2Icon className="animate-spin" aria-hidden="true" /> : <SearchIcon aria-hidden="true" />}
              </Button>
            </div>
            <p className="text-muted-foreground text-xs">{t('graph.clickHint')}</p>
            {categoryColors.length > 0 && (
              <ScrollArea className="max-h-40">
                <div className="grid gap-1">
                  {categoryColors.map(([category, color]) => (
                    <div key={category} className="flex items-center gap-2 text-xs">
                      <span className="size-2.5 shrink-0 rounded-full" style={{ backgroundColor: color }} />
                      <span className="truncate">{category}</span>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="flex min-w-0 grow flex-col">
        <CardContent className="flex min-h-0 flex-1 flex-col p-0">
          {currentKbId ? (
            <div className="relative flex min-h-0 flex-1 flex-col">
              <SigmaContainer
                graph={graph}
                settings={{
                  renderEdgeLabels: true,
                  allowInvalidContainer: true,
                  // Allocate the edge picking buffer so edges are hover/click
                  // selectable (their descriptions are shown in the panel).
                  enableEdgeEvents: true
                }}
                className="min-h-0 flex-1"
              >
                <GraphEvents
                  onExpand={(id) => void handleExpand(id)}
                  onSelect={setSelection}
                />
                <ControlsContainer position="bottom-right">
                  <ZoomControl />
                  <FullScreenControl />
                </ControlsContainer>
              </SigmaContainer>
              {selection && (
                <GraphSelectionPanel
                  selection={selection}
                  graph={graph}
                  nodes={nodes}
                  edges={edges}
                  onClose={() => setSelection(null)}
                />
              )}
              {expandingId && (
                <div className="bg-background/80 absolute top-2 left-2 flex items-center gap-2 rounded-md border px-2 py-1 text-xs">
                  <Loader2Icon className="size-3 animate-spin" aria-hidden="true" />
                  {t('graph.expanding')} {expandingId}
                </div>
              )}
              {nodes.length === 0 && !loadingGraph && (
                <div className="text-muted-foreground absolute inset-0 flex items-center justify-center text-sm">
                  <span className="flex items-center gap-2">
                    <AlertCircleIcon className="size-4" aria-hidden="true" />
                    {t('graph.empty')}
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="text-muted-foreground flex min-h-0 flex-1 items-center justify-center text-sm">
              {t('graph.selectKbHint')}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
