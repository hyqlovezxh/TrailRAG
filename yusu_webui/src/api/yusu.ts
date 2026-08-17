import { useSettingsStore } from '@/stores/settings'

// ---------------------------------------------------------------------------
// Types (mirror yusu_kb/api/routers responses)
// ---------------------------------------------------------------------------

export interface HealthInfo {
  status: string
  service: string
  version: string
}

export interface ModelsInfo {
  embedding: { status: string; message: string }
  chat: { status: string; message: string }
  rerank: { status: string; message: string }
}

export interface KbStats {
  file_count: number
  chunk_count: number
  token_count: number
}

export interface KnowledgeBase {
  kb_id: string
  name: string
  description: string
  created_at?: string | null
  updated_at?: string | null
  stats?: KbStats
  [key: string]: unknown
}

export type FileStatus =
  | 'uploaded'
  | 'parsing'
  | 'parsed'
  | 'error_parsing'
  | 'indexing'
  | 'indexed'
  | 'error_indexing'

export interface FileRecord {
  file_id: string
  kb_id: string
  filename: string
  original_filename?: string | null
  status: FileStatus
  size: number
  chunk_count: number
  token_count: number
  content_type?: string | null
  error?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface Chunk {
  id: string
  kb_id: string
  file_id: string
  content: string
  metadata?: Record<string, unknown>
  /** kb://{kb_id}/{file_id}?chunk={chunk_id} — the dedup key for citations */
  citation_source?: string
}

export interface QueryOutput {
  kb_id: string
  results: Chunk[]
}

export interface QueryParamsInfo {
  config: Record<string, unknown>
  effective: { options: Record<string, unknown> }
}

export type ChatMessage = { role: 'user' | 'assistant'; content: string }

export interface ToolEvent {
  type: 'tool'
  name: string
  args: Record<string, unknown>
  /** Tool result preview, truncated to 200 chars by the backend */
  summary: string
}

export type ChatEvent =
  | { type: 'sources'; chunks: Chunk[] }
  | ToolEvent
  | { type: 'delta'; content: string }
  | { type: 'reasoning'; content: string }
  | { type: 'done' }
  | { type: 'error'; message: string }

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

export interface GraphStatus {
  kb_id: string
  configured: boolean
  config: Record<string, unknown> | null
  locked: boolean
  total_chunks: number
  pending_chunks: number
  indexed_chunks: number
  entity_count: number
  relation_count: number
  build_task_status: string | null
  build_task_progress: number
}

export interface GraphSubgraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface GraphNode {
  entity_id: string
  type: string
  name: string
  normalized_name?: string
  label: string
  attributes?: string[]
  description?: string
  [key: string]: unknown
}

export interface GraphEdge {
  source_id: string
  target_id: string
  triple_id?: string
  text?: string
  type?: string
  file_ids?: string[]
  description?: string
  [key: string]: unknown
}

// ---------------------------------------------------------------------------
// Model providers
// ---------------------------------------------------------------------------

export interface ModelProvider {
  provider_id: string
  display_name: string
  provider_type: string
  default_protocol: string
  base_url: string
  embedding_base_url: string | null
  rerank_base_url: string | null
  models_endpoint: string
  embedding_models_endpoint: string | null
  rerank_models_endpoint: string | null
  api_key_env: string
  capabilities: string[]
  enabled_models: string[]
  headers_json: Record<string, unknown>
  extra_json: Record<string, unknown>
  is_enabled: boolean
  is_builtin: boolean
  credential_status: 'configured' | 'missing'
}

export interface ModelSpecInfo {
  spec: string
  model_id: string
  display_name: string
  dimension?: number | null
  batch_size?: number | null
}

export interface ModelProviderModels {
  provider_id: string
  provider_display_name: string
  models: ModelSpecInfo[]
}

export interface RemoteModel {
  id: string
  object?: string | null
  owned_by?: string | null
  type?: string
  display_name: string
  description?: string | null
  context_length?: number | null
  [key: string]: unknown
}

export interface ModelDefaults {
  default_chat_model_spec: string | null
  default_embedding_model_spec: string | null
  default_rerank_model_spec: string | null
}

export interface ModelStatusInfo {
  status: 'available' | 'unavailable' | 'error'
  message: string
  model_type?: string
  model_id?: string
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

export interface EvalDataset {
  id: string
  dataset_id: string
  name: string
  description: string
  kb_id: string
  item_count: number
  has_gold_chunks: boolean
  has_gold_answers: boolean
  build_metadata: Record<string, unknown>
  created_by: string | null
  created_at: string | null
  updated_at: string | null
}

export interface EvalDatasetItem {
  item_id: string
  item_index: number
  query: string
  gold_chunk_ids: string[]
  gold_answer: string | null
}

export interface EvalDatasetDetail extends EvalDataset {
  items: EvalDatasetItem[]
  pagination: Pagination
}

export interface EvalRun {
  run_id: string
  name: string
  dataset_id: string
  status: 'running' | 'completed' | 'failed' | string
  started_at: string | null
  completed_at: string | null
  total_items: number
  completed_items: number
  overall_score: number | null
  retrieval_config: Record<string, unknown>
  metrics: Record<string, unknown>
}

export interface EvalRunItem {
  query: string
  gold_chunk_ids: string[]
  gold_answer: string | null
  generated_answer: string | null
  retrieved_chunks: Record<string, unknown>[] | null
  metrics: Record<string, unknown>
}

export interface EvalRunDetail extends EvalRun {
  items: EvalRunItem[]
  pagination: Pagination
}

export interface Pagination {
  current_page: number
  page_size: number
  total?: number
  total_items?: number
  total_pages?: number
  has_next: boolean
  has_prev: boolean
}

// ---------------------------------------------------------------------------
// Client helpers
// ---------------------------------------------------------------------------

export class YusuApiError extends Error {
  status: number

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'YusuApiError'
    this.status = status
  }
}

const normalizeBaseUrl = (baseUrl: string): string =>
  (baseUrl || 'http://127.0.0.1:8920').replace(/\/+$/, '')

const resolveConfig = (baseUrl?: string, apiKey?: string | null): { baseUrl: string; apiKey: string | null } => {
  const settings = useSettingsStore.getState()
  return {
    baseUrl: normalizeBaseUrl(baseUrl ?? settings.baseUrl ?? ''),
    apiKey: apiKey !== undefined ? apiKey : settings.apiKey
  }
}

const buildHeaders = (config: { apiKey: string | null }, extra?: HeadersInit): Headers => {
  const headers = new Headers(extra)
  if (config.apiKey) {
    headers.set('Authorization', `Bearer ${config.apiKey}`)
  }
  return headers
}

const parseError = async (response: Response): Promise<never> => {
  let detail = `HTTP ${response.status}`
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') {
      detail = body.detail
    }
  } catch {
    // ignore non-JSON error bodies
  }
  throw new YusuApiError(response.status, detail)
}

const request = async <T>(path: string, init: RequestInit = {}, baseUrl?: string, apiKey?: string | null): Promise<T> => {
  const config = resolveConfig(baseUrl, apiKey)
  const response = await fetch(`${config.baseUrl}${path}`, {
    ...init,
    headers: buildHeaders(config, init.headers)
  })
  if (!response.ok) {
    await parseError(response)
  }
  return (await response.json()) as T
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const getHealth = (baseUrl?: string, apiKey?: string | null): Promise<HealthInfo> =>
  request('/api/health', {}, baseUrl, apiKey)

export const getModels = (): Promise<ModelsInfo> => request('/api/health/models')

export const listDatabases = (): Promise<{ databases: KnowledgeBase[] }> =>
  request('/api/knowledge/databases')

export const getDatabase = (kbId: string): Promise<KnowledgeBase & { files: FileRecord[] }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}`)

export const createDatabase = (name: string, description = ''): Promise<KnowledgeBase> =>
  request('/api/knowledge/databases', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description })
  })

export const deleteDatabase = (kbId: string): Promise<{ deleted: boolean }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}`, { method: 'DELETE' })

export const listFiles = (kbId: string): Promise<{ files: FileRecord[] }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/files`)

export const uploadFile = (kbId: string, file: File): Promise<FileRecord> => {
  const form = new FormData()
  form.append('file', file)
  return request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/files`, {
    method: 'POST',
    body: form
  })
}

export const parseFile = (kbId: string, fileId: string): Promise<{ status: string }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/files/${encodeURIComponent(fileId)}/parse`, {
    method: 'POST'
  })

export const indexFile = (kbId: string, fileId: string): Promise<{ status: string }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/files/${encodeURIComponent(fileId)}/index`, {
    method: 'POST'
  })

export const deleteFile = (kbId: string, fileId: string): Promise<{ deleted: boolean }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/files/${encodeURIComponent(fileId)}`, {
    method: 'DELETE'
  })

export const getQueryParams = (kbId: string): Promise<QueryParamsInfo> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/query-params`)

export const updateQueryParams = (kbId: string, params: Record<string, unknown>): Promise<{ options: Record<string, unknown> }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/query-params`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ params })
  })

export const queryKb = (kbId: string, query: string, params?: Record<string, unknown>): Promise<QueryOutput> =>
  request('/api/knowledge/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kb_id: kbId, query, params })
  })

export const refreshStats = (kbId: string): Promise<Record<string, number>> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/refresh-stats`, { method: 'POST' })

export const getSystemPrompt = (kbId: string): Promise<{ system_prompt: string }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/system-prompt`)

export const updateSystemPrompt = (kbId: string, systemPrompt: string): Promise<{ system_prompt: string }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/system-prompt`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system_prompt: systemPrompt })
  })

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

export const getGraphStatus = (kbId: string): Promise<GraphStatus> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/graph/status`)

export const buildGraph = (kbId: string, batchSize?: number): Promise<{ status: GraphStatus }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/graph/build`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ batch_size: batchSize })
  })

export const resetGraph = (
  kbId: string,
  clearExtractionResult = true,
  clearConfig = false
): Promise<{ success: boolean }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/graph/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clear_extraction_result: clearExtractionResult, clear_config: clearConfig })
  })

export const getGraphConfig = (kbId: string): Promise<{ config: Record<string, unknown> | null }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/graph/config`)

export const updateGraphConfig = (
  kbId: string,
  extractorType: string,
  extractorOptions: Record<string, unknown>
): Promise<{ success: boolean }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/graph/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ extractor_type: extractorType, extractor_options: extractorOptions })
  })

export const getGraphStats = (kbId: string): Promise<GraphStatus & { storage: Record<string, unknown> | null }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/graph/stats`)

export const getGraphLabels = (kbId: string): Promise<{ labels: Record<string, number> }> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/graph/labels`)

export const getGraphSubgraph = (
  kbId: string,
  entityIds: string[],
  maxDepth = 3,
  maxNodes = 5000
): Promise<GraphSubgraph> =>
  request(
    `/api/knowledge/databases/${encodeURIComponent(kbId)}/graph/subgraph?entity_ids=${encodeURIComponent(entityIds.join(','))}&max_depth=${maxDepth}&max_nodes=${maxNodes}`
  )

export const getGraphFull = (kbId: string, limit = 2000): Promise<GraphSubgraph> =>
  request(`/api/knowledge/databases/${encodeURIComponent(kbId)}/graph/full?limit=${limit}`)

// ---------------------------------------------------------------------------
// Model providers
// ---------------------------------------------------------------------------

export const listModelProviders = (): Promise<{ success: boolean; data: ModelProvider[] }> =>
  request('/api/system/model-providers')

export const getModelProvider = (providerId: string): Promise<{ success: boolean; data: ModelProvider }> =>
  request(`/api/system/model-providers/${encodeURIComponent(providerId)}`)

export const createModelProvider = (
  payload: Record<string, unknown>
): Promise<{ success: boolean; data: ModelProvider }> =>
  request('/api/system/model-providers', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })

export const updateModelProvider = (
  providerId: string,
  payload: Record<string, unknown>
): Promise<{ success: boolean; data: ModelProvider }> =>
  request(`/api/system/model-providers/${encodeURIComponent(providerId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })

export const deleteModelProvider = (providerId: string): Promise<{ success: boolean; data: { deleted: string } }> =>
  request(`/api/system/model-providers/${encodeURIComponent(providerId)}`, { method: 'DELETE' })

export const fetchRemoteModels = (providerId: string): Promise<{ success: boolean; data: RemoteModel[] }> =>
  request(`/api/system/model-providers/${encodeURIComponent(providerId)}/remote-models`)

export const refreshModelCache = (): Promise<{ success: boolean; model_count: number; message: string }> =>
  request('/api/system/model-providers/models/cache/refresh', { method: 'POST' })

export const getModelsV2 = (
  modelType: 'chat' | 'embedding' | 'rerank' = 'chat'
): Promise<{ success: boolean; data: Record<string, ModelProviderModels> }> =>
  request(`/api/system/model-providers/models/v2?model_type=${modelType}`)

export const getModelStatus = (spec: string): Promise<{ success: boolean; data: ModelStatusInfo }> =>
  request(`/api/system/model-providers/models/status?spec=${encodeURIComponent(spec)}`)

export const getDefaultModels = (): Promise<{ success: boolean; data: ModelDefaults }> =>
  request('/api/system/model-providers/defaults')

export const updateDefaultModels = (
  defaults: Partial<ModelDefaults>
): Promise<{ success: boolean; data: Partial<ModelDefaults>; message: string }> =>
  request('/api/system/model-providers/defaults', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(defaults)
  })

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

export const uploadEvalDataset = (
  kbId: string,
  file: File,
  name = '',
  description = ''
): Promise<{ success: boolean; data: EvalDataset }> => {
  const form = new FormData()
  form.append('file', file)
  form.append('kb_id', kbId)
  if (name) form.append('name', name)
  if (description) form.append('description', description)
  return request('/api/evaluation/datasets/upload', {
    method: 'POST',
    body: form
  })
}

export const listEvalDatasets = (kbId: string): Promise<{ success: boolean; data: EvalDataset[] }> =>
  request(`/api/evaluation/datasets?kb_id=${encodeURIComponent(kbId)}`)

export const getEvalDataset = (
  kbId: string,
  datasetId: string,
  page = 1,
  pageSize = 10
): Promise<{ success: boolean; data: EvalDatasetDetail }> =>
  request(
    `/api/evaluation/datasets/${encodeURIComponent(datasetId)}?kb_id=${encodeURIComponent(kbId)}&page=${page}&page_size=${pageSize}`
  )

export const downloadEvalDataset = async (
  datasetId: string,
  baseUrl?: string,
  apiKey?: string | null
): Promise<void> => {
  const config = resolveConfig(baseUrl, apiKey)
  const response = await fetch(`${config.baseUrl}/api/evaluation/datasets/${encodeURIComponent(datasetId)}/download`, {
    headers: buildHeaders(config)
  })
  if (!response.ok) {
    await parseError(response)
  }
  const blob = await response.blob()
  const disposition = response.headers.get('Content-Disposition') ?? ''
  const match = /filename="([^"]+)"/.exec(disposition)
  const filename = match ? match[1] : 'dataset.jsonl'
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export const generateEvalDataset = (payload: {
  kb_id: string
  name?: string
  description?: string
  count?: number
  neighbors_count?: number
  concurrency_count?: number
  llm_model_spec?: string
  generation_mode?: 'vector' | 'graph_enhanced'
  graph_expand_top_k?: number
}): Promise<{ success: boolean; data: EvalDataset }> =>
  request('/api/evaluation/datasets/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })

export const deleteEvalDataset = (datasetId: string): Promise<{ success: boolean; data: { deleted: string } }> =>
  request(`/api/evaluation/datasets/${encodeURIComponent(datasetId)}`, { method: 'DELETE' })

export const createEvalRun = (
  kbId: string,
  datasetId: string,
  name?: string,
  modelConfig?: Record<string, unknown>
): Promise<{ success: boolean; data: { run_id: string; status: string } }> =>
  request('/api/evaluation/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kb_id: kbId, dataset_id: datasetId, name, model_config: modelConfig ?? {} })
  })

export const listEvalRuns = (kbId: string): Promise<{ success: boolean; data: EvalRun[] }> =>
  request(`/api/evaluation/runs?kb_id=${encodeURIComponent(kbId)}`)

export const getEvalRun = (
  kbId: string,
  runId: string,
  page = 1,
  pageSize = 20,
  errorOnly = false
): Promise<{ success: boolean; data: EvalRunDetail }> =>
  request(
    `/api/evaluation/runs/${encodeURIComponent(runId)}?kb_id=${encodeURIComponent(kbId)}&page=${page}&page_size=${pageSize}&error_only=${errorOnly}`
  )

export const deleteEvalRun = (kbId: string, runId: string): Promise<{ success: boolean; data: { deleted: string } }> =>
  request(`/api/evaluation/runs/${encodeURIComponent(runId)}?kb_id=${encodeURIComponent(kbId)}`, {
    method: 'DELETE'
  })

// ---------------------------------------------------------------------------
// SSE chat streaming
// ---------------------------------------------------------------------------

const parseSseLines = (chunk: string, buffer: string, onEvent: (event: ChatEvent) => void): string => {
  const text = buffer + chunk
  const lines = text.split('\n')
  const rest = lines.pop() ?? ''
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed.startsWith('data:')) continue
    const payload = trimmed.slice(5).trim()
    if (!payload) continue
    try {
      const event = JSON.parse(payload) as ChatEvent
      if (event && typeof event.type === 'string') {
        onEvent(event)
      }
    } catch {
      // ignore malformed SSE frames
    }
  }
  return rest
}

export interface ChatStreamOptions {
  kbId: string
  messages: ChatMessage[]
  queryParams?: Record<string, unknown>
  toolsEnabled?: boolean
  systemPrompt?: string
  onEvent: (event: ChatEvent) => void
  signal?: AbortSignal
}

export const chatStream = async ({
  kbId,
  messages,
  queryParams,
  toolsEnabled = true,
  systemPrompt,
  onEvent,
  signal
}: ChatStreamOptions): Promise<void> => {
  const config = resolveConfig()
  const response = await fetch(`${config.baseUrl}/api/chat/stream`, {
    method: 'POST',
    headers: buildHeaders(config, { 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      kb_id: kbId,
      messages,
      query_params: queryParams,
      tools_enabled: toolsEnabled,
      system_prompt: systemPrompt ?? ''
    }),
    signal
  })
  if (!response.ok) {
    await parseError(response)
  }
  if (!response.body) {
    throw new YusuApiError(500, '响应无流式内容')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer = parseSseLines(decoder.decode(value, { stream: true }), buffer, onEvent)
    }
    buffer = parseSseLines('', buffer, onEvent)
  } finally {
    reader.releaseLock()
  }
}