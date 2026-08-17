import { afterEach, beforeEach, describe, expect, test, mock } from 'bun:test'
import {
  chatStream,
  createDatabase,
  createEvalRun,
  getEvalRun,
  getGraphStatus,
  getHealth,
  getSystemPrompt,
  listDatabases,
  listEvalDatasets,
  listModelProviders,
  updateDefaultModels,
  updateSystemPrompt,
  uploadFile,
  type ChatEvent,
  type ToolEvent
} from './yusu'
import { useSettingsStore } from '@/stores/settings'

const encoder = new TextEncoder()

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' }
  })

const sseResponse = (frames: string[]): Response => {
  const stream = new ReadableStream({
    start(controller) {
      for (const frame of frames) {
        controller.enqueue(encoder.encode(frame))
      }
      controller.close()
    }
  })
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' }
  })
}

describe('yusu api', () => {
  let fetchMock: ReturnType<typeof mock>

  beforeEach(() => {
    useSettingsStore.getState().setBaseUrl('http://127.0.0.1:8920')
    useSettingsStore.getState().setApiKey(null)
    fetchMock = mock(() => Promise.resolve(new Response('{}', { status: 200 })))
    globalThis.fetch = fetchMock as typeof fetch
  })

  afterEach(() => {
    fetchMock.mockClear()
  })

  test('listDatabases hits the YUSU endpoint and parses JSON', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(jsonResponse({ databases: [{ kb_id: 'kb1', name: '测试库' }] }))
    )

    const { databases } = await listDatabases()
    expect(databases).toEqual([{ kb_id: 'kb1', name: '测试库' }])

    const [input, init] = fetchMock.mock.calls[0]
    expect(String(input)).toBe('http://127.0.0.1:8920/api/knowledge/databases')
    expect(init?.method ?? 'GET').toBe('GET')
  })

  test('sends Bearer header when api key is configured', async () => {
    useSettingsStore.getState().setApiKey('secret')
    await listDatabases()

    const [, init] = fetchMock.mock.calls[0]
    const headers = init?.headers as Headers
    expect(headers.get('Authorization')).toBe('Bearer secret')
  })

  test('createDatabase posts JSON payload', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(jsonResponse({ kb_id: 'kb2', name: '新库', description: 'd' }))
    )

    const kb = await createDatabase('新库', 'd')
    expect(kb.kb_id).toBe('kb2')

    const [, init] = fetchMock.mock.calls[0]
    expect(init?.method).toBe('POST')
    expect(init?.body).toBe(JSON.stringify({ name: '新库', description: 'd' }))
  })

  test('uploadFile sends FormData', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(jsonResponse({ file_id: 'f1', kb_id: 'kb1', filename: 'a.md', status: 'uploaded' }))
    )

    const file = new File(['# hello'], 'a.md', { type: 'text/markdown' })
    const record = await uploadFile('kb1', file)
    expect(record.file_id).toBe('f1')

    const [input, init] = fetchMock.mock.calls[0]
    expect(String(input)).toContain('/api/knowledge/databases/kb1/files')
    expect(init?.method).toBe('POST')
    expect(init?.body).toBeInstanceOf(FormData)
  })

  test('maps 404 error responses to YusuApiError', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(jsonResponse({ detail: '知识库不存在' }, 404))
    )

    await expect(listDatabases()).rejects.toMatchObject({
      name: 'YusuApiError',
      status: 404,
      message: '知识库不存在'
    })
  })

  test('getHealth supports explicit baseUrl/apiKey for the connect screen', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(jsonResponse({ status: 'ok', service: 'yusu-kb', version: '0.1.0' }))
    )

    const health = await getHealth('http://example.com:9000/', 'k')
    expect(health.version).toBe('0.1.0')

    const [input, init] = fetchMock.mock.calls[0]
    expect(String(input)).toBe('http://example.com:9000/api/health')
    expect((init?.headers as Headers).get('Authorization')).toBe('Bearer k')
  })

  test('chatStream parses SSE events in order', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        sseResponse([
          'data: {"type":"sources","chunks":[{"id":"c1","kb_id":"kb1","file_id":"f1","content":"苹果是红色的"}]}\n\n',
          'data: {"type":"delta","content":"苹果"}\n\n',
          'data: {"type":"delta","content":"是红色的"}\n\n',
          'data: {"type":"done"}\n\n'
        ])
      )
    )

    const events: ChatEvent[] = []
    await chatStream({
      kbId: 'kb1',
      messages: [{ role: 'user', content: '苹果是什么颜色' }],
      onEvent: (event) => events.push(event)
    })

    expect(events.map((e) => e.type)).toEqual(['sources', 'delta', 'delta', 'done'])
    expect(events[0]).toMatchObject({ type: 'sources', chunks: [{ id: 'c1' }] })
    expect(events[1]).toMatchObject({ type: 'delta', content: '苹果' })
  })

  test('chatStream handles frames split across chunks and forwards errors', async () => {
    const payload = '{"type":"delta","content":"片段"}'
    const first = `data: ${payload.slice(0, 10)}`
    const rest = `${payload.slice(10)}\n\n`
    fetchMock.mockImplementation(() =>
      Promise.resolve(sseResponse([first, rest, 'data: {"type":"error","message":"boom"}\n\n']))
    )

    const events: ChatEvent[] = []
    await chatStream({ kbId: 'kb1', messages: [{ role: 'user', content: 'q' }], onEvent: (e) => events.push(e) })

    expect(events.map((e) => e.type)).toEqual(['delta', 'error'])
    expect(events[1]).toMatchObject({ type: 'error', message: 'boom' })
  })

  test('chatStream throws YusuApiError on non-200', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(jsonResponse({ detail: '知识库不存在' }, 404))
    )

    await expect(
      chatStream({ kbId: 'missing', messages: [{ role: 'user', content: 'q' }], onEvent: () => {} })
    ).rejects.toMatchObject({ status: 404, message: '知识库不存在' })
  })

  test('chatStream sends toolsEnabled/systemPrompt in the request body', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(sseResponse(['data: {"type":"done"}\n\n']))
    )

    await chatStream({
      kbId: 'kb1',
      messages: [{ role: 'user', content: 'q' }],
      toolsEnabled: false,
      systemPrompt: '请用粤语回答。',
      onEvent: () => {}
    })

    const [, init] = fetchMock.mock.calls[0]
    const body = JSON.parse(String(init?.body))
    expect(body).toMatchObject({
      kb_id: 'kb1',
      tools_enabled: false,
      system_prompt: '请用粤语回答。'
    })
  })

  test('chatStream parses tool events', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        sseResponse([
          'data: {"type":"sources","chunks":[{"id":"c1","kb_id":"kb1","file_id":"f1","content":"x","citation_source":"kb://kb1/f1?chunk=c1"}]}\n\n',
          'data: {"type":"tool","name":"query_kb","args":{"kb_id":"kb1","query_text":"苹果"},"summary":"检索到 2 条结果"}\n\n',
          'data: {"type":"delta","content":"回答"}\n\n',
          'data: {"type":"done"}\n\n'
        ])
      )
    )

    const events: ChatEvent[] = []
    await chatStream({ kbId: 'kb1', messages: [{ role: 'user', content: 'q' }], onEvent: (e) => events.push(e) })

    expect(events.map((e) => e.type)).toEqual(['sources', 'tool', 'delta', 'done'])
    const tool = events[1] as ToolEvent
    expect(tool.name).toBe('query_kb')
    expect(tool.args).toEqual({ kb_id: 'kb1', query_text: '苹果' })
    expect(tool.summary).toBe('检索到 2 条结果')
    const sources = events[0] as Extract<ChatEvent, { type: 'sources' }>
    expect(sources.chunks[0].citation_source).toBe('kb://kb1/f1?chunk=c1')
  })

  test('getGraphStatus hits the graph endpoint', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        jsonResponse({
          kb_id: 'kb1',
          configured: true,
          config: { extractor_type: 'llm' },
          locked: false,
          total_chunks: 10,
          pending_chunks: 2,
          indexed_chunks: 8,
          entity_count: 5,
          relation_count: 4,
          build_task_status: 'completed',
          build_task_progress: 1
        })
      )
    )

    const status = await getGraphStatus('kb1')
    expect(status.entity_count).toBe(5)
    expect(status.build_task_status).toBe('completed')

    const [input] = fetchMock.mock.calls[0]
    expect(String(input)).toContain('/api/knowledge/databases/kb1/graph/status')
  })

  test('system prompt get/update roundtrip', async () => {
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse({ system_prompt: '' })))
    const initial = await getSystemPrompt('kb1')
    expect(initial.system_prompt).toBe('')

    fetchMock.mockImplementation(() =>
      Promise.resolve(jsonResponse({ system_prompt: '你是资深笔录分析师。' }))
    )
    const updated = await updateSystemPrompt('kb1', '你是资深笔录分析师。')
    expect(updated.system_prompt).toBe('你是资深笔录分析师。')

    const [input, init] = fetchMock.mock.calls[1]
    expect(String(input)).toContain('/system-prompt')
    expect(init?.method).toBe('PUT')
    expect(JSON.parse(String(init?.body))).toEqual({ system_prompt: '你是资深笔录分析师。' })
  })

  test('listModelProviders parses the envelope', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        jsonResponse({
          success: true,
          data: [
            {
              provider_id: 'siliconflow-cn',
              display_name: '硅基流动（国内）',
              provider_type: 'openai',
              default_protocol: 'https',
              base_url: 'https://api.siliconflow.cn/v1',
              embedding_base_url: null,
              rerank_base_url: null,
              models_endpoint: '/models',
              embedding_models_endpoint: '/embeddings/models',
              rerank_models_endpoint: null,
              api_key_env: 'SILICONFLOW_API_KEY',
              capabilities: ['chat', 'embedding'],
              enabled_models: [],
              headers_json: {},
              extra_json: {},
              is_enabled: true,
              is_builtin: true,
              credential_status: 'configured'
            }
          ]
        })
      )
    )

    const { data } = await listModelProviders()
    expect(data[0].provider_id).toBe('siliconflow-cn')
    expect(data[0].credential_status).toBe('configured')

    const [input] = fetchMock.mock.calls[0]
    expect(String(input)).toBe('http://127.0.0.1:8920/api/system/model-providers')
  })

  test('updateDefaultModels PUTs the three spec keys', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        jsonResponse({
          success: true,
          message: '默认模型配置已保存',
          data: { default_chat_model_spec: 'sensenova:sensenova-6.8-flash-lite' }
        })
      )
    )

    const result = await updateDefaultModels({ default_chat_model_spec: 'sensenova:sensenova-6.8-flash-lite' })
    expect(result.success).toBe(true)

    const [input, init] = fetchMock.mock.calls[0]
    expect(String(input)).toContain('/api/system/model-providers/defaults')
    expect(init?.method).toBe('PUT')
    expect(JSON.parse(String(init?.body))).toEqual({
      default_chat_model_spec: 'sensenova:sensenova-6.8-flash-lite'
    })
  })

  test('eval endpoints build query strings and envelopes', async () => {
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse({ success: true, data: [] })))
    const listed = await listEvalDatasets('kb1')
    expect(listed.success).toBe(true)
    const [listInput] = fetchMock.mock.calls[0]
    expect(String(listInput)).toBe(
      'http://127.0.0.1:8920/api/evaluation/datasets?kb_id=kb1'
    )

    fetchMock.mockImplementation(() =>
      Promise.resolve(
        jsonResponse({
          success: true,
          data: {
            run_id: 'run_1234abcd',
            name: 'Run 1',
            dataset_id: 'ds_1',
            status: 'completed',
            started_at: null,
            completed_at: null,
            total_items: 1,
            completed_items: 1,
            overall_score: 0.8,
            retrieval_config: {},
            metrics: { 'recall@5': 1.0, judge_status: 'available' },
            items: [],
            pagination: { current_page: 1, page_size: 20, total: 0, has_next: false, has_prev: false }
          }
        })
      )
    )
    const run = await getEvalRun('kb1', 'run_1234abcd', 1, 20, true)
    expect(run.data.overall_score).toBe(0.8)
    const [runInput] = fetchMock.mock.calls[1]
    expect(String(runInput)).toContain('/api/evaluation/runs/run_1234abcd')
    expect(String(runInput)).toContain('error_only=true')
  })

  test('createEvalRun posts model_config', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        jsonResponse({ success: true, data: { run_id: 'run_1234abcd', status: 'running' } })
      )
    )

    const result = await createEvalRun('kb1', 'ds_1', '我的评估', { llm_model_spec: 'sensenova:model' })
    expect(result.data.status).toBe('running')

    const [, init] = fetchMock.mock.calls[0]
    expect(init?.method).toBe('POST')
    const body = JSON.parse(String(init?.body))
    expect(body).toMatchObject({ kb_id: 'kb1', dataset_id: 'ds_1', name: '我的评估' })
    expect(body.model_config).toEqual({ llm_model_spec: 'sensenova:model' })
  })
})