import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeSanitize from 'rehype-sanitize'
import { BrainIcon, MessageSquareTextIcon, SendIcon, SquareIcon, Trash2Icon, Wand2Icon, WrenchIcon } from 'lucide-react'
import Button from '@/components/ui/Button'
import Textarea from '@/components/ui/Textarea'
import { Card } from '@/components/ui/Card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import {
  chatStream,
  getSystemPrompt,
  listDatabases,
  updateSystemPrompt,
  type Chunk,
  type KnowledgeBase,
  type ToolEvent
} from '@/api/yusu'
import { useSettingsStore, type ChatTurn } from '@/stores/settings'
import { chatMarkdownSanitizeSchema } from '@/utils/markdownSanitizeSchema'
import { remarkFootnotes } from '@/utils/remarkFootnotes'
import { cn } from '@/lib/utils'
import { dedupeSources, parseCites } from '@/lib/parseCites'
import 'katex/dist/katex.min.css'

function SourcesPanel({ sources, cited }: { sources: Chunk[]; cited: number[] }) {
  const { t } = useTranslation()
  const [expanded, setExpanded] = useState(false)

  if (sources.length === 0) return null

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="text-muted-foreground hover:text-foreground text-xs font-medium cursor-pointer"
      >
        {expanded ? '▾' : '▸'} {t('chat.sources')} ({sources.length})
      </button>
      {expanded && (
        <div className="mt-1 grid max-h-48 gap-2 overflow-auto rounded-md border p-2">
          {sources.map((source, index) => {
            const number = index + 1
            const isCited = cited.includes(number)
            return (
              <div
                key={source.citation_source || source.id}
                className={cn(
                  'text-muted-foreground text-xs',
                  isCited && 'border-emerald-400/40 bg-emerald-400/5 border-l-2 pl-2'
                )}
              >
                <p className="mb-0.5 font-medium text-foreground/80">
                  [{number}] {String(source.file_id ?? '')}
                </p>
                <p className="line-clamp-3 whitespace-pre-wrap">{source.content}</p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function ToolCallBadge({ call }: { call: ToolEvent }) {
  const argsRaw = JSON.stringify(call.args ?? {}) || ''
  const argsPreview = argsRaw.length > 80 ? `${argsRaw.slice(0, 80)}…` : argsRaw
  return (
    <details className="mb-2 rounded-md border border-blue-400/30 bg-blue-400/5">
      <summary className="flex cursor-pointer items-center gap-1.5 px-2 py-1 text-xs">
        <WrenchIcon className="size-3 shrink-0 text-blue-400" aria-hidden="true" />
        <span className="font-medium text-foreground/90">{call.name}</span>
        {argsPreview && <span className="text-muted-foreground truncate font-mono">{argsPreview}</span>}
      </summary>
      {call.summary && (
        <p className="text-muted-foreground max-h-32 overflow-auto border-t px-2 py-1 text-xs whitespace-pre-wrap">
          {call.summary}
        </p>
      )}
    </details>
  )
}

function MessageBubble({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === 'user'
  const sources = turn.sources ?? []
  const { text, cited } = parseCites(turn.content ?? '', sources)
  return (
    <div className={cn('flex w-full', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[85%] rounded-lg border px-3 py-2 text-sm',
          isUser ? 'bg-emerald-400/10 border-emerald-400/30' : 'bg-background'
        )}
      >
        {!isUser && turn.toolCalls && turn.toolCalls.length > 0 && (
          <div className="mb-2 grid gap-1">
            {turn.toolCalls.map((call, index) => (
              <ToolCallBadge key={`${call.name}-${index}`} call={call} />
            ))}
          </div>
        )}
        {!isUser && turn.reasoning && (
          <details className="mb-2">
            <summary className="flex cursor-pointer items-center gap-1 text-xs text-muted-foreground">
              <BrainIcon className="size-3" aria-hidden="true" />
              {turn.reasoning}
            </summary>
          </details>
        )}
        <div className="prose prose-sm max-w-none dark:prose-invert [&_mark]:bg-amber-200 [&_mark]:text-inherit [&_u]:no-underline [&_u]:border-b [&_u]:border-dashed">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath, remarkFootnotes]}
            rehypePlugins={[[rehypeSanitize, chatMarkdownSanitizeSchema], rehypeKatex]}
            skipHtml={false}
          >
            {text}
          </ReactMarkdown>
        </div>
        {!isUser && sources.length > 0 && <SourcesPanel sources={sources} cited={cited} />}
        {!isUser && turn.error && (
          <p className="text-destructive mt-1 text-xs">{turn.error}</p>
        )}
      </div>
    </div>
  )
}

function KnowledgeBaseSelector({
  databases,
  onChanged
}: {
  databases: KnowledgeBase[]
  onChanged: (kbId: string | null) => void
}) {
  const { t } = useTranslation()
  const currentKbId = useSettingsStore.use.currentKbId()
  const setCurrentKbId = useSettingsStore.use.setCurrentKbId()

  return (
    <Select
      value={currentKbId ?? 'none'}
      onValueChange={(value) => {
        const kbId = value === 'none' ? null : value
        setCurrentKbId(kbId)
        onChanged(kbId)
      }}
    >
      <SelectTrigger className="w-56">
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
  )
}

function SystemPromptDialog({ kbId }: { kbId: string | null }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  const load = async (id: string) => {
    setLoading(true)
    try {
      const data = await getSystemPrompt(id)
      setPrompt(data.system_prompt ?? '')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!kbId) return
    setSaving(true)
    try {
      await updateSystemPrompt(kbId, prompt)
      toast.success(t('chat.systemPromptSaved'))
      setOpen(false)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        setOpen(value)
        if (value && kbId) void load(kbId)
      }}
    >
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" tooltip={t('chat.systemPrompt')} disabled={!kbId}>
          <Wand2Icon aria-hidden="true" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>{t('chat.systemPrompt')}</DialogTitle>
          <DialogDescription>{t('chat.systemPromptHint')}</DialogDescription>
        </DialogHeader>
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={t('chat.systemPromptPlaceholder')}
          rows={10}
          disabled={loading || saving}
          className="font-mono text-xs"
        />
        <DialogFooter>
          <Button variant="secondary" onClick={() => setOpen(false)} disabled={saving}>
            {t('chat.cancel')}
          </Button>
          <Button onClick={handleSave} disabled={loading || saving}>
            {t('chat.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default function ChatView() {
  const { t } = useTranslation()
  const currentKbId = useSettingsStore.use.currentKbId()
  const queryParams = useSettingsStore.use.queryParams()
  const chatMessages = useSettingsStore.use.chatMessages()
  const appendChatMessage = useSettingsStore.use.appendChatMessage()
  const updateLastAssistantMessage = useSettingsStore.use.updateLastAssistantMessage()
  const clearChatMessages = useSettingsStore.use.clearChatMessages()

  const [databases, setDatabases] = useState<KnowledgeBase[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listDatabases()
      .then(({ databases }) => setDatabases(databases))
      .catch((e) => toast.error(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [chatMessages])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || streaming) return
    if (!currentKbId) {
      toast.warning(t('chat.noKbHint'))
      return
    }
    setInput('')
    appendChatMessage({ role: 'user', content: text })
    appendChatMessage({ role: 'assistant', content: '' })

    const controller = new AbortController()
    abortRef.current = controller
    setStreaming(true)
    try {
      // Send the full conversation history (minus the just-appended empty
      // assistant placeholder) so the model keeps multi-turn context.
      const history = [...chatMessages, { role: 'user' as const, content: text }].filter(
        (m) => !(m.role === 'assistant' && !m.content)
      )
      const messages = history.map((m) => ({ role: m.role, content: m.content }))
      await chatStream({
        kbId: currentKbId,
        messages,
        queryParams,
        signal: controller.signal,
        onEvent: (event) => {
          switch (event.type) {
            case 'sources':
              updateLastAssistantMessage({ sources: dedupeSources(event.chunks) })
              break
            case 'tool':
              updateLastAssistantMessage((prev) => ({
                toolCalls: [...(prev.toolCalls ?? []), event]
              }))
              break
            case 'delta':
              updateLastAssistantMessage((prev) => ({
                content: (prev.content ?? '') + event.content
              }))
              break
            case 'reasoning':
              updateLastAssistantMessage((prev) => ({
                reasoning: (prev.reasoning ?? '') + event.content
              }))
              break
            case 'error':
              updateLastAssistantMessage({ error: event.message })
              break
            default:
              break
          }
        }
      })
    } catch (e) {
      if (!controller.signal.aborted) {
        updateLastAssistantMessage({
          error: e instanceof Error ? e.message : String(e)
        })
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }

  const handleStop = () => {
    abortRef.current?.abort()
  }

  const handleClear = () => {
    clearChatMessages()
  }

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <MessageSquareTextIcon className="size-5 text-emerald-400" aria-hidden="true" />
          <h2 className="text-base font-semibold">{t('chat.title')}</h2>
        </div>
        <div className="flex items-center gap-2">
          <KnowledgeBaseSelector databases={databases} onChanged={() => clearChatMessages()} />
          <SystemPromptDialog kbId={currentKbId} />
          <Button variant="ghost" size="sm" tooltip={t('chat.clear')} onClick={handleClear}>
            <Trash2Icon aria-hidden="true" />
          </Button>
        </div>
      </div>

      <Card className="grid min-h-0 grow grid-rows-[1fr_auto]">
        <ScrollArea ref={scrollRef} className="h-full">
          <div className="grid gap-3 p-4">
            {chatMessages.length === 0 && (
              <p className="text-muted-foreground py-16 text-center text-sm">{t('chat.empty')}</p>
            )}
            {chatMessages.map((turn, index) => (
              <MessageBubble key={`${turn.role}-${index}`} turn={turn} />
            ))}
          </div>
        </ScrollArea>
        <div className="flex items-end gap-2 border-t p-3">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            placeholder={currentKbId ? t('chat.placeholder') : t('chat.noKbHint')}
            rows={2}
            disabled={streaming}
            className="resize-none"
          />
          {streaming ? (
            <Button onClick={handleStop} variant="secondary">
              <SquareIcon aria-hidden="true" />
              {t('chat.stop')}
            </Button>
          ) : (
            <Button onClick={handleSend} disabled={!input.trim() || !currentKbId}>
              <SendIcon aria-hidden="true" />
            </Button>
          )}
        </div>
      </Card>
    </div>
  )
}