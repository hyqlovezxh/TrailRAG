import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { createSelectors } from '@/lib/utils'
import type { Chunk, ToolEvent } from '@/api/yusu'

export type Theme = 'dark' | 'light' | 'system'
export type Language = 'en' | 'zh' | 'fr' | 'ar' | 'zh_TW' | 'ru' | 'ja' | 'de' | 'uk' | 'ko' | 'vi'
export type Tab = 'knowledge' | 'chat' | 'models' | 'graph' | 'eval'

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
  sources?: Chunk[]
  /** KB tool calls executed while producing this answer (SSE `tool` events) */
  toolCalls?: ToolEvent[]
  reasoning?: string
  error?: string | null
}

export const defaultQueryParams: Record<string, unknown> = {
  search_mode: 'vector'
}

interface SettingsState {
  // App settings
  theme: Theme
  setTheme: (theme: Theme) => void

  language: Language
  setLanguage: (lang: Language) => void

  currentTab: Tab
  setCurrentTab: (tab: Tab) => void

  // YUSU connection
  baseUrl: string
  setBaseUrl: (url: string) => void

  apiKey: string | null
  setApiKey: (key: string | null) => void

  // Current knowledge base
  currentKbId: string | null
  setCurrentKbId: (kbId: string | null) => void

  queryParams: Record<string, unknown>
  setQueryParams: (params: Record<string, unknown>) => void

  // Chat history
  chatMessages: ChatTurn[]
  setChatMessages: (messages: ChatTurn[]) => void
  appendChatMessage: (message: ChatTurn) => void
  updateLastAssistantMessage: (
    patch: Partial<ChatTurn> | ((prev: ChatTurn) => Partial<ChatTurn>)
  ) => void
  clearChatMessages: () => void
}

const useSettingsStoreBase = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'system',
      language: 'en',
      currentTab: 'knowledge',
      baseUrl: 'http://127.0.0.1:8920',
      apiKey: null,
      currentKbId: null,
      queryParams: { ...defaultQueryParams },
      chatMessages: [],

      setTheme: (theme: Theme) => set({ theme }),
      setLanguage: (language: Language) => set({ language }),
      setCurrentTab: (currentTab: Tab) => set({ currentTab }),
      setBaseUrl: (baseUrl: string) => set({ baseUrl: baseUrl.trim().replace(/\/+$/, '') }),
      setApiKey: (apiKey: string | null) => set({ apiKey: apiKey?.trim() ? apiKey.trim() : null }),
      setCurrentKbId: (currentKbId: string | null) => set({ currentKbId }),
      setQueryParams: (queryParams: Record<string, unknown>) => set({ queryParams }),

      setChatMessages: (chatMessages: ChatTurn[]) => set({ chatMessages: chatMessages.slice(-100) }),
      appendChatMessage: (message: ChatTurn) =>
        set((state) => ({ chatMessages: [...state.chatMessages, message].slice(-100) })),
      updateLastAssistantMessage: (
        patch: Partial<ChatTurn> | ((prev: ChatTurn) => Partial<ChatTurn>)
      ) =>
        set((state) => {
          if (state.chatMessages.length === 0) return state
          const messages = [...state.chatMessages]
          const last = messages[messages.length - 1]
          if (last.role !== 'assistant') return state
          const applied = typeof patch === 'function' ? patch(last) : patch
          messages[messages.length - 1] = { ...last, ...applied }
          return { chatMessages: messages }
        }),
      clearChatMessages: () => set({ chatMessages: [] })
    }),
    {
      name: 'settings-storage',
      storage: createJSONStorage(() => localStorage),
      version: 21,
      // Discard all legacy YUSU settings; keep only theme/language.
      migrate: (state: any) => ({
        theme: state?.theme ?? 'system',
        language: state?.language ?? 'en',
        currentTab: 'knowledge',
        baseUrl: 'http://127.0.0.1:8920',
        apiKey: null,
        currentKbId: null,
        queryParams: { ...defaultQueryParams },
        chatMessages: []
      })
    }
  )
)

const useSettingsStore = createSelectors(useSettingsStoreBase)

export { useSettingsStore }