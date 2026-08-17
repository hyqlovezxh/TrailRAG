import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'
import { getHealth, type HealthInfo } from '@/api/yusu'
import { healthCheckInterval } from '@/lib/constants'

interface BackendState {
  health: boolean
  message: string | null
  status: HealthInfo | null
  lastCheckTime: number
  healthCheckIntervalId: ReturnType<typeof setInterval> | null
  healthCheckFunction: (() => void) | null

  check: () => Promise<boolean>
  clear: () => void
  setHealthCheckFunction: (fn: () => void) => void
  resetHealthCheckTimer: () => void
  clearHealthCheckTimer: () => void
}

const useBackendStateStoreBase = create<BackendState>()((set, get) => ({
  health: true,
  message: null,
  status: null,
  lastCheckTime: Date.now(),
  healthCheckIntervalId: null,
  healthCheckFunction: null,

  check: async () => {
    try {
      const status = await getHealth()
      set({
        health: status.status === 'ok',
        message: null,
        status,
        lastCheckTime: Date.now()
      })
      return true
    } catch (error) {
      set({
        health: false,
        message: error instanceof Error ? error.message : String(error),
        status: null,
        lastCheckTime: Date.now()
      })
      return false
    }
  },

  clear: () => {
    set({ health: true, message: null })
  },

  setHealthCheckFunction: (fn: () => void) => {
    set({ healthCheckFunction: fn })
  },

  resetHealthCheckTimer: () => {
    const { healthCheckIntervalId, healthCheckFunction } = get()
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId)
    }
    if (healthCheckFunction) {
      healthCheckFunction() // run health check immediately
      const newIntervalId = setInterval(healthCheckFunction, healthCheckInterval * 1000)
      set({ healthCheckIntervalId: newIntervalId })
    }
  },

  clearHealthCheckTimer: () => {
    const { healthCheckIntervalId } = get()
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId)
      set({ healthCheckIntervalId: null })
    }
  }
}))

const useBackendState = createSelectors(useBackendStateStoreBase)

export { useBackendState }