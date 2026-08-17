import type { Chunk } from '@/api/yusu'

/**
 * Deduplicate chunks by citation_source (fallback: chunk id).
 * The backend marks each retrieved chunk with kb://{kb_id}/{file_id}?chunk={chunk_id};
 * the frontend numbers sources by this deduplicated order (1-based).
 */
export function dedupeSources(chunks: Chunk[]): Chunk[] {
  const seen = new Set<string>()
  const result: Chunk[] = []
  for (const chunk of chunks) {
    const key = chunk.citation_source || chunk.id
    if (seen.has(key)) continue
    seen.add(key)
    result.push(chunk)
  }
  return result
}

export interface ParsedCites {
  /** markdown-safe text with <cite>n</cite> replaced by [n] (invalid refs removed) */
  text: string
  /** valid 1-based cite numbers, in order of appearance */
  cited: number[]
}

const CITE_PATTERN = /<cite>(\d+)<\/cite>/g

/**
 * Convert backend <cite>n</cite> annotations (see SOURCE_CITE_PROMPT in
 * api/routers/chat.py) into plain [n] markers. Numbers outside the sources
 * range are dropped so the model cannot fabricate citations.
 */
export function parseCites(html: string, sources: Chunk[]): ParsedCites {
  const cited: number[] = []
  const text = html
    .replace(CITE_PATTERN, (_match, raw: string) => {
      const n = Number(raw)
      if (Number.isInteger(n) && n >= 1 && n <= sources.length) {
        cited.push(n)
        return `[${n}]`
      }
      return ''
    })
    // drop any leftover <cite>…</cite> fragments (invalid content) the number pattern missed
    .replace(/<cite>[^<]*<\/cite>/g, '')
    .replace(/<\/?cite>/g, '')
  return { text, cited }
}
