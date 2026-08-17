import { describe, expect, it } from 'bun:test'
import { dedupeSources, parseCites } from './parseCites'
import type { Chunk } from '@/api/yusu'

const chunk = (id: string, citationSource?: string): Chunk => ({
  id,
  kb_id: 'kb1',
  file_id: `f-${id}`,
  content: `content-${id}`,
  ...(citationSource ? { citation_source: citationSource } : {})
})

describe('dedupeSources', () => {
  it('drops chunks with duplicate citation_source keeping first occurrence', () => {
    const sources = [
      chunk('a', 'kb://kb1/f1?chunk=a'),
      chunk('b', 'kb://kb1/f1?chunk=a'),
      chunk('c', 'kb://kb1/f2?chunk=c')
    ]
    const result = dedupeSources(sources)
    expect(result.map((s) => s.id)).toEqual(['a', 'c'])
  })

  it('falls back to chunk id when citation_source is missing', () => {
    const sources = [chunk('a'), chunk('a'), chunk('b')]
    const result = dedupeSources(sources)
    expect(result.map((s) => s.id)).toEqual(['a', 'b'])
  })

  it('returns an empty array for empty input', () => {
    expect(dedupeSources([])).toEqual([])
  })
})

describe('parseCites', () => {
  const sources = [chunk('a'), chunk('b'), chunk('c')]

  it('replaces <cite>n</cite> with [n] and collects valid numbers', () => {
    const { text, cited } = parseCites('张三向李四转账<cite>1</cite>，次日还款<cite>2</cite>。', sources)
    expect(text).toBe('张三向李四转账[1]，次日还款[2]。')
    expect(cited).toEqual([1, 2])
  })

  it('drops out-of-range numbers and does not collect them', () => {
    const { text, cited } = parseCites('编造的引用<cite>9</cite>应被移除<cite>3</cite>。', sources)
    expect(text).toBe('编造的引用应被移除[3]。')
    expect(cited).toEqual([3])
  })

  it('handles adjacent multi-citations', () => {
    const { text, cited } = parseCites('结论<cite>1</cite><cite>2</cite>。', sources)
    expect(text).toBe('结论[1][2]。')
    expect(cited).toEqual([1, 2])
  })

  it('ignores text without cite tags', () => {
    const { text, cited } = parseCites('没有引用的回答。', sources)
    expect(text).toBe('没有引用的回答。')
    expect(cited).toEqual([])
  })

  it('returns empty cited for empty sources', () => {
    const { text, cited } = parseCites('引用<cite>1</cite>但无资料。', [])
    expect(text).toBe('引用但无资料。')
    expect(cited).toEqual([])
  })

  it('ignores non-integer and zero cite numbers', () => {
    const { text, cited } = parseCites('引用<cite>0</cite>与<cite>abc</cite>均无效。', sources)
    expect(text).toBe('引用与均无效。')
    expect(cited).toEqual([])
  })
})
