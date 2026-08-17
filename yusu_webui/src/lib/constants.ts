import { ButtonVariantType } from '@/components/ui/Button'

export const controlButtonVariant: ButtonVariantType = 'ghost'

export const healthCheckInterval = 15 // seconds

export const defaultQueryLabel = '*'

// reference: https://developer.mozilla.org/en-US/docs/Web/HTTP/MIME_types/Common_types
export const supportedFileTypes = {
  'text/plain': [
    '.txt',
    '.md',
    '.mdx',
    '.rtf',
    '.odt',
    '.tex',
    '.epub',
    '.html',
    '.htm',
    '.csv',
    '.json',
    '.xml',
    '.yaml',
    '.yml',
    '.log',
    '.conf',
    '.ini',
    '.properties',
    '.sql',
    '.bat',
    '.sh',
    '.c',
    '.h',
    '.cpp',
    '.hpp',
    '.py',
    '.java',
    '.js',
    '.ts',
    '.swift',
    '.go',
    '.rb',
    '.php',
    '.css',
    '.scss',
    '.less'
  ],
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
}

export const acceptedFileExtensions = Object.values(supportedFileTypes).flat()

export const SiteInfo = {
  name: 'YUSU 语溯',
  home: '/'
}