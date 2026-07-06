/**
 * Trigger a browser download of in-memory text. DOM-touching (Blob + object URL
 * + a transient anchor click), kept tiny and isolated so callers stay testable.
 */
export function downloadText(
  filename: string,
  text: string,
  mime = 'application/octet-stream',
): void {
  const blob = new Blob([text], { type: mime })
  const url = URL.createObjectURL(blob)
  try {
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.style.display = 'none'
    document.body.appendChild(a)
    a.click()
    a.remove()
  } finally {
    URL.revokeObjectURL(url)
  }
}
