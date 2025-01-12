'use client'

import { useState } from 'react'
import { useQueryClient, useMutation } from '@tanstack/react-query'
import { api } from '../lib/api'

export default function MemoryInput() {
  const [text, setText] = useState('')
  const [metadata, setMetadata] = useState<Record<string, string>>({})
  const queryClient = useQueryClient()

  const { mutate, isPending } = useMutation({
    mutationFn: api.createMemory,
    onSuccess: () => {
      setText('')
      setMetadata({})
      queryClient.invalidateQueries({ queryKey: ['memories'] })
    }
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    mutate({ text, metadata })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="w-full h-32 p-2 border rounded-lg"
          placeholder="Enter memory text..."
          required
        />
      </div>

      {/* Metadata Fields */}
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Key"
          className="flex-1 p-2 border rounded-lg"
          onChange={(e) => setMetadata({ ...metadata, [e.target.value]: '' })}
        />
        <input
          type="text"
          placeholder="Value"
          className="flex-1 p-2 border rounded-lg"
          onChange={(e) => {
            const key = Object.keys(metadata)[0]
            setMetadata({ ...metadata, [key]: e.target.value })
          }}
        />
      </div>

      <button
        type="submit"
        disabled={isPending}
        className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
      >
        {isPending ? 'Adding...' : 'Add Memory'}
      </button>
    </form>
  )
}
