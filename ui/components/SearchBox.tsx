'use client'

import { useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { debounce } from 'lodash'

export default function SearchBox() {
  const [query, setQuery] = useState('')

  const debouncedSearch = useCallback(
    debounce((q: string) => {
      if (q.trim()) {
        setQuery(q)
      }
    }, 300),
    []
  )

  const { data: results, isLoading } = useQuery({
    queryKey: ['memories', 'search', query],
    queryFn: () => api.searchMemories(query),
    enabled: !!query.trim()
  })

  return (
    <div className="space-y-4">
      <input
        type="text"
        onChange={(e) => debouncedSearch(e.target.value)}
        placeholder="Search memories..."
        className="w-full p-2 border rounded-lg"
      />

      {isLoading && <div>Searching...</div>}

      {results && (
        <div className="space-y-2">
          {results.map((memory) => (
            <div key={memory.id} className="p-4 border rounded-lg">
              <p>{memory.text}</p>
              <div className="mt-2 text-sm text-gray-500">
                Similarity: {(memory.similarity).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
