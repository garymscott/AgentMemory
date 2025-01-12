'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export default function MemoryList() {
  const { data: memories, isLoading } = useQuery({
    queryKey: ['memories'],
    queryFn: api.getMemories
  })

  if (isLoading) return <div>Loading...</div>

  return (
    <div className="space-y-4">
      {memories?.map((memory) => (
        <div key={memory.id} className="p-4 border rounded-lg">
          <p>{memory.text}</p>
          {memory.metadata && Object.keys(memory.metadata).length > 0 && (
            <div className="mt-2 space-x-2">
              {Object.entries(memory.metadata).map(([key, value]) => (
                <span key={key} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  {key}: {value}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
