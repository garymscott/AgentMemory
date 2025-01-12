const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface Memory {
  id: string
  text: string
  metadata: Record<string, string>
  similarity?: number
}

export interface MemoryCreate {
  text: string
  metadata?: Record<string, string>
}

export const api = {
  createMemory: async (data: MemoryCreate): Promise<string> => {
    const response = await fetch(`${API_URL}/memories/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) throw new Error('Failed to create memory')
    return response.json()
  },

  getMemories: async (): Promise<Memory[]> => {
    const response = await fetch(`${API_URL}/memories/`)
    if (!response.ok) throw new Error('Failed to fetch memories')
    return response.json()
  },

  searchMemories: async (query: string): Promise<Memory[]> => {
    const response = await fetch(`${API_URL}/memories/search/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    })
    if (!response.ok) throw new Error('Failed to search memories')
    return response.json()
  },

  deleteMemory: async (id: string): Promise<boolean> => {
    const response = await fetch(`${API_URL}/memories/${id}`, {
      method: 'DELETE'
    })
    if (!response.ok) throw new Error('Failed to delete memory')
    return response.json()
  }
}