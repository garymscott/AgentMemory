import MemoryInput from '../components/MemoryInput'
import MemoryList from '../components/MemoryList'
import SearchBox from '../components/SearchBox'

export default function Home() {
  return (
    <main className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <h1 className="text-4xl font-bold">Agent Memory</h1>
        
        {/* Memory Input */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-2xl font-semibold mb-4">Add Memory</h2>
          <MemoryInput />
        </div>

        {/* Search Interface */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-2xl font-semibold mb-4">Search Memories</h2>
          <SearchBox />
        </div>

        {/* Memory List */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-2xl font-semibold mb-4">Recent Memories</h2>
          <MemoryList />
        </div>
      </div>
    </main>
  )
}