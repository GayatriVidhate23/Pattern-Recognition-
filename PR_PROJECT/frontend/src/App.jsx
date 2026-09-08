import { useState } from 'react'
import axios from 'axios'
import { UploadCloud, FolderTree, BarChart2, Layers, Search, Download, CheckCircle, FileText } from 'lucide-react'
import ReactECharts from 'echarts-for-react'

// Components
import FileUpload from './components/FileUpload'
import Dashboard from './components/Dashboard'

const API_BASE_URL = 'http://localhost:8000/api'

function App() {
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleUpload = async (files, useDemo = false) => {
    setIsProcessing(true)
    setError(null)
    
    const formData = new FormData()
    if (useDemo) {
      formData.append('use_demo', 'true')
    } else {
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i])
      }
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/cluster`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setResults(response.data)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || 'An error occurred during clustering.')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <FolderTree className="h-8 w-8 text-indigo-600 mr-3" />
              <span className="font-bold text-xl tracking-tight text-slate-800">DocuCluster AI</span>
            </div>
            {results && (
              <div className="flex items-center">
                <button 
                  onClick={handleReset}
                  className="text-sm font-medium text-slate-600 hover:text-indigo-600 transition-colors"
                >
                  Start New Analysis
                </button>
              </div>
            )}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!results && !isProcessing && (
          <div className="max-w-3xl mx-auto mt-12">
            <div className="text-center mb-10">
              <h1 className="text-4xl font-extrabold text-slate-900 mb-4 tracking-tight">
                Intelligent Document Organization
              </h1>
              <p className="text-lg text-slate-600">
                Upload your unstructured documents (PDF, DOCX, TXT) and let our AI automatically group them into logical, semantic clusters using advanced Hierarchical Agglomerative Clustering and TF-IDF vectors.
              </p>
            </div>
            
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-md shadow-sm">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-red-700 font-medium">{error}</p>
                  </div>
                </div>
              </div>
            )}

            <FileUpload onUpload={handleUpload} />
          </div>
        )}

        {isProcessing && (
          <div className="flex flex-col items-center justify-center h-64 mt-20">
            <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-indigo-600 mb-6"></div>
            <h2 className="text-2xl font-semibold text-slate-800 mb-2">Analyzing Documents...</h2>
            <p className="text-slate-500">Extracting text, computing TF-IDF vectors, and generating hierarchical clusters.</p>
            <div className="mt-8 flex space-x-2">
              <div className="w-3 h-3 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-3 h-3 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-3 h-3 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
          </div>
        )}

        {results && !isProcessing && (
          <Dashboard results={results} />
        )}
      </main>
    </div>
  )
}

export default App
