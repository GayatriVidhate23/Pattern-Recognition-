import React, { useCallback, useState } from 'react';
import { UploadCloud, File, X, Database } from 'lucide-react';

export default function FileUpload({ onUpload }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const files = Array.from(e.dataTransfer.files);
      setSelectedFiles(prev => [...prev, ...files]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const files = Array.from(e.target.files);
      setSelectedFiles(prev => [...prev, ...files]);
    }
  };

  const removeFile = (indexToRemove) => {
    setSelectedFiles(prev => prev.filter((_, index) => index !== indexToRemove));
  };

  const submitUpload = () => {
    if (selectedFiles.length >= 2) {
      onUpload(selectedFiles, false);
    } else {
      alert("Please upload at least 2 documents for clustering.");
    }
  };

  const submitDemo = () => {
    onUpload([], true);
  };

  return (
    <div className="w-full">
      <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-8 border border-slate-100">
        
        <div 
          className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200 ease-in-out ${
            dragActive ? 'border-indigo-500 bg-indigo-50/50 scale-[1.01]' : 'border-slate-300 bg-slate-50 hover:bg-slate-100'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input 
            type="file" 
            multiple 
            accept=".pdf,.docx,.txt"
            onChange={handleChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
          />
          
          <div className="flex flex-col items-center justify-center space-y-4 pointer-events-none">
            <div className={`p-4 rounded-full ${dragActive ? 'bg-indigo-100 text-indigo-600' : 'bg-white text-slate-400 shadow-sm'}`}>
              <UploadCloud className="w-10 h-10" />
            </div>
            <div>
              <p className="text-lg font-semibold text-slate-700">
                Drag & drop your documents here
              </p>
              <p className="text-sm text-slate-500 mt-1">
                or click to browse from your computer
              </p>
            </div>
            <div className="flex gap-2 text-xs font-medium text-slate-400 mt-4">
              <span className="bg-white px-2 py-1 rounded shadow-sm border border-slate-100">PDF</span>
              <span className="bg-white px-2 py-1 rounded shadow-sm border border-slate-100">DOCX</span>
              <span className="bg-white px-2 py-1 rounded shadow-sm border border-slate-100">TXT</span>
            </div>
          </div>
        </div>

        {selectedFiles.length > 0 && (
          <div className="mt-8">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider">
                Selected Files ({selectedFiles.length})
              </h3>
              <button 
                onClick={() => setSelectedFiles([])}
                className="text-xs text-red-500 hover:text-red-700 font-medium"
              >
                Clear All
              </button>
            </div>
            <ul className="space-y-2 max-h-48 overflow-y-auto pr-2 custom-scrollbar">
              {selectedFiles.map((file, idx) => (
                <li key={`${file.name}-${idx}`} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-100 group">
                  <div className="flex items-center space-x-3 overflow-hidden">
                    <File className="w-5 h-5 text-indigo-400 flex-shrink-0" />
                    <span className="text-sm text-slate-700 truncate font-medium">{file.name}</span>
                    <span className="text-xs text-slate-400 flex-shrink-0">
                      {(file.size / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <button 
                    onClick={() => removeFile(idx)}
                    className="p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </li>
              ))}
            </ul>
            
            <div className="mt-6">
              <button
                onClick={submitUpload}
                disabled={selectedFiles.length < 2}
                className={`w-full py-3 px-4 rounded-xl font-semibold text-white transition-all shadow-md flex justify-center items-center ${
                  selectedFiles.length >= 2 
                    ? 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-indigo-200 shadow-lg transform hover:-translate-y-0.5' 
                    : 'bg-slate-300 cursor-not-allowed'
                }`}
              >
                Analyze {selectedFiles.length} Document{selectedFiles.length !== 1 ? 's' : ''}
              </button>
              {selectedFiles.length < 2 && (
                <p className="text-xs text-center text-red-500 mt-2">
                  * Minimum 2 documents required for clustering
                </p>
              )}
            </div>
          </div>
        )}

        <div className="mt-10 pt-8 border-t border-slate-100">
          <div className="relative">
            <div className="absolute inset-0 flex items-center" aria-hidden="true">
              <div className="w-full border-t border-slate-200"></div>
            </div>
            <div className="relative flex justify-center">
              <span className="px-3 bg-white text-sm text-slate-400 font-medium uppercase tracking-widest">or</span>
            </div>
          </div>
          
          <div className="mt-8 text-center">
            <button
              onClick={submitDemo}
              className="inline-flex items-center justify-center px-6 py-3 border-2 border-indigo-100 text-indigo-700 bg-indigo-50 hover:bg-indigo-100 hover:border-indigo-200 rounded-xl font-semibold transition-all group"
            >
              <Database className="w-5 h-5 mr-2 text-indigo-500 group-hover:text-indigo-600" />
              Load Demo Dataset
            </button>
            <p className="mt-3 text-sm text-slate-500">
              Try the platform instantly with a curated set of 10 diverse articles.
            </p>
          </div>
        </div>
        
      </div>
    </div>
  );
}
