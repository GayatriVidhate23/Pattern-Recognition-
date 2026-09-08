import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Layers, FileText, BarChart2, Download, Search, Tag } from 'lucide-react';

// Helper to convert SciPy linkage matrix to ECharts Tree
const buildTreeData = (linkage, docs) => {
  if (!linkage || !docs || linkage.length === 0) return {};
  
  const N = docs.length;
  const nodes = new Array(2 * N - 1);
  
  // Initialize leaves
  for (let i = 0; i < N; i++) {
    nodes[i] = { 
      name: docs[i].filename, 
      value: 0,
      itemStyle: { color: '#6366f1' } // indigo
    };
  }
  
  // Build tree from linkage
  for (let i = 0; i < linkage.length; i++) {
    const [leftIdx, rightIdx, distance, size] = linkage[i];
    const newIdx = N + i;
    nodes[newIdx] = {
      name: `Cluster ${i + 1}`,
      value: distance.toFixed(2),
      children: [nodes[leftIdx], nodes[rightIdx]],
      itemStyle: { color: '#cbd5e1' } // slate
    };
  }
  
  return nodes[2 * N - 2]; // Root
};

export default function Dashboard({ results }) {
  const { documents, clusters, similarity_matrix, linkage_matrix, points_2d } = results;
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCluster, setSelectedCluster] = useState('All');

  const numClusters = Object.keys(clusters).length;
  
  // Group docs by cluster
  const docsByCluster = useMemo(() => {
    const grouped = {};
    documents.forEach(doc => {
      if (!grouped[doc.cluster_id]) grouped[doc.cluster_id] = [];
      grouped[doc.cluster_id].push(doc);
    });
    return grouped;
  }, [documents]);

  const treeData = useMemo(() => buildTreeData(linkage_matrix, documents), [linkage_matrix, documents]);

  const scatterData = useMemo(() => {
    // ECharts scatter series data needs to be separated by cluster for different colors/legends
    const series = [];
    Object.keys(clusters).forEach(clusterIdStr => {
      const clusterId = parseInt(clusterIdStr);
      const dataPoints = [];
      
      documents.forEach((doc, idx) => {
        if (doc.cluster_id === clusterId && points_2d[idx]) {
          dataPoints.push({
            name: doc.filename,
            value: [points_2d[idx][0], points_2d[idx][1]],
            docId: doc.id
          });
        }
      });
      
      if (dataPoints.length > 0) {
        series.push({
          name: `Cluster ${clusterId + 1}`,
          type: 'scatter',
          symbolSize: 12,
          data: dataPoints,
          emphasis: {
            focus: 'series'
          }
        });
      }
    });
    return series;
  }, [clusters, documents, points_2d]);

  const getDendrogramOption = () => ({
    tooltip: { trigger: 'item', triggerOn: 'mousemove' },
    series: [
      {
        type: 'tree',
        data: [treeData],
        top: '5%',
        left: '10%',
        bottom: '5%',
        right: '25%',
        symbolSize: 10,
        label: {
          position: 'left',
          verticalAlign: 'middle',
          align: 'right',
          fontSize: 11
        },
        leaves: {
          label: {
            position: 'right',
            verticalAlign: 'middle',
            align: 'left'
          }
        },
        emphasis: { focus: 'descendant' },
        expandAndCollapse: true,
        animationDuration: 550,
        animationDurationUpdate: 750
      }
    ]
  });

  const getScatterOption = () => ({
    title: { text: '2D Document Projection (PCA)', left: 'center', textStyle: { fontSize: 14, fontWeight: 'normal', color: '#64748b' } },
    tooltip: {
      formatter: function (params) {
        return `<div style="font-weight:bold">${params.data.name}</div>${params.seriesName}`;
      }
    },
    legend: { bottom: 0 },
    xAxis: { type: 'value', splitLine: { show: false } },
    yAxis: { type: 'value', splitLine: { show: false } },
    series: scatterData
  });

  const downloadCSV = () => {
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Filename,Cluster ID,Cluster Keywords\n";
    documents.forEach(doc => {
      const clusterKw = clusters[doc.cluster_id] ? clusters[doc.cluster_id].join("; ") : "";
      csvContent += `"${doc.filename}",${doc.cluster_id + 1},"${clusterKw}"\n`;
    });
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "clustering_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredDocs = documents.filter(doc => {
    const matchesSearch = doc.filename.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          doc.text_preview.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCluster = selectedCluster === 'All' || doc.cluster_id.toString() === selectedCluster.toString();
    return matchesSearch && matchesCluster;
  });

  return (
    <div className="space-y-6">
      
      {/* Top Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm flex items-center space-x-4">
          <div className="p-4 bg-indigo-50 rounded-xl text-indigo-600">
            <FileText className="w-8 h-8" />
          </div>
          <div>
            <p className="text-sm text-slate-500 font-medium">Total Documents</p>
            <h3 className="text-3xl font-bold text-slate-800">{documents.length}</h3>
          </div>
        </div>
        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm flex items-center space-x-4">
          <div className="p-4 bg-emerald-50 rounded-xl text-emerald-600">
            <Layers className="w-8 h-8" />
          </div>
          <div>
            <p className="text-sm text-slate-500 font-medium">Clusters Discovered</p>
            <h3 className="text-3xl font-bold text-slate-800">{numClusters}</h3>
          </div>
        </div>
        <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm flex items-center space-x-4 justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-4 bg-amber-50 rounded-xl text-amber-600">
              <BarChart2 className="w-8 h-8" />
            </div>
            <div>
              <p className="text-sm text-slate-500 font-medium">Average Similarity</p>
              <h3 className="text-3xl font-bold text-slate-800">
                {(similarity_matrix.flat().reduce((a, b) => a + b, 0) / (documents.length * documents.length) * 100).toFixed(1)}%
              </h3>
            </div>
          </div>
        </div>
      </div>

      {/* Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6">
          <div className="mb-4">
            <h3 className="text-lg font-bold text-slate-800">Hierarchical Dendrogram</h3>
            <p className="text-sm text-slate-500">Visualizing the agglomerative merging of documents.</p>
          </div>
          <ReactECharts option={getDendrogramOption()} style={{ height: '400px' }} />
        </div>
        
        <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6">
          <div className="mb-4">
            <h3 className="text-lg font-bold text-slate-800">Semantic Proximity</h3>
            <p className="text-sm text-slate-500">2D Scatter plot of TF-IDF vectors using PCA.</p>
          </div>
          <ReactECharts option={getScatterOption()} style={{ height: '400px' }} />
        </div>
      </div>

      {/* Clusters & Documents */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
        <div className="p-6 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-xl font-bold text-slate-800">Organized Documents</h3>
            <p className="text-sm text-slate-500 mt-1">Browse documents grouped by AI-detected themes.</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
              <input 
                type="text" 
                placeholder="Search docs..."
                className="pl-9 pr-4 py-2 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all w-48 sm:w-64"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <button 
              onClick={downloadCSV}
              className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>
        
        <div className="flex flex-col lg:flex-row">
          {/* Sidebar: Clusters */}
          <div className="w-full lg:w-1/4 border-r border-slate-100 bg-slate-50/50 p-4">
            <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 pl-2">Filter by Cluster</h4>
            <ul className="space-y-1">
              <li>
                <button
                  onClick={() => setSelectedCluster('All')}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedCluster === 'All' ? 'bg-indigo-100 text-indigo-700' : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  All Clusters
                </button>
              </li>
              {Object.keys(clusters).map((clusterId) => (
                <li key={clusterId}>
                  <button
                    onClick={() => setSelectedCluster(clusterId)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      selectedCluster === clusterId ? 'bg-indigo-100 text-indigo-700' : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span>Cluster {parseInt(clusterId) + 1}</span>
                      <span className="bg-white text-xs py-0.5 px-2 rounded-full border border-slate-200 text-slate-500">
                        {docsByCluster[clusterId]?.length || 0}
                      </span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Main Content: Documents List */}
          <div className="w-full lg:w-3/4 p-6 bg-white min-h-[400px]">
            {filteredDocs.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-slate-400">
                <Search className="w-12 h-12 mb-4 opacity-20" />
                <p>No documents found matching your criteria.</p>
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                {filteredDocs.map((doc) => {
                  const keywords = clusters[doc.cluster_id] || [];
                  return (
                    <div key={doc.id} className="border border-slate-200 rounded-xl p-5 hover:shadow-md transition-shadow group relative overflow-hidden">
                      <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500"></div>
                      <div className="flex justify-between items-start mb-3">
                        <h5 className="font-semibold text-slate-800 text-sm break-words pr-4 line-clamp-2" title={doc.filename}>
                          {doc.filename}
                        </h5>
                        <span className="bg-indigo-50 text-indigo-600 text-xs px-2 py-1 rounded-md font-medium whitespace-nowrap">
                          Cluster {doc.cluster_id + 1}
                        </span>
                      </div>
                      
                      <p className="text-xs text-slate-500 mb-4 line-clamp-3 leading-relaxed">
                        {doc.text_preview}
                      </p>
                      
                      <div className="mt-auto pt-3 border-t border-slate-100">
                        <div className="flex flex-wrap gap-1.5">
                          {keywords.map((kw, i) => (
                            <span key={i} className="inline-flex items-center text-[10px] font-medium px-1.5 py-0.5 rounded text-slate-600 bg-slate-100">
                              <Tag className="w-3 h-3 mr-1 opacity-50" />
                              {kw}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>
      
    </div>
  );
}
