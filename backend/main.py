from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uuid
from ml_utils import extract_text, cluster_documents, clean_text
import uvicorn

app = FastAPI(title="Hierarchical Clustering API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_DOCUMENTS = [
    {"filename": "AI_in_Healthcare.txt", "text": "Artificial intelligence is transforming healthcare through predictive analytics, personalized medicine, and automated diagnosis. Machine learning models can predict patient outcomes with high accuracy."},
    {"filename": "Machine_Learning_Basics.txt", "text": "Machine learning is a subset of AI that focuses on building systems that learn from data. Supervised learning, unsupervised learning, and reinforcement learning are key paradigms."},
    {"filename": "Deep_Learning_Advances.txt", "text": "Deep learning uses neural networks with many layers to solve complex problems like image recognition and natural language processing. The backpropagation algorithm is fundamental to training these networks."},
    {"filename": "Climate_Change_Impacts.txt", "text": "Global warming is causing rising sea levels, extreme weather events, and loss of biodiversity. Carbon emissions must be reduced to mitigate these severe environmental impacts."},
    {"filename": "Renewable_Energy_Sources.txt", "text": "Solar, wind, and hydroelectric power are key renewable energy sources. Transitioning to green energy is essential for sustainable development and reducing greenhouse gas emissions."},
    {"filename": "Financial_Markets_Overview.txt", "text": "Stock markets and financial instruments allow companies to raise capital. Investors analyze market trends, P/E ratios, and economic indicators to make trading decisions."},
    {"filename": "Investment_Strategies.txt", "text": "Diversification is a core investment strategy to manage risk. Index funds, mutual funds, and bonds provide different risk-return profiles for building a robust portfolio."},
    {"filename": "Nutrition_and_Diet.txt", "text": "A balanced diet rich in vitamins, minerals, and macronutrients is crucial for human health. Limiting processed foods and sugars can prevent metabolic diseases."},
    {"filename": "Exercise_Benefits.txt", "text": "Regular physical activity strengthens the cardiovascular system, improves mental health, and builds muscle mass. Aerobic and resistance training should be combined."},
    {"filename": "Space_Exploration.txt", "text": "NASA and SpaceX are advancing space exploration with reusable rockets and plans for Mars colonization. Telescopes like James Webb are discovering new exoplanets."},
]

@app.post("/api/cluster")
async def perform_clustering(
    files: Optional[List[UploadFile]] = File(None),
    use_demo: Optional[bool] = Form(False)
):
    documents = []
    
    if use_demo:
        for i, doc in enumerate(DEMO_DOCUMENTS):
            documents.append({
                "id": f"demo-{i}",
                "filename": doc["filename"],
                "text": clean_text(doc["text"])
            })
    else:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided and demo mode not selected.")
            
        for file in files:
            content = await file.read()
            text = extract_text(content, file.filename)
            if not text.strip():
                continue # Skip empty or unparseable files
            documents.append({
                "id": str(uuid.uuid4()),
                "filename": file.filename,
                "text": clean_text(text)
            })
            
    if len(documents) < 2:
        raise HTTPException(status_code=400, detail="At least 2 valid documents are required for clustering.")
        
    try:
        results = cluster_documents(documents)
        
        # Format response
        response_docs = []
        for i, doc in enumerate(documents):
            cluster_id = results['cluster_labels'][i]
            response_docs.append({
                "id": doc['id'],
                "filename": doc['filename'],
                "text_preview": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                "cluster_id": int(cluster_id)
            })
            
        return {
            "documents": response_docs,
            "clusters": results['cluster_keywords'],
            "similarity_matrix": results['similarity_matrix'],
            "linkage_matrix": results['linkage_matrix'],
            "points_2d": results['points_2d']
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
