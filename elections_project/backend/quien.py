from flask import Flask, request, jsonify
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import pickle
import re
import nltk
import spacy
import ollama
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask_cors import CORS

# ---------------------------
# 1. Inicializar Flask y habilitar CORS
# ---------------------------
app = Flask(__name__)
CORS(app)

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

# ---------------------------
# 2. Cargar modelos, datos y FAISS
# ---------------------------

# Verificar si hay GPU disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modelo para generar embeddings
model = SentenceTransformer('paraphrase-distilroberta-base-v1', device=device)

# Cargar el archivo CSV con las oraciones procesadas
df = pd.read_csv('oraciones_procesadas_completo.csv', delimiter=';')

# Cargar el archivo CSV con los datos de los candidatos
candidatos_df = pd.read_csv('candidatos.csv', delimiter=';')

# Cargar el índice FAISS
index = faiss.read_index('faiss_index.index')

# Cargar los datos de oraciones
with open('sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

# ---------------------------
# 3. Función para normalizar texto
# ---------------------------
def normalizar_texto(texto):
    """Convierte el texto a minúsculas, sin tildes ni espacios extra."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

# ---------------------------
# 4. Función para preprocesar texto
# ---------------------------
def preprocess_text(text):
    """Elimina stopwords y aplica lematización."""
    tokens = word_tokenize(text.lower(), language='spanish')
    stop_words = set(stopwords.words('spanish'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    doc = nlp(' '.join(tokens))
    return ' '.join([token.lemma_ for token in doc])

# ---------------------------
# 5. Función para obtener información de una oración basada en su ID
# ---------------------------
def get_oracion_data_by_id(oracion_id):
    """Busca la oración original y sus temas clave por su ID."""
    oracion_row = df[df['Oracion_ID'] == oracion_id]
    if not oracion_row.empty:
        return oracion_row.iloc[0]['Oracion'], oracion_row.iloc[0]['Temas Clave']
    return None, None

# ---------------------------
# 6. Función para obtener información del partido y candidatos
# ---------------------------
def get_partido_data_by_id(id):
    """Busca el partido y candidatos por su ID."""
    candidato_row = candidatos_df[candidatos_df['ID'] == id]
    if not candidato_row.empty:
        return (
            candidato_row.iloc[0]['Partido'],
            candidato_row.iloc[0]['CandidatoPresidente'],
            candidato_row.iloc[0]['CandidatoVicePresidente'],
            candidato_row.iloc[0]['ListaPolitica']
        )
    return None, None, None, None

# ---------------------------
# 7. Función para determinar el tipo de consulta usando un switch
# ---------------------------
def determinar_tipo_pregunta(query):
    query = query.lower().strip()
    print(f"[INFO] Analizando pregunta: {query}")

    if re.search(r'\b(quién|quien) es\b', query):
        print("[INFO] Pregunta detectada como CANDIDATO.")
        return "candidatos"
    elif re.search(r'qué propone|cuál es la propuesta|qué dice.*(sobre|de)', query):
        print("[INFO] Pregunta detectada como PROPUESTA.")
        return "oraciones"
    elif re.search(r'qué temas menciona|de qué habla|qué menciona.*en', query):
        print("[INFO] Pregunta detectada como ENTREVISTAS.")
        return "entrevistas"
    else:
        print("[INFO] Pregunta sin clasificación, usará búsqueda general.")
        return "general"

# ---------------------------
# 8. Función de búsqueda en FAISS con OLLAMA
# ---------------------------
def query_faiss_ollama(query, top_k=50):
    tipo_pregunta = determinar_tipo_pregunta(query)

    # Si la pregunta es sobre candidatos, buscar en los datos de candidatos directamente
    if tipo_pregunta == "candidatos":
        return buscar_candidato(query)

    # Si no es sobre candidatos, realizar búsqueda en FAISS
    cleaned_query = preprocess_text(query)
    print(f"[INFO] Consulta procesada: {cleaned_query}")

    # Generar embedding de la consulta
    query_embedding = model.encode([cleaned_query], device=device).astype('float32')

    # Realizar búsqueda en FAISS
    D, I = index.search(query_embedding, top_k)

    # Procesar resultados
    resultados = []
    for i in range(top_k):
        oracion_id = sentences[I[0][i]]['Oracion_ID']
        id = sentences[I[0][i]]['ID']
        oracion_original, temas_clave = get_oracion_data_by_id(oracion_id)
        partido, candidato_presidente, candidato_vicepresidente, lista_politica = get_partido_data_by_id(id)

        if oracion_original:
            resultado = (
                f"Oración: {oracion_original}, "
                f"Temas Clave: {temas_clave}, "
                f"Partido: {partido}, "
                f"Presidente: {candidato_presidente}, "
                f"Vicepresidente: {candidato_vicepresidente}, "
                f"Lista Política: {lista_politica}"
            )
            resultados.append(resultado)

    if not resultados:
        return "No se encontraron resultados relevantes."

    # Construir el prompt para Ollama
    prompt = f"Analiza los siguientes resultados:\n\n" + "\n".join(resultados) + "\n\nGenera un resumen basado en estas declaraciones."

    # Generar la respuesta con Ollama
    respuesta = ollama.chat(model='llama3.2:latest', messages=[{'role': 'user', 'content': prompt}])

    return respuesta['message']['content']

# ---------------------------
# 9. Función de búsqueda de candidatos en FAISS
# ---------------------------
def buscar_candidato(query):
    query = normalizar_texto(query)
    print(f"[INFO] Buscando información sobre el candidato: {query}")

    for _, row in candidatos_df.iterrows():
        nombre_presidente = normalizar_texto(row['CandidatoPresidente'])
        nombre_vicepresidente = normalizar_texto(row['CandidatoVicePresidente'])

        if nombre_presidente in query:
            return {"Candidato": row["CandidatoPresidente"], "Descripción": row["WhoisPresident"]}
        elif nombre_vicepresidente in query:
            return {"Candidato": row["CandidatoVicePresidente"], "Descripción": row["WhoisVice"]}

    return {"error": "No se encontró información del candidato."}

# ---------------------------
# 10. Endpoint para la consulta
# ---------------------------
@app.route('/consulta', methods=['POST'])
def consulta_faiss():
    data = request.json
    query = data.get('consulta', '')

    print(f"\n[INFO] Nueva consulta recibida: {query}\n")

    try:
        respuesta = query_faiss_ollama(query)
        return jsonify({"consulta": query, "respuesta": respuesta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# 11. Iniciar la aplicación
# ---------------------------
if __name__ == '__main__':
    app.run(port=5000, debug=True)
