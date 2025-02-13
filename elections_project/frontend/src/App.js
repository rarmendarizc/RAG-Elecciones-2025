import React, { useState } from "react";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import ReactMarkdown from "react-markdown";

function App() {
  const [consulta, setConsulta] = useState("");
  const [respuesta, setRespuesta] = useState("");
  const [metricas, setMetricas] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const manejarEnvio = async (e) => {
    e.preventDefault();
    setLoading(true);
    setRespuesta("");
    setMetricas(null);
    setError(null);

    try {
      const response = await fetch("http://localhost:5000/consulta", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ consulta }),
      });

      if (!response.ok) {
        throw new Error("Error al conectar con el backend.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let accumulatedResponse = "";
      setRespuesta("");

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        accumulatedResponse += chunk;
        setRespuesta((prev) => prev + chunk);
      }

      try {
        const jsonResponse = JSON.parse(accumulatedResponse);
        setRespuesta(jsonResponse.respuesta || "No se pudo generar una respuesta.");
        if (jsonResponse.metricas) {
          setMetricas({
            precision: jsonResponse.metricas.precision?.toFixed(3) || "0.000",
            recall: jsonResponse.metricas.recall?.toFixed(3) || "0.000",
            f1_score: jsonResponse.metricas.f1_score?.toFixed(3) || "0.000",
          });
        }
      } catch (err) {
        console.error("Error parseando JSON", err);
      }
    } catch (error) {
      setError("Error al conectar con el backend.");
    }

    setLoading(false);
  };

  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-8 text-center">
          <img src="/alphaquery.png" alt="Logo" className={`logo mb-3 ${respuesta ? "small-logo" : ""}`} />
          <form onSubmit={manejarEnvio} className="d-flex">
            <input
              type="text"
              className="form-control me-2"
              placeholder="Escriba su consulta aquí..."
              value={consulta}
              onChange={(e) => setConsulta(e.target.value)}
              style={{ fontSize: "1.2rem", padding: "10px", height: "50px" }} 
            />
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? "Consultando..." : "Enviar"}
            </button>
          </form>
        </div>
      </div>

      {respuesta && (
        <div className="row justify-content-center mt-4">
          <div className="col-md-8">
            <div className="card p-4 shadow-sm respuesta-box">
              <ReactMarkdown className="respuesta-texto">{respuesta}</ReactMarkdown>
            </div>
          </div>

          {metricas && (
            <div className="col-md-4">
              <div className="card p-3 shadow-sm">
                <h4 className="mb-3">Métricas de evaluación</h4>
                <p><strong>Precisión:</strong> {metricas.precision}</p>
                <p><strong>Recall:</strong> {metricas.recall}</p>
                <p><strong>F1-Score:</strong> {metricas.f1_score}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;