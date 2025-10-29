// ===============================================================
// Cana ML Dashboard v3.2 - JavaScript (geo opcional + UX)
// Autor: Adão
// ===============================================================

// ===============================================================
// VERIFICA LOGIN
// ===============================================================
if (!localStorage.getItem("auth")) {
  window.location.href = "/";
}

// Logout
document.getElementById("logoutButton").addEventListener("click", () => {
  localStorage.removeItem("auth");
  window.location.href = "/";
});

// ===============================================================
// NAVEGAÇÃO ENTRE SEÇÕES
// ===============================================================
document.querySelectorAll("nav a").forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    const target = link.getAttribute("data-section");
    document.querySelectorAll("section").forEach(s => s.classList.remove("active"));
    document.getElementById(target).classList.add("active");
  });
});

// ===============================================================
// CONFIGURAÇÃO DA API
// ===============================================================
const API_BASE =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:5001"
    : `http://${window.location.hostname}:5001`;

// Elementos principais
const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const resultContainer = document.getElementById("result-container");
const classeSpan = document.getElementById("classe");
const confiancaSpan = document.getElementById("confianca");
const faixaSpan = document.getElementById("faixa");
const confidenceFill = document.getElementById("confidence-fill");
const interpretacaoEl = document.getElementById("interpretacao");
const tableBody = document.querySelector("#historyTable tbody");
let chartInstance = null;

// ===============================================================
// HELPERS
// ===============================================================
function showToast(msg, color = "#198754") {
  const toast = document.createElement("div");
  toast.textContent = msg;
  toast.style.cssText = `
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: ${color}; color: #fff; padding: 10px 16px; border-radius: 8px;
    box-shadow: 0 6px 18px rgba(0,0,0,.18); z-index: 99999; font-weight: 600;
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2400);
}

// Geolocalização opcional (não bloqueia análise se falhar)
function getGeolocation(timeoutMs = 5000) {
  return new Promise(resolve => {
    if (!("geolocation" in navigator)) {
      console.warn("Geolocalização indisponível neste dispositivo/navegador.");
      return resolve(null);
    }
    const opts = { enableHighAccuracy: false, timeout: timeoutMs, maximumAge: 30000 };
    navigator.geolocation.getCurrentPosition(
      pos => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      err => {
        console.warn("Sem geolocalização — continuando sem coordenadas.", err);
        resolve(null);
      },
      opts
    );
  });
}

// ===============================================================
// PRÉ-VISUALIZAÇÃO DA IMAGEM
// ===============================================================
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
  } else {
    preview.style.display = "none";
  }
});

// ===============================================================
// ENVIO DO ARQUIVO PARA ANÁLISE (com geo opcional)
// ===============================================================
form.addEventListener("submit", async e => {
  e.preventDefault();
  const file = fileInput.files[0];
  const button = form.querySelector("button");

  if (!file) {
    alert("Selecione uma imagem primeiro!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    // UI: bloqueia botão e reseta painel
    button.disabled = true;
    button.textContent = "Analisando...";
    resultContainer.style.display = "none";

    // Tenta capturar geolocalização (sem travar se não tiver)
    const geo = await getGeolocation(5000);
atualizarCoordenadas(geo);
    if (geo) {
      formData.append("latitude", String(geo.lat));
      formData.append("longitude", String(geo.lng));
    }

    await enviarImagem(formData, button);
  } catch (error) {
    console.error("Erro inesperado ao preparar análise:", error);
    showToast("Erro inesperado ao preparar análise", "#dc3545");
    button.textContent = "Erro ❌";
    setTimeout(() => (button.textContent = "Analisar"), 1600);
  } finally {
    button.disabled = false;
  }
});

// Envia para a API e trata resposta/erros
async function enviarImagem(formData, button) {
  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }

    mostrarResultado(data);
    await carregarHistorico();

    button.textContent = "Concluído ✅";
    showToast("Análise concluída com sucesso! ✅");
    setTimeout(() => (button.textContent = "Analisar"), 1400);
  } catch (error) {
    console.error("Erro na análise:", error);
    alert("Erro na análise: " + error.message);
    showToast("Falha na análise ❌", "#dc3545");
    button.textContent = "Erro ❌";
    setTimeout(() => (button.textContent = "Analisar"), 1400);
  }
}

// ===============================================================
// EXIBE O RESULTADO DA ANÁLISE
// ===============================================================
function mostrarResultado(data) {
  classeSpan.textContent = data.classe;
  confiancaSpan.textContent = data.confianca.toFixed(2) + "%";
  faixaSpan.textContent = data.faixa_confianca;
  // ===============================================================
// ALERTA AGRONÔMICO AUTOMÁTICO
// ===============================================================
const alertaEl = document.getElementById("alerta");

if (data.confianca >= 80 && data.classe !== "Sadia") {
  alertaEl.textContent = "🔴 ALERTA: Alta probabilidade de infestação detectada! Recomenda-se ação imediata de manejo e verificação em campo.";
  alertaEl.style.background = "rgba(220, 53, 69, 0.2)";
  alertaEl.style.borderLeft = "6px solid #dc3545";
  alertaEl.style.color = "#721c24";
} else if (data.confianca >= 60 && data.classe !== "Sadia") {
  alertaEl.textContent = "🟡 Atenção: Há sinais de possível infestação. Recomenda-se monitorar a área.";
  alertaEl.style.background = "rgba(255, 193, 7, 0.2)";
  alertaEl.style.borderLeft = "6px solid #ffc107";
  alertaEl.style.color = "#856404";
} else {
  alertaEl.textContent = "🟢 Sem risco identificado — planta aparentemente saudável.";
  alertaEl.style.background = "rgba(40, 167, 69, 0.15)";
  alertaEl.style.borderLeft = "6px solid #28a745";
  alertaEl.style.color = "#155724";
}
alertaEl.style.padding = "10px 15px";
alertaEl.style.borderRadius = "8px";
alertaEl.style.marginTop = "12px";
alertaEl.style.fontWeight = "500";

  resultContainer.style.display = "block";

  // Animação da barra de confiança
  if (confidenceFill) {
    confidenceFill.style.transition = "none";
    confidenceFill.style.width = "0%";
    setTimeout(() => {
      confidenceFill.style.transition = "width 0.8s ease";
      confidenceFill.style.width = `${data.confianca}%`;
    }, 60);

    // Cor por faixa de confiança
    if (data.confianca < 60) {
      confidenceFill.style.backgroundColor = "#dc3545"; // vermelho
    } else if (data.confianca < 85) {
      confidenceFill.style.backgroundColor = "#ffc107"; // amarelo
    } else {
      confidenceFill.style.backgroundColor = "#28a745"; // verde
    }
  }

  // Cor do texto da faixa
  if (data.faixa_confianca === "Alta") {
    faixaSpan.style.color = "#28a745";
  } else if (data.faixa_confianca === "Média") {
    faixaSpan.style.color = "#ffc107";
  } else {
    faixaSpan.style.color = "#dc3545";
  }

  // Interpretação automática
  let mensagem = "";
  if (data.classe === "Sadia") {
    mensagem = "🌿 Planta aparentemente saudável.";
  } else if (data.faixa_confianca === "Baixa") {
    mensagem = "⚠️ Detecção incerta — recomenda-se nova análise com outra imagem.";
  } else if (data.faixa_confianca === "Média") {
    mensagem = "🟡 Possível presença de sintomas. Monitorar de perto.";
  } else {
    mensagem = "🔴 Diagnóstico confiável de doença detectada.";
  }
  interpretacaoEl.textContent = mensagem;
}
function atualizarCoordenadas(geo) {
  const coordsEl = document.getElementById("coords");
  if (!coordsEl) return;

  if (geo) {
    coordsEl.textContent = `${geo.lat.toFixed(5)}, ${geo.lng.toFixed(5)}`;
  } else {
    coordsEl.textContent = "não disponível";
  }
}
// ===============================================================
// CARREGA HISTÓRICO DE ANÁLISES
// ===============================================================
async function carregarHistorico() {
  try {
    const response = await fetch(`${API_BASE}/results`);
    const historico = await response.json();
    popularTabela(historico);
    gerarGrafico(historico);
  } catch (error) {
    console.error("Erro ao carregar histórico:", error);
  }
}

// ===============================================================
// POPULA TABELA
// ===============================================================
function popularTabela(historico) {
  tableBody.innerHTML = "";
  historico.forEach(item => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.id}</td>
      <td>${item.classe}</td>
      <td>${item.confianca.toFixed(2)}%</td>
      <td>${item.nome_arquivo}</td>
      <td>${new Date(item.data).toLocaleString("pt-BR")}</td>
    `;
    tableBody.appendChild(row);
  });
}

// ===============================================================
// GERA O GRÁFICO DE DISTRIBUIÇÃO DE CLASSES
// ===============================================================
function gerarGrafico(historico) {
  const contagem = {};
  historico.forEach(item => {
    contagem[item.classe] = (contagem[item.classe] || 0) + 1;
  });

  const labels = Object.keys(contagem);
  const valores = Object.values(contagem);
  const cores = ["#28a745", "#dc3545", "#ffc107", "#007bff", "#6610f2", "#fd7e14", "#20c997"];

  const ctx = document.getElementById("classChart")?.getContext("2d");
  if (!ctx) return;

  if (chartInstance) chartInstance.destroy();

  chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Quantidade de Casos",
        data: valores,
        backgroundColor: cores.slice(0, labels.length),
      }],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true },
      },
    },
  });
}

// ===============================================================
// INICIALIZAÇÃO
// ===============================================================
carregarHistorico();






