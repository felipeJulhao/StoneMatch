async function postJSON(url, body) {
  const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})});
  return r.json();
}

const resultsDiv = document.getElementById("results");
const kInput     = document.getElementById("topk");
const ingestBtn  = document.getElementById("btn-ingest");
const ingestMsg  = document.getElementById("ingest-status");
const capBtn     = document.getElementById("btn-capture");

ingestBtn.addEventListener("click", async () => {
  ingestMsg.textContent = "Processando...";
  try {
    const js = await postJSON("/ingest", {});
    ingestMsg.textContent = js.status === "ok" ? "Índice pronto ✅" : JSON.stringify(js);
  } catch(e) {
    ingestMsg.textContent = "Erro: " + e;
  }
});

capBtn.addEventListener("click", async () => {
  resultsDiv.innerHTML = "Capturando e pareando...";
  const k = Math.max(1, Math.min(20, parseInt(kInput.value || "5", 10)));
  try {
    const js = await postJSON("/snapshot_match", {k});
    if (js.error) { resultsDiv.textContent = js.error; return; }
    if (!js.topk || js.topk.length===0) { resultsDiv.textContent = "Sem resultados."; return; }

    resultsDiv.innerHTML = "";
    js.topk.forEach((r, i) => {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `
        <img src="${r.url}" alt="${r.id}">
        <div class="meta">
          <div class="rank">#${i+1}</div>
          <div class="id">${r.id}</div>
          <div class="score">score: ${r.score.toFixed(3)}</div>
        </div>`;
      resultsDiv.appendChild(card);
    });
  } catch (e) {
    resultsDiv.textContent = "Erro: " + e;
  }
});
