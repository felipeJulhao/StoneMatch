const video = document.getElementById("video");
const captureBtn = document.getElementById("capture");
const responseBox = document.getElementById("response");
const resultsDiv = document.getElementById("results");
const kInput = document.getElementById("topk");
const ingestBtn = document.getElementById("btn-ingest");
const ingestMsg = document.getElementById("ingest-status");
const inserirBtn = document.getElementById("btn-inserir");

// Função auxiliar para requisições POST
async function postJSON(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    if (!r.ok) throw new Error(`HTTP error! status: ${r.status}`);
    return r.json();
}


navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

captureBtn.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL("image/png");

    fetch("/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
    })
    .then(res => res.json())
    .then(data => {
        responseBox.textContent = JSON.stringify(data, null, 2);
    });
});

inserirBtn.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();

    reader.onload = async (e) => {
        const imageData = e.target.result;
        await performMatch(imageData);
    };

    reader.onerror = () => {
        resultsDiv.textContent = "Erro ao ler o arquivo selecionado.";
    };

    reader.readAsDataURL(file);

    event.target.value = '';
});


ingestBtn.addEventListener("click", async () => {
    ingestMsg.textContent = "Processando...";
    ingestBtn.disabled = true;
    try {
        const js = await postJSON("/ingest", {});
        ingestMsg.textContent = js.status === "ok" ? "Índice pronto" : JSON.stringify(js);
    } catch (e) {
        ingestMsg.textContent = "Erro: " + e;
    } finally {
        ingestBtn.disabled = false;
    }
});

const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.accept = "image/*";
fileInput.style.display = "none";



