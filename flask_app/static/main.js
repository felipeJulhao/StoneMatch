const video = document.getElementById("video");
const captureBtn = document.getElementById("capture");
const resultsDiv = document.getElementById("results");
const inserirBtn = document.getElementById("btn-inserir");

const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.accept = "image/*";
fileInput.style.display = "none";
document.body.appendChild(fileInput);

// Captura da câmera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error("Erro ao acessar câmera:", err));

// Função para enviar imagem para o Flask
async function sendImageToServer(imageData) {
    try {
        const response = await fetch("/upload", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData })
        });
        const data = await response.json();
        console.log(data);
        resultsDiv.textContent = data.message || JSON.stringify(data, null, 2);
    } catch (err) {
        console.error("Erro no upload:", err);
        resultsDiv.textContent = "Erro ao enviar imagem.";
    }
}

// Evento de captura da câmera
captureBtn.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/png");
    sendImageToServer(imageData);
});

inserirBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => sendImageToServer(reader.result);
    reader.onerror = () => resultsDiv.textContent = "Erro ao ler o arquivo.";
    reader.readAsDataURL(file);

    event.target.value = ''; // reseta input
});