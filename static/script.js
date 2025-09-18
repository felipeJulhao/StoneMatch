document.addEventListener("DOMContentLoaded", function () {
    // --------------------------
    // Elementos do DOM
    // --------------------------
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('image');
    const processBtn = document.getElementById('processBtn');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsGrid = document.getElementById('resultsGrid');
    const processedImage = document.getElementById('processedImage');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const sortSelect = document.getElementById('sortSelect');
    const filterSelect = document.getElementById('filterSelect');
    const prevPageBtn = document.getElementById('prevPage');
    const nextPageBtn = document.getElementById('nextPage');
    const pageInfo = document.getElementById('pageInfo');
    const summaryContainer = document.createElement("div");
    summaryContainer.className = "summary-card";
    resultsContainer.appendChild(summaryContainer);

    // --------------------------
    // Estado
    // --------------------------
    let allStones = [];
    let filteredStones = [];
    let summaryData = null;
    let currentPage = 1;
    const itemsPerPage = 6;

    // --------------------------
    // Listeners
    // --------------------------
    imageInput.addEventListener('change', handleFileSelect);
    uploadForm.addEventListener('submit', handleFormSubmit);
    sortSelect.addEventListener('change', applySortingAndFiltering);
    filterSelect.addEventListener('change', applySortingAndFiltering);
    prevPageBtn.addEventListener('click', goToPrevPage);
    nextPageBtn.addEventListener('click', goToNextPage);

    // --------------------------
    // Funções
    // --------------------------
    function handleFileSelect() {
        processBtn.disabled = !(imageInput.files && imageInput.files[0]);
    }

    async function handleFormSubmit(e) {
        e.preventDefault();
        if (!imageInput.files || !imageInput.files[0]) {
            alert('Por favor, selecione uma imagem primeiro.');
            return;
        }

        showLoading();
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);

        try {
            const response = await fetch('/api/process', { method: 'POST', body: formData });
            if (!response.ok) throw new Error(`Erro na requisição: ${response.statusText}`);

            const data = await response.json();
            allStones = data.stones;
            summaryData = data.summary;

            updateFilterOptions(allStones);
            applySortingAndFiltering();
            processedImage.src = data.annotated_image_url;
            resultsContainer.classList.remove('hidden');
            renderSummary(summaryData);

        } catch (error) {
            console.error("Falha ao processar:", error);
            alert("Erro ao processar a imagem. Tente novamente.");
        } finally {
            hideLoading();
        }
    }

    function updateFilterOptions(stones) {
        const types = [...new Set(stones.map(s => s.type))];
        filterSelect.innerHTML = '<option value="all">Todos os tipos</option>';
        types.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type.charAt(0).toUpperCase() + type.slice(1);
            filterSelect.appendChild(option);
        });
    }

    function renderStones() {
        resultsGrid.innerHTML = '';
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const stonesToShow = filteredStones.slice(startIndex, endIndex);

        if (stonesToShow.length === 0) {
            resultsGrid.innerHTML = '<p class="no-results">Nenhum resultado encontrado.</p>';
            updatePagination();
            return;
        }

        stonesToShow.forEach(stone => resultsGrid.appendChild(createStoneCard(stone)));
        updatePagination();
    }

    function createStoneCard(stone) {
        const card = document.createElement('div');
        card.className = 'stone-card';

        let confidenceClass = 'medium-confidence';
        if (stone.confidence >= 0.85) confidenceClass = 'high-confidence';
        else if (stone.confidence < 0.6) confidenceClass = 'low-confidence';

        // Painel de cores dominantes
        let coresHtml = '';
        if (stone.cores_dominantes && stone.cores_dominantes.length > 0) {
            coresHtml = `<div class="color-panel">`;
            stone.cores_dominantes.forEach(cor => {
                const rgb = cor.match(/\d+/g); // extrai [R,G,B]
                if (rgb) {
                    const cssColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
                    coresHtml += `<div class="color-box" style="background:${cssColor}" title="${cssColor}"></div>`;
                }
            });
            coresHtml += `</div>`;
        }

        card.innerHTML = `
        <div class="card-header">
            <div class="stone-id">Pedra #${stone.id}</div>
            <div class="confidence ${confidenceClass}">${(stone.confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="card-body">
            <div class="stone-preview">
                <img src="${stone.crop_url}" alt="Pedra ${stone.id}" style="max-width:100px; border-radius:6px;">
            </div>
            <div class="stone-info">
                <span class="info-label">Grupo:</span> ${stone.type}
            </div>
            <div class="stone-info">
                <span class="info-label">Área:</span> ${stone.area} px²
            </div>
            <div class="stone-info">
                <span class="info-label">Tem listras?</span> ${stone.tem_listras ? "Sim" : "Não"}
            </div>
            <div class="stone-info">
                <span class="info-label">Cores dominantes:</span>
                ${coresHtml}
            </div>
        </div>
    `;
        return card;
    }

    function renderSummary(summary) {
        if (!summary) {
            summaryContainer.innerHTML = '';
            return;
        }

        let html = `
        <h2><i class="fas fa-chart-pie"></i> Resumo</h2>
        <p><strong>Total de pedras:</strong> ${summary.total_pedras}</p>
        <ul class="summary-list">
    `;

        summary.groups.forEach(group => {
            html += `
            <li>
                <span class="summary-group">${group.group}:</span> 
                ${group.total} pedras 
                (área média: ${group.avg_area}px²)
            </li>
        `;
        });

        html += `</ul>`;
        summaryContainer.innerHTML = html;
    }

    function applySortingAndFiltering() {
        const sortBy = sortSelect.value;
        const filterBy = filterSelect.value;

        filteredStones = filterBy === 'all' ? [...allStones] : allStones.filter(s => s.type === filterBy);
        if (sortBy === 'confidence') filteredStones.sort((a, b) => b.confidence - a.confidence);
        else if (sortBy === 'area') filteredStones.sort((a, b) => b.area - a.area);
        else filteredStones.sort((a, b) => a.id - b.id);

        currentPage = 1;
        renderStones();
    }

    function goToPrevPage() { if (currentPage > 1) { currentPage--; renderStones(); } }
    function goToNextPage() { if (currentPage < Math.ceil(filteredStones.length / itemsPerPage)) { currentPage++; renderStones(); } }
    function updatePagination() {
        const maxPage = Math.ceil(filteredStones.length / itemsPerPage) || 1;
        pageInfo.textContent = `Página ${currentPage} de ${maxPage}`;
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = currentPage === maxPage;
    }

    function showLoading() { loadingOverlay.classList.remove('hidden'); }
    function hideLoading() { loadingOverlay.classList.add('hidden'); }

    window.exportImages = async function () {
        if (!confirm("Deseja exportar as imagens?")) return;
        window.location.href = "/export_images";
    }

    window.clearImages = async function () {
        if (!confirm("Deseja realmente limpar as imagens?")) return;
        await fetch("/clear_outputs", { method: "POST" });
        alert("Imagens apagadas.");
        location.reload();
    }
});
