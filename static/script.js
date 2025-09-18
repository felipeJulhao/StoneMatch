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

    // --------------------------
    // Estado da aplicação
    // --------------------------
    let allStones = [];
    let filteredStones = [];
    let currentPage = 1;
    const itemsPerPage = 6;

    // --------------------------
    // Event Listeners
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
        if (imageInput.files && imageInput.files[0]) {
            processBtn.disabled = false;
        }
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
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Erro na requisição: ${response.statusText}`);
            }

            const data = await response.json();

            // Atualiza o estado da aplicação com os dados recebidos
            allStones = data.stones;
            updateFilterOptions(allStones);
            applySortingAndFiltering(); // Aplica o sort/filtro inicial

            // Exibe os resultados
            processedImage.src = data.annotated_image_url;
            resultsContainer.classList.remove('hidden');

        } catch (error) {
            console.error("Falha ao processar a imagem:", error);
            alert("Ocorreu um erro ao processar a imagem. Tente novamente.");
        } finally {
            hideLoading();
        }
    }

    function updateFilterOptions(stones) {
        const types = [...new Set(stones.map(s => s.type))]; // Pega tipos únicos
        filterSelect.innerHTML = '<option value="all">Todos os tipos</option>'; // Reseta
        types.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            // Capitaliza a primeira letra para ficar mais bonito
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
            resultsGrid.innerHTML = '<p class="no-results">Nenhum resultado encontrado para esta seleção.</p>';
            updatePagination();
            return;
        }

        stonesToShow.forEach(stone => {
            const card = createStoneCard(stone);
            resultsGrid.appendChild(card);
        });

        updatePagination();
    }

    function createStoneCard(stone) {
        const card = document.createElement('div');
        card.className = 'stone-card';

        let confidenceClass = 'medium-confidence';
        if (stone.confidence >= 0.85) confidenceClass = 'high-confidence';
        else if (stone.confidence < 0.6) confidenceClass = 'low-confidence';

        // Usando os dados que o backend agora fornece
        card.innerHTML = `
            <div class="card-header">
                <div class="stone-id">Pedra #${stone.id}</div>
                <div class="confidence ${confidenceClass}">${(stone.confidence * 100).toFixed(1)}%</div>
            </div>
            <div class="card-body">
                <div class="stone-info">
                    <span class="info-label">Tipo:</span> ${stone.type}
                </div>
                <div class="stone-info">
                    <span class="info-label">Melhor Match:</span> ${stone.best_match_file.split('/').pop()}
                </div>
                 <div class="stone-info">
                    <span class="info-label">Área:</span> ${stone.area} px²
                </div>
                <div class="stone-info">
                    <span class="info-label">Distância:</span> ${stone.distance}
                </div>
            </div>
        `;
        return card;
    }

    function applySortingAndFiltering() {
        const sortBy = sortSelect.value;
        const filterBy = filterSelect.value;

        filteredStones = filterBy === 'all'
            ? [...allStones]
            : allStones.filter(stone => stone.type === filterBy);

        switch (sortBy) {
            case 'confidence': filteredStones.sort((a, b) => b.confidence - a.confidence); break;
            case 'area': filteredStones.sort((a, b) => b.area - a.area); break;
            case 'id': filteredStones.sort((a, b) => a.id - b.id); break;
        }

        currentPage = 1;
        renderStones();
    }

    function goToPrevPage() {
        if (currentPage > 1) {
            currentPage--;
            renderStones();
        }
    }

    function goToNextPage() {
        const maxPage = Math.ceil(filteredStones.length / itemsPerPage);
        if (currentPage < maxPage) {
            currentPage++;
            renderStones();
        }
    }

    function updatePagination() {
        const maxPage = Math.ceil(filteredStones.length / itemsPerPage) || 1;
        pageInfo.textContent = `Página ${currentPage} de ${maxPage}`;
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = currentPage === maxPage;
    }

    function showLoading() {
        loadingOverlay.classList.remove('hidden');
    }

    function hideLoading() {
        loadingOverlay.classList.add('hidden');
    }
});