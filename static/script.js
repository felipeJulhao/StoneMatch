document.addEventListener("DOMContentLoaded", function () {
    // Elementos do DOM
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

    // Variáveis de estado
    let currentPage = 1;
    const itemsPerPage = 6;
    let allStones = [];
    let filteredStones = [];

    // Dados de exemplo (substituir pelos dados reais do backend)
    const sampleData = [
        { id: 1, type: "Quartzo", color: "Branco", width: 15.2, height: 10.5, area: 159.6, confidence: 0.95 },
        { id: 2, type: "Granito", color: "Cinza", width: 12.8, height: 8.3, area: 106.2, confidence: 0.87 },
        { id: 3, type: "Ardosia", color: "Preto", width: 9.7, height: 7.1, area: 68.9, confidence: 0.92 },
        { id: 4, type: "Quartzo", color: "Rosa", width: 18.4, height: 12.6, area: 231.8, confidence: 0.78 },
        { id: 5, type: "Mármore", color: "Bege", width: 14.3, height: 9.8, area: 140.1, confidence: 0.85 },
        { id: 6, type: "Granito", color: "Verde", width: 11.5, height: 8.9, area: 102.4, confidence: 0.91 },
        { id: 7, type: "Quartzo", color: "Cinza", width: 16.7, height: 11.2, area: 187.0, confidence: 0.88 },
        { id: 8, type: "Ardosia", color: "Azul", width: 13.4, height: 10.1, area: 135.3, confidence: 0.93 }
    ];

    // Inicialização
    allStones = sampleData;
    filteredStones = [...allStones];

    // Event Listeners
    if (imageInput) {
        imageInput.addEventListener('change', handleFileSelect);
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }

    if (sortSelect) {
        sortSelect.addEventListener('change', applySortingAndFiltering);
    }

    if (filterSelect) {
        filterSelect.addEventListener('change', applySortingAndFiltering);
    }

    if (prevPageBtn) {
        prevPageBtn.addEventListener('click', goToPrevPage);
    }

    if (nextPageBtn) {
        nextPageBtn.addEventListener('click', goToNextPage);
    }

    // Funções
    function handleFileSelect() {
        if (imageInput.files && imageInput.files[0]) {
            processBtn.disabled = false;

            // Pré-visualização da imagem (opcional)
            const reader = new FileReader();
            reader.onload = function(e) {
                // Aqui você poderia mostrar uma prévia se desejar
            };
            reader.readAsDataURL(imageInput.files[0]);
        }
    }

    function handleFormSubmit(e) {
        e.preventDefault();

        if (!imageInput.files || !imageInput.files[0]) {
            alert('Por favor, selecione uma imagem primeiro.');
            return;
        }

        // Mostrar loading
        showLoading();

        // Simular processamento (substituir por chamada AJAX real)
        setTimeout(() => {
            // Em uma implementação real, você faria uma requisição para o backend
            // e processaria a resposta aqui

            // Para demonstração, usamos dados de exemplo
            processedImage.src = "https://via.placeholder.com/800x500/2c7be5/ffffff?text=Imagem+Processada";
            renderStones();
            resultsContainer.classList.remove('hidden');

            hideLoading();
        }, 2000);
    }

    function renderStones() {
        resultsGrid.innerHTML = '';

        // Calcular itens para a página atual
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const stonesToShow = filteredStones.slice(startIndex, endIndex);

        if (stonesToShow.length === 0) {
            resultsGrid.innerHTML = '<div class="no-results">Nenhum resultado encontrado</div>';
            return;
        }

        stonesToShow.forEach(stone => {
            const stoneCard = createStoneCard(stone);
            resultsGrid.appendChild(stoneCard);
        });

        updatePagination();
    }

    function createStoneCard(stone) {
        const card = document.createElement('div');
        card.className = 'stone-card';

        // Adicionar classe premium para pedras com alta confiança
        if (stone.confidence > 0.9) {
            card.classList.add('premium');
        }

        // Determinar classe de confiança
        let confidenceClass = 'medium-confidence';
        if (stone.confidence >= 0.9) {
            confidenceClass = 'high-confidence';
        } else if (stone.confidence < 0.7) {
            confidenceClass = 'low-confidence';
        }

        card.innerHTML = `
            <div class="card-header">
                <div class="stone-id">Pedra #${stone.id}</div>
                <div class="confidence ${confidenceClass}">
                    <i class="fas fa-certificate"></i>
                    ${(stone.confidence * 100).toFixed(1)}%
                </div>
            </div>
            <div class="card-body">
                <div class="stone-info">
                    <span class="info-label">Tipo:</span>
                    <span class="info-value">${stone.type}</span>
                </div>
                <div class="stone-info">
                    <span class="info-label">Cor:</span>
                    <span class="info-value">${stone.color}</span>
                </div>
                <div class="stone-info">
                    <span class="info-label">Dimensões:</span>
                    <div class="dimensions">
                        <div class="dimension">
                            <div class="dimension-label">Largura</div>
                            <div class="dimension-value">${stone.width}cm</div>
                        </div>
                        <div class="dimension">
                            <div class="dimension-label">Altura</div>
                            <div class="dimension-value">${stone.height}cm</div>
                        </div>
                    </div>
                </div>
                <div class="stone-info">
                    <span class="info-label">Área:</span>
                    <span class="info-value">${stone.area}cm²</span>
                </div>
            </div>
        `;

        return card;
    }

    function applySortingAndFiltering() {
        const sortBy = sortSelect.value;
        const filterBy = filterSelect.value;

        // Aplicar filtro
        if (filterBy === 'all') {
            filteredStones = [...allStones];
        } else {
            filteredStones = allStones.filter(stone =>
                stone.type.toLowerCase() === filterBy
            );
        }

        // Aplicar ordenação
        switch(sortBy) {
            case 'confidence':
                filteredStones.sort((a, b) => b.confidence - a.confidence);
                break;
            case 'area':
                filteredStones.sort((a, b) => b.area - a.area);
                break;
            case 'id':
                filteredStones.sort((a, b) => a.id - b.id);
                break;
        }

        // Reset para a primeira página
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
        const maxPage = Math.ceil(filteredStones.length / itemsPerPage);

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

    if (allStones.length > 0) {
        renderStones();
    }
});