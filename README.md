# 💎 StoneMatch – Classificação e Pareamento de Pedras Naturais com IA

## 📌 Sobre o Projeto
O **StoneMatch** é um sistema baseado em **Visão Computacional e Deep Learning** desenvolvido durante o evento **Code a Conquer – URI Erechim**.  

Nosso objetivo foi **automatizar a classificação e o pareamento de pedras semipreciosas**, substituindo um processo manual, demorado e sujeito a erros.  
Com a solução proposta, conseguimos identificar automaticamente cores, formatos e padrões das pedras.  

---

## 🚀 Funcionalidades
- 📤 **Upload de imagens** contendo várias pedras ao mesmo tempo.  
- ✂️ **Segmentação automática** e corte individual de cada pedra detectada.  
- 🎨 **Classificação por cor** (verde, marrom, vermelha, azul, etc.).  
- 🔍 **Extração de características visuais**:
  - Cores dominantes (painel de cores gerado automaticamente).  
  - Presença de listras/padrões na pedra.  
  - Dimensões e área em pixels.  
- 🖼️ **Imagem anotada** com identificação de cada pedra.  
- 📊 **Relatório em CSV** com todas as informações processadas.  
- 📦 **Exportação de imagens recortadas** e opção de **limpar resultados** pelo painel.  

---

## 🧠 Tecnologias Utilizadas
**Backend e Processamento**
- Python 3.13.7 
- Flask – API e interface web  
- OpenCV – Processamento e segmentação de imagens  
- PyTorch (ResNet50) – Extração de embeddings visuais  
- scikit-learn (KMeans) – Agrupamento de cores dominantes  
- Pandas – Estruturação e exportação de relatórios  

**Frontend**
- HTML5 + CSS3 (responsivo e estilizado)  
- JavaScript (renderização dinâmica e interações no painel)  

---

## 📂 Estrutura do Projeto
```

StoneMatch/
│── app.py              # Aplicação Flask
│── build\_index.py      # Indexação da galeria de pedras
│── config.yaml         # Arquivo de configuração
│── requirements.txt    # Dependências do projeto
│
├── templates/
│   └── index.html      # Interface principal
│
├── static/
│   ├── style.css       # Estilos
│   ├── script.js       # Lógica do frontend
│   ├── crops/          # Pedras recortadas
│   ├── matches/        # Matches gerados
│
├── outputs/
│   ├── report.csv      # Relatório final
│   └── index.npy/json  # Dados indexados
│
└── gallery/            # Galeria de imagens de referência

````

---

## ⚙️ Instalação
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/StoneMatch.git
   cd StoneMatch
````

2. Crie e ative o ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure o arquivo `config.yaml` conforme sua galeria de imagens.

---

## ▶️ Uso

1. **Construa o índice da galeria**:

   ```bash
   python build_index.py
   ```

2. **Inicie a aplicação Flask**:

   ```bash
   python app.py
   ```

3. Abra no navegador: [http://localhost:5000](http://localhost:5000)

---

## 📊 Exemplo de Resultados

* **Imagem original anotada** com cada pedra numerada.

* **Cards interativos** com:

  * Preview da pedra recortada.
  * Cores dominantes em painel.
  * Área em pixels.
  * Presença de listras/padrões.

* **Resumo totalizador**:

  * Número de pedras detectadas.
  * Totais por grupo de cor.
  * Área média de cada grupo.

---

## 👥 Equipe

Projeto desenvolvido durante o **Code a Conquer – URI Erechim**.

👥 Integrantes do grupo:
Ariel Angonese | Eduardo Ongaratto | Esther Kossmann | Felipe Julhão | Igor Bernardi | Kauane Zicato | Lucas Oliveira | Luiz Felipe Buaszczyk | Rafael Gustavo Kammler

---

## 📌 Licença

Este projeto foi desenvolvido para fins acadêmicos e demonstrativos.
Sinta-se à vontade para estudar, modificar e expandir. 🚀

```