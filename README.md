# 🚨 Sistema de Detecção de Intrusões com XGBoost 🚨

Este projeto implementa um sistema de detecção de intrusões baseado em aprendizado de máquina utilizando **XGBoost**. A solução foi projetada para treinar, testar e implantar um modelo de detecção de intrusões, com suporte para APIs e interface gráfica interativa via Streamlit.

---

## 🛠️ Instalação

### Pré-requisitos

- **Python 3.8 ou superior**
- **Pandas**, **NumPy**, **Scikit-learn**, **XGBoost**, **Imbalanced-learn**, **CuPy**
- **FastAPI** e **Streamlit**

### Passo a passo

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu_usuario/sistema-detecao-intrusoes.git
   cd sistema-detecao-intrusoes
   ```

2. Crie e ative um ambiente virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:

   ```bash
   pip install -r app/requirements.txt
   ```

---

## 🧠 Treinamento do Modelo

O script `main.py` realiza as seguintes etapas:

1. **Pré-processamento**:
   - Conversão de variáveis categóricas para one-hot encoding.
   - Normalização dos dados usando `MinMaxScaler`.

2. **Balanceamento de Classes**:
   - Utilização de SMOTE para lidar com classes desbalanceadas.

3. **Treinamento com XGBoost**:
   - O modelo é treinado utilizando lotes para otimizar a memória GPU (via CuPy).

4. **Salvamento do Modelo**:
   - O modelo treinado e o escalador são salvos no diretório `model/`.

### Executar o treinamento:

```bash
python main.py
```

---

## 🔬 Teste do Modelo

Utilize o script `test_model.py` para avaliar o modelo treinado em novos dados:

```bash
python test_model.py
```

O script gera:
- Relatório de classificação
- Matriz de confusão
- As 10 features mais importantes do modelo

---

## 🌐 API com FastAPI

O backend está implementado em FastAPI para realizar previsões em tempo real.

### Executar o servidor FastAPI:

1. Execute o servidor:
   ```bash
   uvicorn main:app --reload
   ```

2. Acesse a documentação interativa:
   - [Swagger UI](http://127.0.0.1:8000/docs)

### Endpoints principais:

- **POST /predict**: Recebe dados das features e retorna a predição (`Normal` ou `Intrusão`).

---

## 💻 Interface Gráfica com Streamlit

O frontend interativo permite carregar arquivos JSON ou inserir manualmente os dados das features.

### Executar o Streamlit:

1. No diretório principal, execute:
   ```bash
   streamlit run app.py
   ```

2. Acesse no navegador:
   - [http://localhost:8501](http://localhost:8501)

### Funcionalidades do Streamlit:

- Carregar arquivos JSON com as features.
- Inserir dados manualmente.
- Enviar dados para a API e visualizar o resultado da predição.

---

## 🛡️ Testando o Sistema

### Pré-requisitos para o teste:

- Certifique-se de que o servidor FastAPI está rodando.
- Configure a URL da API no Streamlit para apontar para o servidor FastAPI.

### Passo a passo:

1. Abra o Streamlit e selecione o método de entrada.
2. Envie os dados e visualize o resultado da predição diretamente na interface.

---

## 📊 Principais Ferramentas e Tecnologias

- **XGBoost**: Modelo de aprendizado de máquina.
- **CuPy**: Computação eficiente na GPU.
- **SMOTE**: Balanceamento de classes.
- **FastAPI**: Backend para consumo do modelo.
- **Streamlit**: Frontend interativo para consultas.

---

## 🖥️ Deployment

### Rodando em Produção

1. Gere os modelos treinados no ambiente de desenvolvimento.
2. Configure o ambiente de produção com as mesmas dependências.
3. Inicie o servidor FastAPI e o Streamlit.

Para um ambiente mais robusto, utilize **Docker** e orquestre com **Kubernetes** se necessário.

---

## 🧾 Licença

Este projeto é disponibilizado sob a licença [MIT](LICENSE).

---

Desenvolvido com ❤️ por [Seu Nome].
