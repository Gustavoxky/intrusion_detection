import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title="Consulta de Dados - Intrusion Detection", layout="wide", page_icon="🚨")
st.title("🚨 Sistema de Detecção de Intrusões 🚨")
st.markdown("Carregue um arquivo JSON com as **features** ou insira os dados manualmente para realizar uma consulta.")

st.sidebar.title("⚙️ Configurações")
api_url = st.sidebar.text_input("🔗 URL da API", value="http://127.0.0.1:8000/predict")
st.sidebar.write("⚠️ **Certifique-se de que a API está rodando antes de enviar os dados.**")

st.markdown("### 🛠️ Escolha o método de entrada de dados:")
input_method = st.radio(
    "Como você deseja inserir os dados?",
    ("📂 Carregar arquivo JSON", "✍️ Inserir dados manualmente"),
    index=0,
)

def send_request(features):
    try:
        response = requests.post(api_url, json={"features": features})
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def display_response(response_data):
    if isinstance(response_data, dict) and "prediction" in response_data:
        prediction = response_data["prediction"]
        if prediction.lower() == "intrusão":
            st.error(f"⚠️ **Predição: {prediction}**", icon="🚨")
        else:
            st.success(f"✅ **Predição: {prediction}**")
    elif isinstance(response_data, dict):
        st.json(response_data)
    elif isinstance(response_data, list):
        try:
            df = pd.DataFrame(response_data)
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.json(response_data)
    else:
        st.write(response_data)

if input_method == "📂 Carregar arquivo JSON":
    st.markdown("### 📂 **Carregar arquivo JSON**")
    uploaded_file = st.file_uploader("Carregue um arquivo JSON contendo as features", type=["json"])

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            features = data.get("features", [])

            if len(features) != 3460:
                st.error(f"❌ **Número incorreto de features: {len(features)}. Esperado: 3460.**")
            else:
                st.success("✅ **Arquivo carregado com sucesso!**")

                if st.button("🚀 Enviar para a API"):
                    st.info("Enviando dados para a API...")
                    result = send_request(features)
                    if result["success"]:
                        display_response(result["data"])
                    else:
                        st.error("❌ **Erro na consulta:**")
                        st.write(result["error"])
        except Exception as e:
            st.error("❌ **Erro ao processar o arquivo JSON.**")
            st.write(str(e))

elif input_method == "✍️ Inserir dados manualmente":
    st.markdown("### ✍️ **Inserir Dados Manualmente**")
    st.text_area("Insira os dados das features (3460 valores separados por vírgulas):", key="manual_input")

    manual_input = st.session_state.manual_input
    if manual_input:
        try:
            features = [float(x.strip()) for x in manual_input.split(",")]

            if len(features) != 3460:
                st.error(f"❌ **Número incorreto de features: {len(features)}. Esperado: 3460.**")
            else:
                st.success("✅ **Dados inseridos corretamente!**")

                if st.button("🚀 Enviar para a API"):
                    st.info("Enviando dados para a API...")
                    result = send_request(features)
                    if result["success"]:
                        display_response(result["data"])
                    else:
                        st.error("❌ **Erro na consulta:**")
                        st.write(result["error"])
        except ValueError:
            st.error("❌ **Erro: Certifique-se de inserir apenas números separados por vírgulas.**")

st.sidebar.subheader("ℹ️ Informações sobre o Sistema")
st.sidebar.write("- **Total de features esperadas:** 3460")
st.sidebar.write("- Desenvolvido com ❤️ por você!")
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 [Documentação da API](http://127.0.0.1:8000/docs)")
