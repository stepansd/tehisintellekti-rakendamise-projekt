import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎓 AI Kursuse Nõustaja - Samm 5")
st.caption("RAG süsteem koos eel-filtreerimisega.")

with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password", key="api_key")
    st.info("Selles versioonis on koodis filter: ainult ingliskeelsed kursused.")

@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df

embedder, df, embeddings_df = get_models()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Võtame aadressi session_state'ist, et see ei läheks rerun'il kaotsi
        current_api_key = st.session_state.get("api_key", "").strip()
        st.write(f"DEBUG: ключ = '{current_api_key[:10]}...' длина = {len(current_api_key)}")

        if not current_api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Otsin sobivaid kursusi..."):
                merged_df = pd.merge(df, embeddings_df, on='unique_ID')
                mask = ((merged_df['semester'] == 'kevad') & (merged_df["eap"] == 6))
                filtered_df = merged_df[mask].copy()

                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vasta filtritele.")
                    context_text = "Sobivaid kursusi ei leitud."
                else:
                    query_vec = embedder.encode([prompt])[0]
                    filtered_df['score'] = cosine_similarity([query_vec], np.stack(filtered_df['embedding']))[0]
                    results_N = 5
                    results_df = filtered_df.sort_values('score', ascending=False).head(results_N)
                    results_df.drop(['score', 'embedding'], axis=1, inplace=True)
                    context_text = results_df.to_string()

                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=current_api_key,
                    default_headers={"HTTP-Referer": "http://localhost:8501"}
                )

                system_prompt = {
                    "role": "system",
                    "content": f"Oled nõustaja. Kasuta järgmisi kursusi:\n\n{context_text}"
                }

                messages_to_send = [system_prompt] + st.session_state.messages

                try:
                    stream = client.chat.completions.create(
                        model="google/gemma-3-27b-it",
                        messages=messages_to_send,
                        stream=True
                    )
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Viga: {e}")