import streamlit as st
from openai import OpenAI

# 1. Lehe seadistamine
st.set_page_config(page_title="AI Kursuse Noustaja", page_icon="🎓")

st.title("🎓 AI Kursuse Nõustaja")
st.caption("Mudel: Google Gemma 3 27B IT (OpenRouter)")

# 2. KÜLGRIBA
with st.sidebar:
    st.header("Seaded")
    api_key = st.text_input("OpenRouter API Key", type="password")
    
    st.divider()
    if st.button("Puhasta vestlus"):
        st.session_state.messages = []
        st.rerun()

# --- MUDELI ID MÄÄRAMINE ---
# OpenRouteris on Gemma 3 27B IT nimi tavaliselt selline:
MODEL_ID = "google/gemma-3-27b" 

# 3. VESTLUSE AJALOO ALGATAMINE
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. SENISE VESTLUSE KUVAMINE
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. KASUTAJA SISEND JA AI VASTUS
if prompt := st.chat_input("Kirjuta siia oma küsimus..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not api_key:
            st.warning("Palun lisa külgribale oma OpenRouter API võti!")
        else:
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )

                stream = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "AI Kursuse Noustaja App",
                    },
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": "Sa oled asjatundlik AI-kursuste nõustaja. Vasta alati eesti keeles."}
                    ] + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )

                full_response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                # Kui OpenRouter ütleb, et mudelit ei leitud, proovi MODEL_ID = "google/gemma-3-27b:free"
                st.error(f"Viga: {str(e)}")