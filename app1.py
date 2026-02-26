import streamlit as st

# Iluasjad: pealkiri, alapealkiri
st.title("🎓 AI Kursuse Nõustaja")
st.caption("Lihtne vestlusliides automaatvastusega.")

# 1. Algatame vestluse ajaloo, kui seda veel pole
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Kuvame vestluse senise ajaloo (History)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Korjame üles uue kasutaja sisendi (Action)
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    # Kuvame kohe kasutaja sõnumi
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Salvestame kasutaja sõnumi ajalukku
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Vastuse genereerimine
    response = f"Sinu küsimus oli: '{prompt}'. LLM pole veel ühendatud, see on automaatvastus."

    # Kuvame vastuse
    with st.chat_message("assistant"):
        st.markdown(response)
        
    # Salvestame vastuse ajalukku
    st.session_state.messages.append({"role": "assistant", "content": response})