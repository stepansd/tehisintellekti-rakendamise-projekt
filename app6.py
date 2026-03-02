import streamlit as st
import pandas as pd
import numpy as np
import csv
import os
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# TAGASISIDE LOGIMISE FUNKTSIOON
# ─────────────────────────────────────────────
def log_feedback(timestamp, prompt, filters, context_ids, context_names, response, rating, error_category):
    file_path = 'tagasiside_log.csv'
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Aeg', 'Kasutaja päring', 'Filtrid', 'Leitud ID-d', 'Leitud ained', 'LLM Vastus', 'Hinnang', 'Veatüüp'])
        writer.writerow([timestamp, prompt, filters, str(context_ids), str(context_names), response, rating, error_category])

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# ─────────────────────────────────────────────
# MUDEL JA ANDMED
# ─────────────────────────────────────────────
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df

embedder, df, embeddings_df = get_models()

st.title("🎓 AI Kursuse Nõustaja - Samm 6")
st.caption("RAG süsteem metaandmete filtreerimisega (Variant B) + vigade analüüs")

# ─────────────────────────────────────────────
# KÜLGRIBA
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Seaded")
    api_key = st.text_input("OpenRouter API Key", type="password", key="api_key")

    st.divider()
    st.header("🔍 Metaandmete filtrid")
    st.caption("Jäta '(kõik)' kui pole oluline")

    semesters  = ["(kõik)"] + sorted(df["semester"].dropna().unique().tolist())
    eap_values = ["(kõik)"] + [str(int(e)) for e in sorted(df["eap"].dropna().unique().tolist())]
    levels     = ["(kõik)"] + sorted(df["oppeaste"].dropna().unique().tolist())
    languages  = ["(kõik)"] + sorted(df["keel"].dropna().unique().tolist())
    exam_opts  = ["(kõik)"] + sorted(df["hindamisviis"].dropna().unique().tolist())

    sel_semester = st.selectbox("📅 Semester", semesters)
    sel_eap      = st.selectbox("📊 Ainepunktid (EAP)", eap_values)
    sel_level    = st.selectbox("🎓 Õppeaste", levels)
    sel_lang     = st.selectbox("🌐 Õppekeel", languages)
    sel_exam     = st.selectbox("✏️ Hindamisviis", exam_opts)
    n_results    = st.slider("Tulemuste arv", 1, 10, 5)

    st.divider()
    st.header("💰 Kulu jälgimine")

    PRICES = {
        "google/gemma-3-27b-it":    {"input": 0.10, "output": 0.20},
        "openai/gpt-4o-mini":       {"input": 0.15, "output": 0.60},
        "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    }
    model_name = st.selectbox("Mudel", list(PRICES.keys()))

    col1, col2 = st.columns(2)
    col1.metric("Sisend", f"{st.session_state.total_input_tokens} tok")
    col2.metric("Väljund", f"{st.session_state.total_output_tokens} tok")
    st.metric("Jooksev kulu", f"${st.session_state.total_cost:.6f}")

    if st.button("🗑️ Tühjenda vestlus"):
        st.session_state.messages = []
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

# ─────────────────────────────────────────────
# AKTIIVSED FILTRID
# ─────────────────────────────────────────────
active = []
if sel_semester != "(kõik)": active.append(f"Semester: **{sel_semester}**")
if sel_eap      != "(kõik)": active.append(f"EAP: **{sel_eap}**")
if sel_level    != "(kõik)": active.append(f"Õppeaste: **{sel_level}**")
if sel_lang     != "(kõik)": active.append(f"Keel: **{sel_lang}**")
if sel_exam     != "(kõik)": active.append(f"Hindamisviis: **{sel_exam}**")

if active:
    st.info("🔎 Aktiivsed filtrid: " + " · ".join(active))
else:
    st.info("🔎 Filtrid puuduvad — otsitakse kõikide kursuste hulgast")

# ─────────────────────────────────────────────
# VESTLUSE AJALUGU + KAPOTT + TAGASISIDE
# ─────────────────────────────────────────────
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "debug_info" in message:
            debug = message["debug_info"]

            # Kapoti alla vaatamine
            with st.expander("🔍 Vaata kapoti alla (RAG ja filtrid)"):
                st.caption(f"**Aktiivsed filtrid:** {debug.get('filters', '—')}")
                st.write(f"Filtrid jätsid alles **{debug.get('filtered_count', 0)}** kursust.")

                st.write("**RAG otsingu tulemus:**")
                ctx_df = debug.get('context_df')
                if ctx_df is not None and not ctx_df.empty:
                    show_cols = [c for c in ["unique_ID", "nimi_et", "eap", "semester", "oppeaste", "score"] if c in ctx_df.columns]
                    st.dataframe(ctx_df[show_cols], hide_index=True, use_container_width=True)
                else:
                    st.warning("Ühtegi kursust ei leitud.")

                st.text_area(
                    "LLM-ile saadetud süsteemiviip:",
                    debug.get('system_prompt', ''),
                    height=150,
                    disabled=True,
                    key=f"prompt_area_{i}"
                )
                st.caption(f"🔢 Tokenid: {debug.get('in_tok', 0)} sisend + {debug.get('out_tok', 0)} väljund · ${debug.get('cost', 0):.6f}")

            # Tagasiside vorm
            with st.expander("📝 Hinda vastust"):
                with st.form(key=f"feedback_form_{i}"):
                    rating = st.radio(
                        "Hinnang vastusele:",
                        ["👍 Hea", "👎 Halb"],
                        horizontal=True,
                        key=f"rating_{i}"
                    )
                    error_cat = st.selectbox(
                        "Kui vastus oli halb, siis mis läks valesti?",
                        [
                            "—",
                            "Filtrid olid liiga karmid/valed (metaandmete filtreerimine)",
                            "Otsing leidis valed ained (RAG vektorotsing)",
                            "LLM hallutsineeris/vastas valesti (LLM genereerimine)"
                        ],
                        key=f"error_cat_{i}"
                    )
                    if st.form_submit_button("Salvesta hinnang"):
                        ctx_df = debug.get('context_df')
                        ctx_ids   = ctx_df["unique_ID"].tolist() if ctx_df is not None and not ctx_df.empty else []
                        ctx_names = ctx_df["nimi_et"].tolist()   if ctx_df is not None and not ctx_df.empty and "nimi_et" in ctx_df.columns else []
                        log_feedback(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            debug.get('user_prompt', ''),
                            debug.get('filters', ''),
                            ctx_ids,
                            ctx_names,
                            message["content"],
                            rating,
                            error_cat if rating == "👎 Halb" else "—"
                        )
                        st.success("✅ Tagasiside salvestatud faili tagasiside_log.csv!")

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    current_filters_str = (
        f"Semester:{sel_semester}, EAP:{sel_eap}, Õppeaste:{sel_level}, "
        f"Keel:{sel_lang}, Hindamisviis:{sel_exam}"
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        current_api_key = st.session_state.get("api_key", "").strip()

        if not current_api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Otsin sobivaid kursusi..."):

                # Filtreerimine
                merged_df = pd.merge(df, embeddings_df, on='unique_ID')
                mask = pd.Series([True] * len(merged_df), index=merged_df.index)

                if sel_semester != "(kõik)":
                    mask &= merged_df["semester"] == sel_semester
                if sel_eap != "(kõik)":
                    mask &= merged_df["eap"].astype(int).astype(str) == sel_eap
                if sel_level != "(kõik)":
                    mask &= merged_df["oppeaste"] == sel_level
                if sel_lang != "(kõik)":
                    mask &= merged_df["keel"] == sel_lang
                if sel_exam != "(kõik)":
                    mask &= merged_df["hindamisviis"] == sel_exam

                filtered_df = merged_df[mask].copy()
                filtered_count = len(filtered_df)

                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vastanud filtritele.")
                    context_text = "Sobivaid kursusi ei leitud."
                    results_df_display = pd.DataFrame()
                else:
                    query_vec = embedder.encode([prompt])[0]
                    filtered_df["score"] = cosine_similarity(
                        [query_vec], np.stack(filtered_df["embedding"])
                    )[0]
                    results_df = filtered_df.sort_values("score", ascending=False).head(n_results)
                    results_df_display = results_df.drop(columns=["embedding"], errors="ignore").copy()
                    context_text = results_df.drop(columns=["score", "embedding"], errors="ignore").to_string()

                # LLM päring
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=current_api_key,
                    default_headers={"HTTP-Referer": "http://localhost:8501"}
                )

                system_prompt_content = (
                    "Oled Tartu Ülikooli kursuste nõustaja. "
                    "Aita üliõpilasel leida sobivad kursused. Vasta eesti keeles.\n\n"
                    f"Leitud kursused:\n\n{context_text}"
                )

                messages_to_send = [
                    {"role": "system", "content": system_prompt_content}
                ] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                try:
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=messages_to_send,
                        stream=True
                    )
                    response = st.write_stream(stream)

                    in_tok  = sum(len(str(m.get("content", ""))) // 4 for m in messages_to_send)
                    out_tok = len(response) // 4
                    price   = PRICES[model_name]
                    cost    = (in_tok * price["input"] + out_tok * price["output"]) / 1_000_000

                    st.session_state.total_input_tokens  += in_tok
                    st.session_state.total_output_tokens += out_tok
                    st.session_state.total_cost          += cost

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "debug_info": {
                            "user_prompt":    prompt,
                            "filters":        current_filters_str,
                            "filtered_count": filtered_count,
                            "context_df":     results_df_display,
                            "system_prompt":  system_prompt_content,
                            "in_tok":         in_tok,
                            "out_tok":        out_tok,
                            "cost":           cost,
                        }
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"Viga: {e}")