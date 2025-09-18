# streamlit_app.py
import streamlit as st

# ⬇️ Replace this import with wherever your compiled LangGraph "app" lives.
# It must expose: app.invoke({"question": str, "k": int}) -> {"answer": str, "sources": list}
from rag_graph import app  # e.g., your module that builds and compiles the graph

st.set_page_config(page_title="Friends RAG Chat", page_icon="💬")
st.title("💬 Friends RAG (minimal)")

if "history" not in st.session_state:
    st.session_state.history = []   # [(role, text)]

k = st.sidebar.slider("Top-k", 2, 10, 6)

# render history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# chat input (bottom of page)
q = st.chat_input("Ask about Friends…")  # appears once per page
if q:
    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            out = app.invoke({"question": q, "k": k})
            ans = out.get("answer") or "_I don't know._"
            st.markdown(ans)
            srcs = out.get("sources") or []
            if srcs:
                with st.expander("Sources"):
                    for s in srcs[:5]:
                        st.write("-", s)

    st.rerun()