import streamlit as st
import os
from rag import app

st.set_page_config(page_title="Friends Chatbot", page_icon="💬")
st.title("Friends Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []   # [(role, text)]

if "provider" not in st.session_state:
    st.session_state.provider = "azure_openai"

k = st.sidebar.slider("Top-k", 2, 10, 6)

new_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["huggingface", "azure_openai"],
    index=["huggingface", "azure_openai"].index(st.session_state.provider),
    help="Select LLM provider (! changes will clear chat history)"
)

if new_provider != st.session_state.provider:
    st.session_state.provider = new_provider
    st.session_state.history = []
    st.rerun()

st.sidebar.success(f"Active: {st.session_state.provider}")

os.environ["LLM_PROVIDER"] = st.session_state.provider
# render history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

q = st.chat_input("Ask anything about Friends…")
if q:
    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                out = app.invoke({
                    "question": q,
                    "k": k,
                    "chat_history": st.session_state.history
                })
                ans = out.get("answer") or "_I don't know._"
                st.markdown(ans)
                st.session_state.history.append(("assistant", ans))
                srcs = out.get("sources") or []
                if srcs:
                    with st.expander("Sources"):
                        for s in srcs[:5]:
                            st.write("-", s)
            except Exception as e:
                st.error(f"Sorry, something went wrong: {str(e)}")
                st.session_state.history.append(("assistant", "Sorry, I encountered an error. Please try again."))

    st.rerun()