import uuid

import httpx
import streamlit as st

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="TradeIQ - Financial Research Assistant",
    page_icon="📈",
    layout="wide",
)


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar() -> dict:
    with st.sidebar:
        st.title("TradeIQ")
        st.caption("Financial Research Knowledge Base")

        st.divider()
        st.subheader("Filters")

        ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL")
        filing_type = st.selectbox("Filing Type", ["All", "10-K", "10-Q", "Earnings Transcript"])
        sectors = ["All", "technology", "finance", "healthcare", "energy", "consumer"]
        sector = st.selectbox("Sector", sectors)
        top_k = st.slider("Sources to retrieve", 1, 10, 5)

        st.divider()
        st.subheader("Session")

        if st.button("New Conversation"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        st.caption(f"Session: {st.session_state.session_id[:8]}...")

        # Health check
        st.divider()
        try:
            resp = httpx.get(f"{API_URL}/health", timeout=5)
            data = resp.json()
            st.success(f"API Connected | {data['vector_store_count']} docs")
        except Exception:
            st.error("API not connected. Run: make run")

    filters = {}
    if ticker:
        filters["ticker"] = ticker.upper()
    if filing_type != "All":
        filters["filing_type"] = filing_type
    if sector != "All":
        filters["sector"] = sector

    return {"filters": filters if filters else None, "top_k": top_k}


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)} documents)", expanded=False):
        for i, source in enumerate(sources):
            score = source.get("relevance_score", 0)
            name = source.get("source", "Unknown")
            metadata = source.get("metadata", {})

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{name}**")
                ticker = metadata.get("ticker", "")
                filing = metadata.get("filing_type", "")
                if ticker or filing:
                    st.caption(f"{ticker} | {filing}")
            with col2:
                st.metric("Relevance", f"{score:.3f}")

            st.markdown(f"```\n{source.get('content', '')[:300]}...\n```")
            if i < len(sources) - 1:
                st.divider()


def main() -> None:
    init_session_state()
    settings = render_sidebar()

    st.title("Ask about SEC filings, earnings, and trading strategies")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])
            if msg.get("timing"):
                t = msg["timing"]
                st.caption(
                    f"Retrieval: {t.get('retrieval_time_ms', 0):.0f}ms | "
                    f"Generation: {t.get('generation_time_ms', 0):.0f}ms | "
                    f"Total: {t.get('query_time_ms', 0):.0f}ms | "
                    f"Model: {t.get('model_used', 'N/A')}"
                )

    # Chat input
    if prompt := st.chat_input("Ask a question about financial data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"), st.spinner("Researching..."):
            try:
                payload = {
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                    "top_k": settings["top_k"],
                }
                if settings["filters"]:
                    payload["filters"] = settings["filters"]

                resp = httpx.post(f"{API_URL}/query", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                st.markdown(data["answer"])
                render_sources(data.get("sources", []))

                timing = {
                    "retrieval_time_ms": data.get("retrieval_time_ms", 0),
                    "generation_time_ms": data.get("generation_time_ms", 0),
                    "query_time_ms": data.get("query_time_ms", 0),
                    "model_used": data.get("model_used", ""),
                }
                st.caption(
                    f"Retrieval: {timing['retrieval_time_ms']:.0f}ms | "
                    f"Generation: {timing['generation_time_ms']:.0f}ms | "
                    f"Total: {timing['query_time_ms']:.0f}ms | "
                    f"Model: {timing['model_used']}"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                    "timing": timing,
                })

            except httpx.ConnectError:
                st.error("Cannot connect to API. Make sure the backend is running: `make run`")
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
