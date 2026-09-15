import streamlit as st

from config import DOCUMENTS_DIR
from rag import (
    add_document_to_collection,
    get_collection_count,
    clear_collection,
)
from qa import answer_question


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .app-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.8rem;
    }

    .welcome-card {
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        background: #fafafa;
        margin-top: 2rem;
        text-align: center;
    }

    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .welcome-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .welcome-text {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .feature-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: white;
        text-align: left;
        margin-bottom: 0.7rem;
    }

    .answer-box {
        padding: 1.2rem 1.3rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        line-height: 1.7;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .source-box {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-bottom: 0.6rem;
    }

    .evidence-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background: #f8fafc;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    .muted {
        color: #6b7280;
        font-size: 0.9rem;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="app-title">📚 Document Q&A Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
    Ask questions about your documents using hybrid retrieval,
    neural reranking, and a local Hugging Face LLM.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT and Markdown.",
    )

    if st.button(
        "🚀 Index Documents",
        use_container_width=True,
        type="primary",
    ):

        if not uploaded_files:

            st.warning(
                "Please upload at least one document."
            )

        else:

            total_chunks = 0

            progress = st.progress(0)

            status = st.empty()

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                status.write(
                    f"Indexing `{uploaded_file.name}`..."
                )

                file_path = (
                    DOCUMENTS_DIR
                    / uploaded_file.name
                )

                file_path.write_bytes(
                    uploaded_file.getbuffer()
                )

                chunks = add_document_to_collection(
                    str(file_path)
                )

                total_chunks += chunks

                progress.progress(
                    (index + 1)
                    / len(uploaded_files)
                )

            status.success(
                f"Indexed {len(uploaded_files)} "
                f"document(s) • "
                f"{total_chunks} chunk(s)"
            )

            st.rerun()

    st.divider()

    st.subheader("📊 Statistics")

    chunk_count = get_collection_count()

    st.metric(
        "Indexed Chunks",
        chunk_count,
    )

    st.divider()

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True,
    ):

        clear_collection()

        st.session_state.messages = []

        st.success(
            "Knowledge base cleared."
        )

        st.rerun()

    st.divider()

    st.caption(
        "🔒 Documents are processed locally."
    )


# --------------------------------------------------
# Main Application
# --------------------------------------------------

chunk_count = get_collection_count()


if chunk_count == 0:

    # --------------------------------------------------
    # Empty State
    # --------------------------------------------------

    st.markdown(
        """
        <div class="welcome-card">

            <div class="welcome-icon">📚</div>

            <div class="welcome-title">
                Your document assistant is ready
            </div>

            <div class="welcome-text">
                Upload your documents from the sidebar
                and start asking questions about them.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ✨ What you can do")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="feature-card">
            📄 <strong>Upload Documents</strong><br>
            <span class="muted">
            PDF, TXT and Markdown files
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">
            🔎 <strong>Smart Retrieval</strong><br>
            <span class="muted">
            Hybrid search + neural reranking
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">
            🤖 <strong>Local AI</strong><br>
            <span class="muted">
            Qwen runs locally on your machine
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💡 Example questions")

    st.markdown(
        """
        - What is this document about?
        - What are the key concepts discussed?
        - Explain the main idea in simple terms.
        - What does the document say about machine learning?
        """
    )


else:

    # --------------------------------------------------
    # Chat History
    # --------------------------------------------------

    for message in st.session_state.messages:

        role = message["role"]

        if role == "user":

            with st.chat_message("user"):

                st.markdown(
                    message["content"]
                )

        else:

            with st.chat_message(
                "assistant",
                avatar="🤖",
            ):

                st.markdown(
                    message["content"]
                )

                sources = message.get(
                    "sources",
                    [],
                )

                if sources:

                    with st.expander(
                        f"📚 Sources & Evidence ({len(sources)})"
                    ):

                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):

                            source_name = source[
                                "source"
                            ]

                            page = source[
                                "page"
                            ]

                            chunk_id = source[
                                "chunk_id"
                            ]

                            evidence = source[
                                "text"
                            ]

                            if (
                                page is not None
                                and page != -1
                            ):

                                location = (
                                    f"Page {page}"
                                )

                            else:

                                location = (
                                    "Document"
                                )

                            st.markdown(
                                f"""
                                <div class="source-box">
                                <strong>
                                📄 Source {index}: {source_name}
                                </strong>
                                <br>
                                <span class="muted">
                                {location} · Chunk {chunk_id}
                                </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            with st.expander(
                                "🔍 View retrieved evidence"
                            ):

                                st.markdown(
                                    f"""
                                    <div class="evidence-box">
                                    {evidence}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )


    # --------------------------------------------------
    # Chat Input
    # --------------------------------------------------

    question = st.chat_input(
        "Ask a question about your documents..."
    )


    if question:

        # ----------------------------------------------
        # User message
        # ----------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # ----------------------------------------------
        # Generate answer
        # ----------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                result = answer_question(
                    question,
                    candidate_k=10,
                    final_k=3,
                )

            answer = result[
                "answer"
            ]

            sources = result.get(
                "sources",
                [],
            )

            st.markdown(answer)


            # ------------------------------------------
            # Sources + Evidence
            # ------------------------------------------

            if sources:

                with st.expander(
                    f"📚 Sources & Evidence ({len(sources)})"
                ):

                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):

                        source_name = source[
                            "source"
                        ]

                        page = source[
                            "page"
                        ]

                        chunk_id = source[
                            "chunk_id"
                        ]

                        evidence = source[
                            "text"
                        ]

                        if (
                            page is not None
                            and page != -1
                        ):

                            location = (
                                f"Page {page}"
                            )

                        else:

                            location = (
                                "Document"
                            )

                        st.markdown(
                            f"""
                            <div class="source-box">
                            <strong>
                            📄 Source {index}: {source_name}
                            </strong>
                            <br>
                            <span class="muted">
                            {location} · Chunk {chunk_id}
                            </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        with st.expander(
                            "🔍 View retrieved evidence"
                        ):

                            st.markdown(
                                f"""
                                <div class="evidence-box">
                                {evidence}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )


            # ------------------------------------------
            # Save assistant response
            # ------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )