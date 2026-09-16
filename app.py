import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from config import DOCUMENTS_DIR
from rag import (
    add_document_to_collection,
    get_collection_count,
    clear_collection,
)
from qa import answer_question


# --------------------------------------------------
# Speech to Text Component
# --------------------------------------------------

speech_to_text = components.declare_component(
    "speech_to_text",
    path=str(Path(__file__).parent / "stt_component"),
)


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

    /* =============================================
       REMOVE MAIN PAGE SCROLL
       ============================================= */

    html,
    body {
        overflow: hidden !important;
    }

    .stApp {
        overflow: hidden !important;
    }

    [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
    }

    section[data-testid="stMain"] {
        overflow: hidden !important;
        height: 100vh !important;
    }

    section[data-testid="stMain"] > div {
        overflow: hidden !important;
    }


    /* =============================================
       MAIN PAGE
       ============================================= */

    .block-container {
        max-width: 1200px;

        padding-top: 2rem;

        /* Space reserved for fixed composer */
        padding-bottom: 130px !important;

        overflow: hidden !important;
    }


    /* =============================================
       HEADER
       ============================================= */

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


    /* =============================================
       WELCOME CARD
       ============================================= */

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


    /* =============================================
       FEATURE CARDS
       ============================================= */

    .feature-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: white;
        text-align: left;
        margin-bottom: 0.7rem;
    }


    /* =============================================
       SOURCES
       ============================================= */

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


    /* =============================================
       SIDEBAR
       ============================================= */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }


    /* =============================================
       FIXED BOTTOM CHAT COMPOSER
       ============================================= */

    div[data-testid="stCustomComponentV1"] {

        position: fixed !important;

        left: 30% !important;
        right: 3.5% !important;

        bottom: 18px !important;

        width: auto !important;
        height: 75px !important;

        margin: 0 !important;
        padding: 0 !important;

        z-index: 999999 !important;

        background: transparent !important;

        border: none !important;

        box-shadow: none !important;
    }


    /* =============================================
       COMPONENT INNER WRAPPER
       ============================================= */

    div[data-testid="stCustomComponentV1"] > div {

        width: 100% !important;
        height: 75px !important;

        margin: 0 !important;
        padding: 0 !important;

        background: transparent !important;
    }


    /* =============================================
       CHAT HISTORY
       ONLY THIS AREA SCROLLS
       ============================================= */

    div[data-testid="stChatMessage"] {
        margin-bottom: 12px;
    }


    /* =============================================
       HIDE SCROLLBAR OF MAIN PAGE
       ============================================= */

    section[data-testid="stMain"]::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
    }

    section[data-testid="stMain"] {
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }


    /* =============================================
       RESPONSIVE
       ============================================= */

    @media (max-width: 900px) {

        div[data-testid="stCustomComponentV1"] {

            left: 5% !important;
            right: 5% !important;

            bottom: 12px !important;
        }

        .block-container {
            padding-bottom: 115px !important;
        }
    }


    @media (max-width: 600px) {

        div[data-testid="stCustomComponentV1"] {

            left: 3% !important;
            right: 3% !important;

            bottom: 8px !important;
        }

        .block-container {
            padding-bottom: 105px !important;
        }
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

if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

if "input_version" not in st.session_state:
    st.session_state.input_version = 0


# --------------------------------------------------
# Helper: Extract Question
# --------------------------------------------------

def get_question_text(value):

    if value is None:
        return None

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):

        possible_keys = [
            "value",
            "text",
            "question",
            "message",
            "data",
        ]

        for key in possible_keys:

            if key in value:

                extracted = value[key]

                if isinstance(extracted, str):
                    return extracted.strip()

        return None

    return None


# --------------------------------------------------
# Welcome Popup
# --------------------------------------------------

@st.dialog("👋 Welcome!")
def welcome_popup():

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:10px;
        ">

            <div style="
                font-size:55px;
                margin-bottom:10px;
            ">
                🤖
            </div>

            <h2 style="
                margin-bottom:10px;
            ">
                Hi! 👋 I am your chatbot.
            </h2>

            <p style="
                color:#6b7280;
                font-size:16px;
                line-height:1.6;
            ">
                Upload your documents and ask me
                anything about them.
            </p>

            <p style="
                color:#6b7280;
                font-size:15px;
            ">
                You can type your question or use the 🎙️ microphone.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🚀 Start Asking Questions",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.welcome_shown = True

        st.rerun()


if not st.session_state.welcome_shown:
    welcome_popup()


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


    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    st.subheader("📊 Statistics")

    chunk_count = get_collection_count()

    st.metric(
        "Indexed Chunks",
        chunk_count,
    )


    st.divider()


    # --------------------------------------------------
    # Clear Knowledge Base
    # --------------------------------------------------

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True,
    ):

        clear_collection()

        st.session_state.messages = []

        st.session_state.input_version = 0

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

    # ==================================================
    # SCROLLABLE CHAT HISTORY
    # ==================================================
    #
    # ONLY THIS CONTAINER SCROLLS.
    # THE MAIN PAGE ITSELF DOES NOT SCROLL.
    #
    # ==================================================

    with st.container(
        height=560,
        border=False,
    ):

        for message in st.session_state.messages:

            role = message["role"]


            # ------------------------------------------
            # User Message
            # ------------------------------------------

            if role == "user":

                with st.chat_message("user"):

                    st.markdown(
                        message["content"]
                    )


            # ------------------------------------------
            # Assistant Message
            # ------------------------------------------

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


                    # ----------------------------------
                    # Sources & Evidence
                    # ----------------------------------

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


    # ==================================================
    # FIXED BOTTOM CHAT COMPOSER
    # ==================================================

    component_key = (
        f"speech_to_text_{st.session_state.input_version}"
    )


    raw_question = speech_to_text(
        default=None,
        key=component_key,
        height=75,
    )


    # --------------------------------------------------
    # Extract Question
    # --------------------------------------------------

    question = get_question_text(
        raw_question
    )


    # --------------------------------------------------
    # Process Question
    # --------------------------------------------------

    if question:

        question = question.strip()


        if question:

            normalized_question = (
                question.lower().strip()
            )


            # ------------------------------------------
            # Save User Message
            # ------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )


            # ------------------------------------------
            # Greeting Words
            # ------------------------------------------

            greeting_words = {
                "hi",
                "hii",
                "hiii",
                "hello",
                "hey",
                "hey there",
                "good morning",
                "good afternoon",
                "good evening",
            }


            # ------------------------------------------
            # Goodbye / Thanks Words
            # ------------------------------------------

            goodbye_words = {
                "bye",
                "goodbye",
                "thanks",
                "thank you",
                "thx",
                "thankyou",
            }


            # ------------------------------------------
            # Greeting Response
            # ------------------------------------------

            if normalized_question in greeting_words:

                answer = (
                    "Hello! 👋 How can I help you "
                    "with your documents?"
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                    }
                )


                st.session_state.input_version += 1

                st.rerun()


            # ------------------------------------------
            # Goodbye / Thanks Response
            # ------------------------------------------

            elif normalized_question in goodbye_words:

                answer = (
                    "You're welcome! 😊 Have a great day!"
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                    }
                )


                st.session_state.input_version += 1

                st.rerun()


            # ------------------------------------------
            # RAG Question
            # ------------------------------------------

            else:

                with st.spinner(
                    "Searching documents and generating answer..."
                ):

                    result = answer_question(
                        question,
                        candidate_k=10,
                        final_k=3,
                    )


                answer = result["answer"]

                sources = result.get(
                    "sources",
                    [],
                )


                # --------------------------------------
                # Save Assistant Response
                # --------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )


                # --------------------------------------
                # Fresh Input Component
                # --------------------------------------

                st.session_state.input_version += 1

                st.rerun()