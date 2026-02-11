import streamlit as st
import ollama

# --- CONFIGURATION ---
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

st.title("Simple RAG Explorer")
st.markdown("Upload a text file and ask questions about its content.")

# --- SHARED FUNCTIONS ---
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

# --- SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])

# --- SESSION STATE INITIALIZATION ---
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = []
if 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = None

# --- CORE RAG LOGIC ---
if uploaded_file is not None:
    # Only process if it's a new file
    if st.session_state.last_uploaded != uploaded_file.name:
        with st.status("Processing file and generating embeddings...") as status:
            # Read file
            content = uploaded_file.read().decode("utf-8")
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Clear old DB
            st.session_state.vector_db = []
            
            # Build Vector DB
            for i, chunk in enumerate(lines):
                embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
                st.session_state.vector_db.append((chunk, embedding))
            
            st.session_state.last_uploaded = uploaded_file.name
            status.update(label="Database Ready!", state="complete")

    # --- CHAT INTERFACE ---
    query = st.text_input("Ask a question about your file:")

    if query:
        # 1. Retrieve
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        similarities = []
        for chunk, embedding in st.session_state.vector_db:
            sim = cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        retrieved_knowledge = similarities[:3]

        # 2. Display Context (Optional)
        with st.expander("See retrieved context"):
            for chunk, sim in retrieved_knowledge:
                st.write(f"**({sim:.2f})** {chunk}")

        # 3. Generate Answer
        instruction_prompt = f'''You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information:
        {'\n'.join([f' - {chunk}' for chunk, sim in retrieved_knowledge])}
        '''

        st.subheader("Response:")
        response_placeholder = st.empty()
        full_response = ""

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True,
        )

        for chunk in stream:
            full_response += chunk['message']['content']
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

else:
    st.info("Please upload a .txt file in the sidebar to get started.")