import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

# Your existing code
huggingfacehub_api_token = "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"
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id=repo_id,
    model_kwargs={"temperature": 0.6, "max_new_tokens": 500},
)
template = """
You are an artificial intelligence assistant, and your task is to generate a short summary on a provided text.
Summarize the text below, delimited by triple 
backticks, in at most 30 words

Text: '''{text}'''
"""

prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
# Streamlit app
st.title("AI Text Summarizer")
st.write("Enter the text and choose a focus area to generate a summary.")

# User input for the text
user_input_text = st.text_area("Enter the text:", value="", height=300)


# Generate summary on button click
if st.button("Generate Summary"):
    # Run langchain model
    summary = llm_chain.run(user_input_text)
    st.subheader("Generated Summary:")
    st.write(summary)
