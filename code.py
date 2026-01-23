!pip install langchain langchain-groq gradio langchain-tavily
from langchain_groq import ChatGroq
from google.colab import userdata
from langchain_tavily import TavilySearch
import gradio as gr
groq_api_key = userdata.get("GROQ_API_KEY")
tavily_api_key = userdata.get("TAVILY_API_KEY")
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0, groq_api_key= groq_api_key)
search = TavilySearch(max_results=5,tavily_api_key=tavily_api_key)
query ="What is the current weather?"
print(f"Agent is searching for: {query}...")
res=search.invoke(query)
context= "\n".join([r['content'] for r in res .get('results',[])])
prompt=f"Using this info: {context}\n\nQuestion: {query}\nAnswer: "
print(f"Final answer: {llm.invoke(prompt).content.strip()}")
def search_agent(query):
    if not query.strip():
        return " Please enter a valid query."

    res = search.invoke(query)
    context = "\n".join([r['content'] for r in res.get('results', [])])

    prompt = f"""
    Using the following information from web search:
    {context}

    Question: {query}
    Answer clearly and concisely:
    """

    response = llm.invoke(prompt)

    return response.content.strip()
custom_css = """
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.gradio-container {
    max-width: 900px !important;
    margin: auto;
}

#title {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    color: white;
}

#subtitle {
    text-align: center;
    color: #cfd8dc;
    margin-bottom: 20px;
}

.card {
    background: #111827;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.4);
}
"""
with gr.Blocks(css=custom_css) as demo:

    gr.Markdown("<div id='title'>Nithi's Search Assistant</div>")
    gr.Markdown("<div id='subtitle'>Nithi's Search agent</div>")

    with gr.Column(elem_classes="card"):
        query_input = gr.Textbox(
            label="Enter your question",
            placeholder="Enter your Question",
            lines=2
        )

        search_btn = gr.Button(
            "Search",
            variant="primary"
        )

        output_box = gr.Textbox(
            label="AI Answer",
            lines=10,
            interactive=False
        )

    search_btn.click(
        fn=search_agent,
        inputs=query_input,
        outputs=output_box
    )
demo.launch(debug=True)
