# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from llm_prompts import Prompt, test_all
from dotenv import load_dotenv

load_dotenv()
# -

# # Command-line test

# +
# Prompt.set_system_prompt('You are a helpful, honest and honest assistant.')
# Prompt.set_system_prompt('You are a funny assistant speaking in pirate English')
Prompt.set_system_prompt('You are a helpful, harmless and honest assistant who has to answer questions in a funny style.')


# You can use the LSP or the typing to check the available models
#Prompt.set_model("GPT", "gpt-3.5-turbo")
Prompt.set_model("Mistral", "mistral-large")
#Prompt.set_model("Llama", "llama-2-70b-chat")

Prompt.set_max_tokens(512)
Prompt.set_verbose(True)
Prompt.set_temperature(0.00)

streaming = True

prompt = 'Who was Albert Einstein?'

if streaming == True:
    stream = Prompt.exec_streaming(prompt)
    for chunk in stream:
        print(chunk, end="")
    print("")
else:
    response = Prompt.exec(prompt)
    print(response)
    print("")
print()
# -

# # Gradio program

# +
import gradio as gr

Prompt.set_verbose(False)

modelfamilies_model_dict = {
    "GPT": ["gpt-3.5-turbo", "gpt-4"],
    "Mistral": ["mistral-tiny", "mistral-small", "mistral-medium", "mistral-large"],
    "Llama": ["llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat", "codellama-7b-instruct", "codellama-70b-instruct"],
}

def exec_prompt(chat_history, prompt, system_prompt, model_family = "Mistral", model="mistral-large", temperature=0.7, max_tokens=512):
    if prompt == "": prompt = "I have no question"
    if model == "mistral-large": model = "mistral-large-latest"
    if model_family == "Mistral" and temperature > 1: temperature = 1
    if model_family == "Llama" and temperature < 0.01: temperature = 0.01
    Prompt.set_model(model_family, model)
    Prompt.set_system_prompt(system_prompt)
    Prompt.set_temperature(temperature)
    Prompt.set_max_tokens(max_tokens)    
    
    chat_history = chat_history or []
    chat_history.append([prompt, ""])
    response = Prompt.exec(chat_history)
    chat_history[-1][1] = response
    return chat_history, ""

def exec_prompt_streaming(chat_history, prompt, system_prompt, model_family = "Mistral", model="mistral-large", temperature=0.7, max_tokens=512):
    if prompt == "": prompt = "I have no question"
    if model == "mistral-large": model = "mistral-large-latest"
    if model_family == "Mistral" and temperature > 1: temperature = 1
    if model_family == "Llama" and temperature < 0.01: temperature = 0.01
    Prompt.set_system_prompt(system_prompt)
    Prompt.set_temperature(temperature)
    Prompt.set_max_tokens(max_tokens)
    Prompt.set_model(model_family, model)
    
    chat_history = chat_history or []
    chat_history.append([prompt, ""])
    stream = Prompt.exec_streaming(chat_history)
    for new_token in stream:
        if new_token is not None:
            chat_history[-1][1] += str(new_token)
        yield chat_history, ""

gr.close_all()

callback = gr.CSVLogger()

with gr.Blocks(title="CompSoft") as demo:
    gr.Markdown("# Nokia LLM Demo")
    system_prompt = gr.Textbox(label="System prompt", value="You are a helpful, harmless and honest assistant.")
    with gr.Row():
        modelfamily = gr.Dropdown(list(modelfamilies_model_dict.keys()), label="Model family", value="Mistral")
        model = gr.Dropdown(list(modelfamilies_model_dict["Mistral"]), label="Model", value="mistral-large")       
        temperature = gr.Slider(label="Temperature:", minimum=0, maximum=2, value=1,
            info="LLM generation temperature")
        max_tokens = gr.Slider(label="Max tokens", minimum=100, maximum=2000, value=500, 
            info="Maximum number of generated tokens")
    with gr.Row():
        chatbot=gr.Chatbot(label="ComponentSoft_GPT", height=400, show_copy_button=True)
    with gr.Row():
        prompt = gr.Textbox(label="Your prompt", value="Who was Albert Einstein?")
    with gr.Row():
        submit_btn_nostreaming = gr.Button("Answer without streaming")
        submit_btn_streaming = gr.Button("Answer with streaming")
        clear_btn = gr.ClearButton([prompt, chatbot])
        flag_btn = gr.Button("Flag")
    
    
    @modelfamily.change(inputs=modelfamily, outputs=[model])
    def update_modelfamily(modelfamily):
        model = list(modelfamilies_model_dict[modelfamily])
        return gr.Dropdown(choices=model, value=model[0], interactive=True)

    submit_btn_streaming.click(exec_prompt_streaming, inputs=[chatbot,prompt, system_prompt,modelfamily,model,temperature,max_tokens], outputs=[chatbot, prompt])
    submit_btn_nostreaming.click(exec_prompt, inputs=[chatbot,prompt,system_prompt, modelfamily,model,temperature,max_tokens], outputs=[chatbot, prompt])

    callback.setup([system_prompt, modelfamily, model, temperature, max_tokens, chatbot], "flagged_data_points")
    flag_btn.click(lambda *args: callback.flag(args), [system_prompt, modelfamily, model, temperature, max_tokens, chatbot], None, preprocess=False)
    
    gr.Examples(
        ["Who was Albert Einstein?", "When did he live?", "What were a few of his most important achievements?", "Who were some other important personality from his profession and his age?",
        "Write a Python function which calculates the value of PI in N steps with maximum precision using float64 numbers.", "Write the same function in Typescript.", 
         "The same in Java?", "And what about C#?", "In Fortran?", "In Cobol?"],
        prompt
    )

#demo.launch()
demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=("Nokia", "Karakaari7"), max_threads=20, show_error=True, favicon_path="/home/rconsole/GIT/AI-434/source/labfiles/data/favicon.ico", state_session_capacity=20)

# + active=""
# gr.close_all()
# -


