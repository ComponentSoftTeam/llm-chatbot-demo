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
import os
from enum import Enum
from typing import Iterable

import gradio as gr
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_fireworks.chat_models import ChatFireworks
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI

load_dotenv()

# The user for the Gradio interface
GRADIO_USER = os.environ["GRADIO_USER"]
GRADIO_PASSWORD = os.environ["GRADIO_PASSWORD"]


# -

# # Setup the models
#
# Set up the model families, models, and their constructors

# +
class ModelFamily(Enum):
    """Represents the model families available for selection."""
    
    GPT = 'OpenAI GPT'
    GEMINI = 'Google Gemini'
    CLAUDE = 'Anthropic Claude'
    MISTRAL = 'MistralAI Mistral'
    LLAMA = 'Meta Llama'

class ModelName(Enum):
    """
    Enum representing different model names.
    Each model name is associated with a model family and a model identifier.
    """
      
    LLAMA3_1_405B_INSTRUCT = (ModelFamily.LLAMA, 'llama-3.1-405b-instruct')
    LLAMA3_1_70B_INSTRUCT = (ModelFamily.LLAMA, 'llama-3.1-70b-instruct')
    LLAMA3_1_8B_INSTRUCT = (ModelFamily.LLAMA, 'llama-3.1-8b-instruct')
    GPT_3_5_TURBO = (ModelFamily.GPT, 'gpt-3.5-turbo')
    GPT_4O = (ModelFamily.GPT, 'gpt-4o')
    GPT_4_TURBO = (ModelFamily.GPT, 'gpt-4-turbo')
    MISTRAL_LARGE = (ModelFamily.MISTRAL, 'mistral-large')
    OPEN_MIXTRAL_8X22B = (ModelFamily.MISTRAL, 'open-mixtral-8x22b')
    MISTRAL_SMALL = (ModelFamily.MISTRAL, 'mistral-small')
    GEMINI_1_5_FLASH = (ModelFamily.GEMINI, 'gemini-1.5-flash')
    GEMINI_1_5_PRO = (ModelFamily.GEMINI, 'gemini-1.5-pro')
    CLAUDE_3_HAIKU = (ModelFamily.CLAUDE, 'claude-3-haiku')
    CLAUDE_3_5_SONNET = (ModelFamily.CLAUDE, 'claude-3.5-sonnet')
    CLAUDE_3_OPUS = (ModelFamily.CLAUDE, 'claude-3-opus')


def get_llm(model_name: ModelName, temperature: float, max_new_tokens: int) -> BaseChatModel:
    """
    Returns a chat model based on the specified model name, temperature, and maximum number of new tokens.

    Args:
        model_name (ModelName): The name of the model to use.
        temperature (float): The temperature parameter for generating responses, [0, 2].
        max_new_tokens (int): The maximum number of new tokens to generate in the response.

    Returns:
        BaseChatModel: The chat model based on the specified parameters.

    Raises:
        RuntimeError: If an invalid model name is provided.
    """

    match model_name:
        case ModelName.LLAMA3_1_405B_INSTRUCT: 
            return ChatFireworks(
                name="accounts/fireworks/models/llama-v3p1-405b-instruct",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.LLAMA3_1_70B_INSTRUCT: 
            return ChatFireworks(
                name="accounts/fireworks/models/llama-v3p1-70b-instruct",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.LLAMA3_1_8B_INSTRUCT:
            return ChatFireworks(
                name="accounts/fireworks/models/llama-v3p1-8b-instruct",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.GPT_3_5_TURBO:
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.GPT_4O:
            return ChatOpenAI(
                model="gpt-4o",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.GPT_4_TURBO:
            return ChatOpenAI(
                model="gpt-4-turbo",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )        

        case ModelName.MISTRAL_LARGE:
            return ChatMistralAI(
                name="mistral-large-latest",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.OPEN_MIXTRAL_8X22B:
            return ChatMistralAI(
                name="open-mixtral-8x22b",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.MISTRAL_SMALL:
            return ChatMistralAI(
                name="mistral-small-latest",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )        

        case ModelName.GEMINI_1_5_FLASH:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                max_output_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.GEMINI_1_5_PRO:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                max_output_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.CLAUDE_3_HAIKU:
            return ChatAnthropic(
                model_name="claude-3-haiku-20240307",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )    

        case ModelName.CLAUDE_3_5_SONNET:
            return ChatAnthropic(
                model_name="claude-3-5-sonnet-20240620",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )

        case ModelName.CLAUDE_3_OPUS:
            return ChatAnthropic(
                model_name="claude-3-opus-20240229",
                max_tokens = max_new_tokens,
                temperature = temperature,
            )

        case _:
            raise RuntimeError("Invalid input model_name: {model_name}")


# -

# ## Get the model family - model name dictionary
#
# This dictionary is used to set and update the dropdown menus in the Gradio UI 

# +
# Model names by families
model_by_families: dict[str, list[str]] = {family.value: [] for family in ModelFamily}
for model in ModelName:
    family, name = model.value
    model_by_families[family.value].append(name)

print(model_by_families)
# -

# # Generic llm query

# +
# Conversation prompt template
chat_prompt_template = ChatPromptTemplate.from_messages([
    ('system', '{system_prompt}'),
    MessagesPlaceholder('history'),
    ('human', '{prompt}'),
])

str_output_parser = StrOutputParser()

def exec_prompt(
        chat_history: list[list[str]] | None,
        prompt: str,
        system_prompt: str,
        model_family: str ,
        model: str,
        temperature: float,
        max_tokens: int,
        streaming: bool,
    ) -> Iterable[tuple[list[list[str]], str]]:
    """
    Executes a prompt in the chatbot system and returns the chat history and response.

    Args:
        chat_history (list[list[str]] | None): The chat history as a list of human and AI messages.
        prompt (str): The prompt to be executed.
        system_prompt (str): The system prompt to be used.
        model_family (str): The model family to be used.
        model (str): The specific model to be used.
        temperature (float): The temperature parameter for generating responses.
        max_tokens (int): The maximum number of tokens in the generated response.
        streaming (bool): Whether to use streaming or not. In streaming mode, the response is generated in chunks, otherwise the response is generated in one go.

    Returns:
        Iterable[tuple[list[list[str]], str]]: An iterable of tuples containing the updated chat history and an empty string signifying that the user prompt is moved from the input field into the history.

    """
    
    if not prompt:
        prompt = "I have no question"

    if not chat_history:
        chat_history = []

    if model_family in ["Mistral", "Gemini", "Claude"] and temperature > 1:
        temperature = 1

    model_family_kind = ModelFamily(model_family)
    model_name_kind = ModelName((model_family_kind, model))

    llm = get_llm(
        model_name=model_name_kind,
        temperature=temperature,
        max_new_tokens=max_tokens,
    )

    history = []
    for human, ai in chat_history:
        history.append(('human', human))
        history.append(('ai', ai))

   
    chain = chat_prompt_template | llm | str_output_parser

    if streaming:
        response_iter = chain.stream({
            "system_prompt": system_prompt,
            "history": history,
            "prompt": prompt,
        })

        chat_history.append([prompt, ""])
        for response_chunk in response_iter:
            chat_history[-1][1] += response_chunk
            yield (chat_history, "")

    else:
        response = chain.invoke({
            "system_prompt": system_prompt,
            "history": history,
            "prompt": prompt,
        })

        chat_history.append([prompt, response])
        yield (chat_history, "")

# -

# ## Wrappers for the gradio streaming and not streaming methods

# +

def exec_prompt_wrapper_streaming(
        chat_history: list[list[str]] | None, prompt: str, system_prompt: str, model_family: str, model: str, temperature: float, max_tokens: int
) -> Iterable[tuple[list[list[str]], str]]:
    """
    Executes a prompt in a streaming manner using the specified parameters.

    This is a wrapper function around the `exec_prompt` function that sets the `streaming` parameter to `True`.

    Args:
        chat_history (list[list[str]] | None): The chat history as a list of lists of strings. Each inner list represents a message in the conversation.
        prompt (str): The prompt to be executed.
        system_prompt (str): The system prompt to be used.
        model_family (str): The model family to be used.
        model (str): The model to be used.
        temperature (float): The temperature parameter for generating the output [0, 2].
        max_tokens (int): The maximum number of tokens in the generated output.

    Yields:
        Iterable[tuple[list[list[str]], str]]: An iterable of tuples, where each tuple contains the updated chat history and an empty string signifying that the user prompt is moved from the input field into the history.
    """
    
    yield from exec_prompt(
        chat_history=chat_history,
        prompt=prompt,
        system_prompt=system_prompt,
        model_family=model_family,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )

def exec_prompt_wrapper(
    chat_history: list[list[str]] | None, prompt: str, system_prompt: str, model_family: str, model: str, temperature: float, max_tokens: int
) -> tuple[list[list[str]], str]:
    """
    Executes the prompt using the specified parameters, implicitly setting the `streaming` parameter to `False`.

    This is a wrapper function around the `exec_prompt` function that sets the `streaming` parameter to `False`.

    Args:
        chat_history (list[list[str]] | None): The chat history.
        prompt (str): The prompt to generate a response for.
        system_prompt (str): The system prompt.
        model_family (str): The model family.
        model (str): The model.
        temperature (float): The temperature for response generation [0, 2].
        max_tokens (int): The maximum number of tokens for the response.

    Returns:
        tuple[list[list[str]], str]: A tuple containing the updated chat history and an empty string, signifying that the user prompt is moved from the input field into the history
    """

    return next(exec_prompt(
    chat_history=chat_history,
    prompt=prompt,
    system_prompt=system_prompt,
    model_family=model_family,
    model=model,
    temperature=temperature,
    max_tokens=max_tokens,
    streaming=False,
    ))



# -

# # Gradio UI

# +
gr.close_all()
callback = gr.CSVLogger()

def save_datapoint(*args):
    callback.flag(args)
    gr.Info("Data point flagged for review.")

with gr.Blocks(title="CompSoft") as demo:
    # UI Elements

    gr.Markdown("# Component Soft LLM Demo")
    system_prompt = gr.Textbox(label="System prompt", value="You are a helpful, harmless and honest assistant.")

    with gr.Row():
        model_family = gr.Dropdown(
            choices=list(model_by_families.keys()),
            label="Model family",
            value="OpenAI GPT",
        )

        model_name = gr.Dropdown(
            choices=list(model_by_families[model_family.value]),
            label="Model",
            value="gpt-4o",
        )

        temperature = gr.Slider(
            label="Temperature:",
            minimum=0,
            maximum=2,
            value=1,
            info="LLM generation temperature",
        )

        max_tokens = gr.Slider(
            label="Max tokens",
            minimum=100,
            maximum=2000,
            value=500, 
            info="Maximum number of generated tokens",
        )

    with gr.Row():
        chatbot = gr.Chatbot(
            label="ComponentSoft_GPT",
            height=400,
            show_copy_button=True,
        )

    with gr.Row():
        prompt = gr.Textbox(
            label="Your prompt",
            value="Who was Albert Einstein?",
        )

    with gr.Row():
        submit_btn_nostreaming = gr.Button(value="Answer without streaming")
        submit_btn_streaming = gr.Button(value="Answer with streaming")
        clear_btn = gr.ClearButton([prompt, chatbot])
        flag_btn = gr.Button("Flag")

    gr.Examples(
        examples=[
            "Who was Albert Einstein?",
            "When did he live?",
            "What were a few of his most important achievements?",
            "Who were some other important personality from his profession and his age?",
            "Write a Python function which calculates the value of PI in N steps with maximum precision using float64 numbers.",
            "Write the same function in Typescript.",
            "The same in Java?",
            "And what about C#?",
            "In Fortran?",
            "In Cobol?"
        ],
        inputs=prompt
    )


    # Event listeners
    model_family.change(
            fn=lambda family: gr.Dropdown(
                choices=list(model_by_families[family]),
                label="Model",
                value=model_by_families[family][0],
                interactive=True,
            ),
            inputs=model_family,
            outputs=model_name,
    )

    submit_btn_streaming.click(
        fn=exec_prompt_wrapper_streaming,
        inputs=[chatbot, prompt, system_prompt, model_family, model_name, temperature, max_tokens],
        outputs=[chatbot, prompt],
    )

    submit_btn_nostreaming.click(
        fn=exec_prompt_wrapper,
        inputs=[chatbot, prompt, system_prompt, model_family, model_name, temperature, max_tokens],
        outputs=[chatbot, prompt],
    )

    flag_btn.click(
        fn=save_datapoint,  # type: ignore
        inputs=[system_prompt, model_family, model_name, temperature, max_tokens, chatbot],
        preprocess=False,
    )

    callback.setup([system_prompt, model_family, model_name, temperature, max_tokens, chatbot], "flagged_data_points")

#demo.launch()
#demo.launch(share=True)
demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=(GRADIO_USER, GRADIO_PASSWORD), max_threads=20, show_error=True, favicon_path="favicon.ico", state_session_capacity=20)

# + active=""
# gr.close_all()
