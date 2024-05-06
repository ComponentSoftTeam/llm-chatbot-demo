import logging
from pprint import pprint
import time
from colorama import Fore
from abc import abstractmethod
import os
from typing import Any, Iterable, List, Literal, Tuple, get_args, overload, Dict
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv, find_dotenv
from component_utils import cached
import re
import httpx

from fireworks.client import Fireworks
import fireworks.client

logging.basicConfig(level=logging.INFO)
word_splitter = re.compile(r'(\s+)')

load_dotenv(find_dotenv(), override=True)

Messages = List[Tuple[str, str]]

class LLMProvider:
    @classmethod
    @abstractmethod
    def get_client(cls) -> Any: ...

    @abstractmethod
    def exec(self, model: str, chat_history: Messages, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """"Executes a model synchronously and returns the response"""

    @abstractmethod
    def exec_streaming(self, model: str, chat_history: Messages, system_prompt: str,  temperature: float, max_tokens: int) -> Iterable[str]:
        """"Executes a model asynchronously and returns a generator of responses"""

    @abstractmethod
    def get_chat_history(self, chat_history: Messages, system_prompt: str) -> Any:
        """"Formats the chat history for the model"""


class OpenAICompletionProvider(LLMProvider):
    client = OpenAI() # The api key is queried from the env

    @classmethod
    def get_client(cls) -> OpenAI: 
        return cls.client

    COMPLETION_MODELS = {"gpt-3.5-turbo-instruct",
                         "davinci-002", "babbage-002"}
    def get_chat_history(self, chat_history: Messages, system_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for user_prompt, assistant_response in chat_history:
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            if assistant_response:
                messages.append(
                    {"role": "assistant", "content": assistant_response})

        if len(messages) == 2:
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
        else:
            prompt = f"{messages[0]['content']}\n\n"
            prompt += '\n'.join([
                f'{"Assistant:" if i % 2 == 0 else "User:"} {chat["content"]}' for i, chat in enumerate(messages)
            ])

        return prompt

    def exec(self, model: str, chat_history: Messages, system_prompt: str,  temperature: float, max_tokens: int) -> str:
        history = self.get_chat_history(chat_history, system_prompt)
        response = OpenAICompletionProvider.client.completions.create(
            model=model,
            prompt=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].text


    def exec_streaming(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str, temperature: float, max_tokens: int) -> Iterable[str]:
        history = self.get_chat_history(chat_history, system_prompt)
        stream = OpenAICompletionProvider.client.completions.create(
            model=model,
            prompt=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        yield from (chunk.choices[0].text for chunk in stream if chunk.choices[0].text is not None)



class OpenAIProvider(LLMProvider):
    client = OpenAI() # The api key is queried from the env

    @classmethod
    def get_client(cls) -> OpenAI: 
        return cls.client

    def get_chat_history(self, chat_history: Messages, system_prompt: str) -> List[ChatCompletionMessageParam]:
        messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
        for user_prompt, assistant_response in chat_history:
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            if assistant_response:
                messages.append(
                    {"role": "assistant", "content": assistant_response})

        return messages


    def exec(self, model: str, chat_history: Messages, system_prompt: str,  temperature: float, max_tokens: int) -> str:
        history = self.get_chat_history(chat_history, system_prompt)
        response = OpenAIProvider.client.chat.completions.create(
            model=model,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )

        return response.choices[0].message.content or ""

    def exec_streaming(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str, temperature: float, max_tokens: int) -> Iterable[str]:
        history = self.get_chat_history(chat_history, system_prompt)
        stream = OpenAIProvider.client.chat.completions.create(
            model=model,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        yield from (chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content is not None)


class MistralProvider(LLMProvider):
    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

    @classmethod
    def get_client(cls) -> MistralClient: 
        return cls.client

    def get_chat_history(self, chat_history: Messages, system_prompt: str) -> List[ChatMessage]:
        messages = [ChatMessage(role="system", content=system_prompt)]
        for user_prompt, assistant_response in chat_history:
            if user_prompt:
                messages.append(ChatMessage(role='user', content=user_prompt))
            if assistant_response:
                messages.append(ChatMessage(role='assistant',
                                content=assistant_response))
        return messages

    def exec(self, model: str, chat_history: Messages, system_prompt: str, temperature: float, max_tokens: int) -> str:
        history = self.get_chat_history(chat_history, system_prompt)
        chat_response = MistralProvider.client.chat(
            model=model,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            safe_prompt=False
        )

        return chat_response.choices[0].message.content

    def exec_streaming(self, model: str, chat_history: Messages, system_prompt: str, temperature: float, max_tokens: int) -> Iterable[str]:
        history = self.get_chat_history(chat_history, system_prompt)
        stream = MistralProvider.client.chat_stream(
            model=model,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            safe_prompt=False
        )

        chunks = (chunk.choices[0].delta.content for chunk in stream)
        yield from (chunk for chunk in chunks if chunk)


class LlamaProvider(LLMProvider):
    client = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))

    @classmethod
    def get_client(cls) -> Fireworks: 
        return cls.client

    def get_chat_history(self, chat_history: Messages, system_prompt: str) -> List[Dict[str, str]]:

        history = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})

        for user, assistant in chat_history:
            if user:
                history.append({"role": "user", "content": user})
            if assistant:
                history.append({"role": "assistant", "content": assistant})

        return history
    
    def exec(self, model: str, chat_history: Messages, system_prompt: str,  temperature: float, max_tokens: int) -> str:

        history = self.get_chat_history(chat_history, system_prompt)

        response = LlamaProvider.client.chat.completions.create(
          model=f"accounts/fireworks/models/{model}",
          messages=history,
          temperature=temperature,
          max_tokens=max_tokens,
          stream=False,
        )

        return response.choices[0].message.content  # type: ignore

    def exec_streaming(self, model: str, chat_history: Messages, system_prompt: str,  temperature: float, max_tokens: int) -> Iterable[str]:

        history = self.get_chat_history(chat_history, system_prompt)

        response = LlamaProvider.client.chat.completions.create(
          model=f"accounts/fireworks/models/{model}",
          messages=history,
          temperature=temperature,
          max_tokens=max_tokens,
          stream=True,
        )

        chunks = (chunk.choices[0].delta.content for chunk in response)

        yield from (chunk for chunk in chunks if chunk)



class GoogleProvider(LLMProvider):
    client = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))

    @classmethod
    def get_client(cls) -> Fireworks: 
        return cls.client

    def get_chat_history(self, chat_history: Messages, system_prompt: str) -> List[Dict[str, str]]:

        history = []
        if system_prompt:
            history.append({"role": "user", "content": system_prompt})
            history.append({"role": "assistant", "content": "Sure thing, I'll follow your instructions."})

        for user, assistant in chat_history:
            if user:
                history.append({"role": "user", "content": user})
            if assistant:
                history.append({"role": "assistant", "content": assistant})

        return history
    
    def exec(self, model: str, chat_history: Messages, system_prompt: str,  temperature: float, max_tokens: int) -> str:

        history = self.get_chat_history(chat_history, system_prompt)

        response = LlamaProvider.client.chat.completions.create(
          model=f"accounts/fireworks/models/{model}",
          messages=history,
          temperature=temperature,
          max_tokens=max_tokens,
          stream=False,
        )

        return response.choices[0].message.content  # type: ignore

    def exec_streaming(self, model: str, chat_history: Messages, system_prompt: str,  temperature: float, max_tokens: int) -> Iterable[str]:

        history = self.get_chat_history(chat_history, system_prompt)

        response = LlamaProvider.client.chat.completions.create(
          model=f"accounts/fireworks/models/{model}",
          messages=history,
          temperature=temperature,
          max_tokens=max_tokens,
          stream=True,
        )

        chunks = (chunk.choices[0].delta.content for chunk in response)

        yield from (chunk for chunk in chunks if chunk)



class Prompt:
    MODEL_FAMILIES = Literal["GPT", "Mistral", "Llama", "Google", "Misc"]
    OPENAI_MODELS = Literal["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
    OPENAI_COMPLETION_MODELS = Literal["gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"]
    MISTRAL_MODELS = Literal["mistral-tiny", "mistral-small", "mistral-large-latest",
                             #"mistral-medium", 
                             "open-mistral-7b", "open-mixtral-8x7b", "open-mixtral-8x22b"]
    LLAMA_MODELS = Literal["llama-v3-8b-instruct", "llama-v3-70b-instruct"]
    GOOGLE_MODELS = Literal["gemma-7b-it"]

    MODELS = OPENAI_MODELS | OPENAI_COMPLETION_MODELS | MISTRAL_MODELS | LLAMA_MODELS | GOOGLE_MODELS

    model = "mistral-tiny"
    model_family: MODEL_FAMILIES = "Mistral"
    __provider: LLMProvider = MistralProvider()

    __system_prompt = "You are a helpful assistant"
    __temperature = 0.7
    __max_tokens = 512
    __verbose = False

    @classmethod
    def exec(cls, chat_history: str | Messages, system_prompt: str | None = None, temperature: float | None = None, max_tokens: int | None = None, verbose: bool | None = None) -> str:
        chat_history = [(chat_history, '')] if isinstance(chat_history, str) else chat_history
        system_prompt = system_prompt or cls.__system_prompt
        temperature = temperature or cls.__temperature
        max_tokens = max_tokens or cls.__max_tokens
        verbose = cls.__verbose if verbose is None else verbose

        if verbose:
            print(f"\nModel: {cls.model_family} {cls.model}")
            print(f"System Prompt: {system_prompt}")
            print(f"Temperature: {temperature}")
            print(f"Max Tokens: {max_tokens}")
            print("Chat History: ", end="")
            pprint(chat_history, compact=True)
            print("Prompt: ", end="")
            pprint(cls.__provider.get_chat_history(
                chat_history, system_prompt), compact=True)
            print()

        res = cls.__provider.exec(
            cls.model, chat_history, system_prompt, temperature, max_tokens)

        return res

    @classmethod
    def exec_streaming(cls, chat_history: str | Messages, system_prompt: str | None = None, temperature: float | None = None, max_tokens: int | None = None, verbose: bool | None = None) -> Iterable[str]:
        chat_history = [(chat_history, '')] if isinstance(chat_history, str) else chat_history
        system_prompt = system_prompt or cls.__system_prompt
        temperature = temperature or cls.__temperature
        max_tokens = max_tokens or cls.__max_tokens
        verbose = cls.__verbose if verbose is None else verbose

        if verbose:
            print(f"\nModel: {cls.model_family} {cls.model}")
            print(f"System Prompt: {system_prompt}")
            print(f"Temperature: {temperature}")
            print(f"Max Tokens: {max_tokens}")
            print("Chat History: ", end="")
            pprint(chat_history, compact=True)
            print("Prompt: ", end="")
            pprint(cls.__provider.get_chat_history(
                chat_history, system_prompt), compact=True)
            print()

        stream = cls.__provider.exec_streaming(
            cls.model, chat_history, system_prompt, temperature, max_tokens)

        yield from stream


    @overload
    @classmethod
    def set_model(cls, model_family: Literal["GPT"], model: OPENAI_MODELS | OPENAI_COMPLETION_MODELS): ...

    @overload
    @classmethod
    def set_model(
        cls, model_family: Literal["Mistral"], model: MISTRAL_MODELS): ...

    @overload
    @classmethod
    def set_model(
        cls, model_family: Literal["Llama"], model: LLAMA_MODELS): ...

    @overload
    @classmethod
    def set_model(
        cls, model_family: Literal["Google"], model: GOOGLE_MODELS): ...

    @classmethod
    def set_model(cls, model_family: MODEL_FAMILIES, model: str):
        match model_family:
            case 'GPT': 
                if model in get_args(cls.OPENAI_MODELS):
                    cls.__provider = OpenAIProvider()
                else:
                    cls.__provider = OpenAICompletionProvider()
            case 'Mistral': cls.__provider = MistralProvider()
            case 'Llama': cls.__provider = LlamaProvider()
            case 'Google': cls.__provider = GoogleProvider()
            case _: raise ValueError("Model family is unknown or incorrectly set")

        cls.model_family = model_family
        if model == "mistral-large":
            model = "mistral-large-latest"

        cls.model = model

    @classmethod
    def set_system_prompt(cls, system_prompt: str):
        cls.__system_prompt = system_prompt

    @classmethod
    def set_temperature(cls, temperature: float):
        if cls.model_family == "Mistral" and temperature > 1:
            temperature = 1
        if cls.model_family == "Llama" and temperature < 0.01:
            temperature = 0.01
        cls.__temperature = temperature

    @classmethod
    def set_max_tokens(cls, max_tokens: int):
        cls.__max_tokens = max_tokens

    @classmethod
    def set_verbose(cls, verbose: bool):
        cls.__verbose = verbose

    @staticmethod
    def get_available_models():
        openai_models = OpenAIProvider.get_client().models.list()
        mistral_models = MistralProvider.get_client().list_models()

        # FIXME: The upstream Fireworks repo has a pydantic error in the list request, so a network request is made to the underlying api

        fireworks_models_api = f"{fireworks.client.base_url}/models"
        fireworks_api_client = httpx.Client(
            headers={"Authorization": f"Bearer { os.getenv('FIREWORKS_API_KEY') }"},
        )
        fireworks_models = (
            model["id"] 
            for model in fireworks_api_client.get(fireworks_models_api).json()["data"]
            if model["id"].startswith("accounts/fireworks/models/")
        )


        return {
            "GPT": [model["id"] for model in openai_models.model_dump()["data"]],
            "Mistral": [model["id"] for model in mistral_models.model_dump()["data"]],
            "Fireworks": [model for model in fireworks_models]
        }

    @classmethod
    def print_models(cls):
        available_models = cls.get_available_models()
        for model_family, models in available_models.items():
            models = list(set(models))
            # color the output
            print(f"{Fore.MAGENTA}[{model_family}]:")
            for model in models:
                print(f"\t{Fore.GREEN}{model}{Fore.RESET}")
            print("\n")
    
    @classmethod
    def print_useful_models(cls):
        available_models = cls.get_available_models()
        for model_family, models in available_models.items():
            models = list(set(models))
            # color the output
            print(f"{Fore.MAGENTA}[{model_family}]:")
            for model in models:
                is_llama_v3 = any(model.endswith(llama) for llama in get_args(cls.LLAMA_MODELS))
                is_google = any(model.endswith(google) for google in get_args(cls.GOOGLE_MODELS))

                if model_family in ["GPT", "Mistral"] or is_llama_v3 or is_google: 
                    print(f"\t{Fore.GREEN}{model}")
            print("\n")
    

# Only cache if the response tuple is OK
@cached(pred=lambda res: res[0])
def test(model_family, model, stream=False) -> Tuple[bool, str]:

    ok = True
    try:
        Prompt.set_model(model_family, model)
        Prompt.set_system_prompt("Respond:")

        start_time = time.time()
        if stream:
            res = Prompt.exec_streaming('Hello!', verbose=True)
            res = ''.join(res)
        else:
            res = Prompt.exec('Hello!', verbose=True)
        duration_seconds = time.time() - start_time
    except Exception as e:
        logging.error(f"Error with model {model_family} {model}: {e}")
        ok = False
        res = ""
        duration_seconds = 0

    if ok:
        result = f"{Fore.GREEN}OK{Fore.RESET} ({len(res.split())}:{len(res)}) in {Fore.YELLOW}{duration_seconds:.2f}s{Fore.RESET}"
    else:
        result = f"{Fore.RED}FAILED{Fore.RESET}"

    return (ok, (
        f"{Fore.MAGENTA}[{model_family}]{Fore.BLUE}[{model}]{Fore.RESET}"
        f"{Fore.LIGHTBLACK_EX}[{'Stream' if stream else 'Normal'}]{Fore.RESET}\t"
        f"{result}"
    ))


def test_all():

    openai = get_args(Prompt.OPENAI_MODELS) + get_args(Prompt.OPENAI_COMPLETION_MODELS)
    mistral = get_args(Prompt.MISTRAL_MODELS)
    llama = get_args(Prompt.LLAMA_MODELS)
    google = get_args(Prompt.GOOGLE_MODELS)

    Prompt.set_max_tokens(80)
    models = [
            ("GPT", openai),
            ("Mistral", mistral),
            ("Llama", llama),
            ("Google", google),
    ]


    for model_family, models in models:
        for model in models:
            ok, res = test(model_family, model, stream=False)
            print(res)
            ok, res = test(model_family, model, stream=True)
            print(res)

#Prompt.print_useful_models()
if '__main__' == __name__:
        # test_all()
        print(Prompt.get_available_models())
