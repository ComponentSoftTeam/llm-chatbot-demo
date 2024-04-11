import logging
from pprint import pprint
import time
from colorama import Fore, Back, Style
import httpx
from abc import abstractmethod
import os
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, get_args, overload
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import replicate.client
from dotenv import load_dotenv, find_dotenv
from component_utils import cached
import re

logging.basicConfig(level=logging.INFO)
word_splitter = re.compile(r'(\s+)')

load_dotenv(find_dotenv())


OpenAI.api_key = os.environ['OPENAI_API_KEY']
Mistral_api_key = os.environ["MISTRAL_API_KEY"]
REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]


OpenAI_client = OpenAI(api_key=OpenAI.api_key)
Mistral_client = MistralClient(api_key=Mistral_api_key)
Llama_client = replicate.client.Client(api_token=REPLICATE_API_TOKEN, timeout=httpx.Timeout(
    300.0, read=300.0, write=300.0, connect=300.0, pool=300.0)
)


class LLMProvider:
    @abstractmethod
    def exec(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str, temperature: float, max_tokens: int) -> str:
        pass

    @abstractmethod
    def exec_streaming(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str,  temperature: float, max_tokens: int) -> Iterable[str]:
        pass

    @abstractmethod
    def get_chat_history(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str) -> Any:
        pass


class OpenAIProvider(LLMProvider):
    # Non chat models
    COMPLETION_MODELS = {"gpt-3.5-turbo-instruct",
                         "davinci-002", "babbage-002"}

    def get_chat_history(self, model: str, chat_history: List[Tuple[str]], system_prompt: str) -> List[dict]:
        if model in OpenAIProvider.COMPLETION_MODELS:
            history = self.__get_chat_history(chat_history, system_prompt)
            return self.custom_chat_template(history)
        else:
            return self.__get_chat_history(chat_history, system_prompt)

    def __get_chat_history(self, chat_history: List[Tuple[str, str]], system_prompt: str) -> List[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        for user_prompt, assistant_response in chat_history:
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            if assistant_response:
                messages.append(
                    {"role": "assistant", "content": assistant_response})

        return messages

    def custom_chat_template(self, chat_history: List[dict]):
        if len(chat_history) == 2:
            prompt = f"{chat_history[0]['content']}\n\n{chat_history[1]['content']}"
        else:
            prompt = f"{chat_history[0]['content']}\n\n"
            prompt += '\n'.join([
                f'{"Assistant:" if i % 2 == 0 else "User:"} {chat["content"]}' for i, chat in enumerate(chat_history)
            ])

        return prompt

    def exec(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str,  temperature: float, max_tokens: int) -> str:
        if model in self.COMPLETION_MODELS:
            return self.exec_completion(model, chat_history, system_prompt, temperature, max_tokens)

        response = OpenAI_client.chat.completions.create(
            model=model,
            messages=self.get_chat_history(model, chat_history, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )

        return response.choices[0].message.content

    def exec_streaming(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str, temperature: float, max_tokens: int) -> Iterable[str]:
        if model in self.COMPLETION_MODELS:
            yield from self.exec_streaming_completion(model, chat_history, system_prompt, temperature, max_tokens)
            return

        stream = OpenAI_client.chat.completions.create(
            model=model,
            messages=self.get_chat_history(model, chat_history, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        yield from (chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content is not None)

    def exec_completion(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str, temperature: float, max_tokens: int) -> str:
        response = OpenAI_client.completions.create(
            model=model,
            prompt=self.get_chat_history(model, chat_history, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].text

    def exec_streaming_completion(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str, temperature: float, max_tokens: int) -> Iterable[str]:

        stream = OpenAI_client.completions.create(
            model=model,
            prompt=self.get_chat_history(model, chat_history, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        yield from (chunk.choices[0].text for chunk in stream if chunk.choices[0].text is not None)


class MistralProvider(LLMProvider):
    def get_chat_history(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str) -> List[dict]:
        messages = [ChatMessage(role="system", content=system_prompt)]
        for user_prompt, assistant_response in chat_history:
            if user_prompt:
                messages.append(ChatMessage(role='user', content=user_prompt))
            if assistant_response:
                messages.append(ChatMessage(role='assistant',
                                content=assistant_response))
        return messages

    def exec(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str, temperature: float, max_tokens: int) -> str:
        chat_response = Mistral_client.chat(
            model=model,
            messages=self.get_chat_history(model, chat_history, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            safe_prompt=False
        )

        return chat_response.choices[0].message.content

    def exec_streaming(self, model: str, chat_history: List[dict], system_prompt: str, temperature: float, max_tokens: int) -> Iterable[str]:
        stream = Mistral_client.chat_stream(
            model=model,
            messages=self.get_chat_history(model, chat_history, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            safe_prompt=False
        )

        yield from (chunk.choices[0].delta.content for chunk in stream)


class LlamaProvider(LLMProvider):
    REPLICATE_CUSTOM_HOSTED_MODELS = {"meta/llama-2-7b", "meta/llama-2-13b", "meta/llama-2-70b",
                                      "meta/llama-2-7b-chat", "meta/llama-2-13b-chat", "meta/llama-2-70b-chat"}

    def get_chat_history(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str) -> str:
        messages = ""
        if chat_history[0][1] == "": 
                messages = f"{chat_history[0][0]} [/INST]"
        else:
            for user, assistant in chat_history:
                messages += f" {user} [/INST]"
                if assistant:
                    messages += f" {assistant}</s><s>[INST]"
        return messages[:-len(" [/INST]")]
    
    def exec(self, model: str, chat_history: List[dict], system_prompt: str,  temperature: float, max_tokens: int) -> str:
        if ':' in model and model.split(":")[0] in LlamaProvider.REPLICATE_CUSTOM_HOSTED_MODELS:
            model = model.split(":")[0]
  
        if "chat" in model or "instruct" in model:
            #prompt_template = f"<s>[INST] <<SYS>> {{system_prompt}} <</SYS>>\n {{prompt}}"
            #prompt_template = f"[INST] <<SYS>> {{system_prompt}} <</SYS>>\n {{prompt}}"
            prompt = self.get_chat_history(model, chat_history, system_prompt)
            output = Llama_client.run(
                model,
                input={
                    "debug": False,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    #"prompt_template": prompt_template,
                }
            )            
        else:
            prompt = str(chat_history[0][0])
            output = Llama_client.run(
                model,
                input={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                }
            )

        return ''.join(output)

    def exec_streaming(self, model: str, chat_history: List[dict], system_prompt: str,  temperature: float, max_tokens: int) -> Iterable[str]:
        if ':' in model and model.split(":")[0] in LlamaProvider.REPLICATE_CUSTOM_HOSTED_MODELS:
            model = model.split(":")[0]

        if "chat" in model or "instruct" in model:
            #prompt_template = f"<s>[INST] <<SYS>> {{system_prompt}} <</SYS>>\n {{prompt}}"
            #prompt_template = f"[INST] <<SYS>> {{system_prompt}} <</SYS>>\n {{prompt}}"
            prompt = self.get_chat_history(model, chat_history, system_prompt)
            stream = Llama_client.stream(
                model,
                input={
                    "debug": False,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    #"prompt_template": prompt_template,
                },
            )
        else:
            prompt = str(chat_history[0][0])
            
            stream = Llama_client.stream(
                model,
                input={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                },
            )


        yield from (str(event) for event in stream)


class GoogleProvider(LLMProvider):
    def get_chat_history(self, model: str, chat_history: List[Tuple[str, str]], system_prompt: str) -> str:
        prompt = ""
        if 'gemma' in model and ('it' in model or len(chat_history) > 1):
            prompt = f"{system_prompt}\n"
            prompt += '\n'.join(f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>" for user, assistant in chat_history[:-1])
            prompt += f"\n<start_of_turn>user\n{chat_history[-1][0]}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"
        elif 'gemma' in model:  # Not instruction tuned with single param
            prompt = f"{system_prompt}\n\n{chat_history[0][0]}"
        elif 'flan' in model:
            prompt = f"{system_prompt}\n\n"
            prompt += '\n'.join(f"Question: {user}\nAnswer: {assistant}" for user, assistant in chat_history)
        else:
            raise ValueError(f"Model is not supported {model}")

        return prompt

    def exec(self, model: str, chat_history: str, system_prompt: str,  temperature: float, max_tokens: int) -> str:
        output = Llama_client.run(
            model,
            input={
                "prompt": self.get_chat_history(model, chat_history, system_prompt),
                "temperature": temperature,
                "max_new_tokens": max_tokens,
            }
        )

        return ''.join(output)

    def exec_streaming(self, model: str, chat_history: List[dict], system_prompt: str,  temperature: float, max_tokens: int) -> Iterable[str]:
        if model.startswith("replicate/flan-t5-xl"):
            logging.warning(
                "The Flan t5 model does not support output streaming.")
            res = self.exec(model, chat_history, system_prompt,
                            temperature, max_tokens)
            yield from iter(word_splitter.split(res))
            return

        stream = Llama_client.stream(
            model,
            input={
                "prompt": self.get_chat_history(model, chat_history, system_prompt),
                "temperature": temperature,
                "max_new_tokens": max_tokens
            },
        )

        yield from (str(event) for event in stream)


class Prompt:
    MODEL_FAMILIES = Literal["GPT", "Mistral", "Llama", "Google", "Misc"]
    OPENAI_MODELS = Literal["gpt-3.5-turbo", "gpt-4-turbo-preview",
                            "gpt-4", "gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"]
    MISTRAL_MODELS = Literal["mistral-small-latest", "mistral-large-latest",
                             "mistral-medium-latest", "open-mistral-7b", "open-mixtral-8x7b", "mistral-tiny"]
    LLAMA_MODELS = Literal["llama-7b", "llama-2-7b-chat", "llama-2-7b", "llama-2-70b-chat", "llama-2-70b", "llama-2-13b-chat", "llama-2-13b", "codellama-7b-python", "codellama-7b-instruct", "codellama-7b",
                           "codellama-70b-python", "codellama-70b-instruct", "codellama-70b", "codellama-34b-python", "codellama-34b-instruct", "codellama-34b", "codellama-13b-python", "codellama-13b-instruct", "codellama-13b"]
    GOOGLE_MODELS = Literal["gemma-2b", "gemma-2b-it",
                            "gemma-7b", "gemma-7b-it", "flan-t5-xl"]
    MODELS = Union[OPENAI_MODELS, MISTRAL_MODELS, LLAMA_MODELS, GOOGLE_MODELS]

    model = "mistral-tiny"
    model_family: MODEL_FAMILIES = "Mistral"
    __provider: LLMProvider = MistralProvider()

    __system_prompt = "You are a helpful assistant"
    __temperature = 0.7
    __max_tokens = 512
    __verbose = False

    REPLICATE_MODEL_MAP = {
        model.name: f"{model.owner}/{model.name}:{model.default_example.version}" if model.default_example and model.default_example.version else f"{model.owner}/{model.name}"
        for model in Llama_client.collections.get("streaming-language-models").models
        if model.owner in ["meta", "google-deepmind", "replicate"]
    }

    @classmethod
    def exec(cls, chat_history: Union[str, List[Tuple[str, str]]], system_prompt: str = None, temperature: float = None, max_tokens: int = None, verbose: Optional[bool] = None) -> str:
        _chat_history = [(chat_history, '')] if type(
            chat_history) is str else chat_history
        _system_prompt = system_prompt or cls.__system_prompt
        _temperature = temperature or cls.__temperature
        _max_tokens = max_tokens or cls.__max_tokens
        _verbose = cls.__verbose if verbose is None else verbose

        if _verbose:
            print(f"\nModel: {cls.model_family} {cls.model}")
            print(f"System Prompt: {_system_prompt}")
            print(f"Temperature: {_temperature}")
            print(f"Max Tokens: {_max_tokens}")
            print("Chat History: ", end="")
            pprint(_chat_history, compact=True)
            print("Prompt: ", end="")
            pprint(cls.__provider.get_chat_history(
                cls.model, _chat_history, _system_prompt), compact=True)
            print()

        res = cls.__provider.exec(
            cls.model, _chat_history, _system_prompt, _temperature, _max_tokens)

        """if _verbose: quoted by EE
            print(f"Response: {res}")"""

        return res

    @classmethod
    def exec_streaming(cls, chat_history: Union[str, List[Tuple[str, str]]], system_prompt: str = None, temperature: float = None, max_tokens: int = None, verbose: Optional[bool] = None) -> Iterable[str]:
        _chat_history = [(chat_history, '')] if type(
            chat_history) is str else chat_history
        _system_prompt = system_prompt or cls.__system_prompt
        _temperature = temperature or cls.__temperature
        _max_tokens = max_tokens or cls.__max_tokens
        _verbose = cls.__verbose if verbose is None else verbose

        if _verbose:
            print(f"\nModel: {cls.model_family} {cls.model}")
            print(f"System Prompt: {_system_prompt}")
            print(f"Temperature: {_temperature}")
            print(f"Max Tokens: {_max_tokens}")
            print("Chat History: ", end="")
            pprint(_chat_history, compact=True)
            print("Prompt: ", end="")
            pprint(cls.__provider.get_chat_history(
                cls.model, _chat_history, _system_prompt), compact=True)
            print()

        res_stream = cls.__provider.exec_streaming(
            cls.model, _chat_history, _system_prompt, _temperature, _max_tokens)

        """if _verbose:  # quoted by EE
            def verbose_stream(stream):
                print("Response: ", end="")
                for chunk in stream:
                    if _verbose:
                        print(chunk, end="")
                    yield chunk
                print()

            yield from verbose_stream(res_stream)
        else:
            yield from res_stream"""

        yield from res_stream

    @overload
    @classmethod
    def set_model(cls, model_family: Literal["GPT"], model: OPENAI_MODELS): ...

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
            case 'GPT': cls.__provider = OpenAIProvider()
            case 'Mistral': cls.__provider = MistralProvider()
            case 'Llama': cls.__provider = LlamaProvider()
            case 'Google': cls.__provider = GoogleProvider()
            case _: raise ValueError("Model family is unknown or incorrectly set")

        cls.model_family = model_family
        if model == "mistral-large":
            model = "mistral-large-latest"

        cls.model = model
        if model_family in ["Llama", "Google"]:
            cls.model = cls.REPLICATE_MODEL_MAP[model]

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
        openai_models = OpenAI_client.models.list()
        mistral_models = Mistral_client.list_models()
        llama_models = Llama_client.collections.get(
            "streaming-language-models").models

        return {
            "GPT": [model["id"] for model in openai_models.model_dump()["data"]],
            "Mistral": [model["id"] for model in mistral_models.model_dump()["data"]],
            "Replicate": [model.name for model in llama_models]
        }

    @classmethod
    def print_models(cls):
        available_models = cls.get_available_models()
        for model_family, models in available_models.items():
            models = list(set(models))
            # color the output
            print(f"{Fore.MAGENTA}[{model_family}]:")
            for model in models:
                print(f"\t{Fore.GREEN}{model}")
            print("\n")

    @classmethod
    def print_useful_models(cls):
        available_models = cls.get_available_models()
        for model_family, models in available_models.items():
            models = list(set(models))
            # color the output
            print(f"{Fore.MAGENTA}[{model_family}]:")
            for model in models:
                if model_family in ["GPT", "Mistral"] or model in get_args(cls.LLAMA_MODELS) or model in get_args(cls.GOOGLE_MODELS):
                    print(f"\t{Fore.GREEN}{model}")
            print("\n")
    # added by EE finishes

# Only cache if the response tuple is OK


@cached(pred=lambda res: res[0])
def test(model_family, model, stream=False) -> Tuple[bool, str]:

    ok = True
    try:
        Prompt.set_model(model_family, model)
        Prompt.set_system_prompt("Respond:")
        Prompt.set_max_tokens(10)

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

    if ok:
        result = f"{Fore.GREEN}OK{Fore.RESET} ({len(res.split())}:{len(res)}) in {Fore.YELLOW}{duration_seconds:.2f}s{Fore.RESET}"
    else:
        result = f"{Fore.RED}FAILED{Fore.RESET}"

    return (ok, (
        f"{Fore.MAGENTA}[{model_family}]{Fore.BLUE}[{model}]{Fore.RESET}"
        f"{Fore.LIGHTBLACK_EX}[{'Stream' if stream else 'Normal'}]{Fore.RESET}\t"
        f"{result}"
    ))


test_set = [
    ('OpenAI', 'gpt-3.5-turbo'),
    ('OpenAI', 'gpt-4-turbo-preview'),
    ('OpenAI', 'gpt-4'),
    ('OpenAI', 'gpt-3.5-turbo-instruct'),
    ('OpenAI', 'davinci-002'),
    ('OpenAI', 'babbage-002'),
    ('Mistral', 'mistral-small-latest'),
    ('Mistral', 'mistral-large-latest'),
    ('Mistral', 'mistral-medium-latest'),
    ('Mistral', 'open-mistral-7b'),
    ('Mistral', 'open-mixtral-8x7b'),
    ('Mistral', 'mistral-tiny'),
    ('Llama', 'llama-2-7b'),
    ('Llama', 'llama-2-7b-chat'),
    ('Llama', 'llama-2-13b'),
    ('Llama', 'llama-2-13b-chat'),
    ('Llama', 'llama-2-70b'),
    ('Llama', 'llama-2-70b-chat'),
    # ('Llama', 'codellama-7b-python'),
    # ('Llama', 'codellama-7b-instruct'),
    # ('Llama', 'codellama-7b'),
    # ('Llama', 'codellama-70b-python'),
    # ('Llama', 'codellama-70b-instruct'),
    # ('Llama', 'codellama-70b'),
    # ('Llama', 'codellama-34b-python'),
    # ('Llama', 'codellama-34b-instruct'),
    # ('Llama', 'codellama-34b'),
    # ('Llama', 'codellama-13b-python'),
    # ('Llama', 'codellama-13b-instruct'),
    # ('Llama', 'codellama-13b'),
    ('Google', 'gemma-2b'),
    ('Google', 'gemma-2b-it'),
    ('Google', 'gemma-7b'),
    ('Google', 'gemma-7b-it'),
    ('Google', 'flan-t5-xl'),
]


def test_all():
    for model_family, model in test_set:
        ok, res = test(model_family, model, stream=False)
        print(res)
        ok, res = test(model_family, model, stream=True)
        print(res)
