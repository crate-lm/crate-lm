import asyncio
import contextlib
import os
import random
import traceback
from asyncio import Semaphore
from functools import wraps
from typing import Any, Callable, Optional

import httpx
import orjson


def is_api_error(err: Exception) -> bool:
    if isinstance(err, httpx.HTTPStatusError):
        response = err.response
        error_data = response.json().get("error", {})
        error_message = error_data.get("message")
        if response.status_code in [400, 404, 415]:
            if error_data.get("type") == "idempotency_error":
                print(f"Retrying after idempotency error: {error_message} ({response.url})")
                return True
            else:
                # Invalid request
                return False
        else:
            print(f"Retrying after API error: {error_message} ({response.url})")
            return True

    elif isinstance(err, httpx.ConnectError):
        print(f"Retrying after connection error... ({err.request.url})")
        return True

    elif isinstance(err, httpx.TimeoutException):
        print(f"Retrying after a timeout error... ({err.request.url})")
        return True

    elif isinstance(err, httpx.ReadError):
        print(f"Retrying after a read error... ({err.request.url})")
        return True
    
    else:
        print("error:", err, type(err))
        exit()

    print(f"Retrying after an unexpected error: {repr(err)}")
    traceback.print_tb(err.__traceback__)
    return True


def exponential_backoff(
    retry_on: Callable[[Exception], bool] = lambda err: True
) -> Callable[[Callable], Callable]:
    """
    Returns a decorator which retries the wrapped function as long as the specified retry_on
    function returns True for the exception, applying exponential backoff with jitter after
    failures, up to a retry limit.
    """
    init_delay_s = 1.0
    max_delay_s = 10.0
    # Roughly 30 minutes before we give up.
    max_tries = 200
    backoff_multiplier = 2.0
    jitter = 0.2

    def decorate(f: Callable) -> Callable:
        assert asyncio.iscoroutinefunction(f)

        @wraps(f)
        async def f_retry(*args: Any, **kwargs: Any) -> None:
            delay_s = init_delay_s
            for i in range(max_tries):
                try:
                    return await f(*args, **kwargs)
                except Exception as err:
                    if not retry_on(err) or i == max_tries - 1:
                        raise
                    jittered_delay = random.uniform(delay_s * (1 - jitter), delay_s * (1 + jitter))
                    await asyncio.sleep(jittered_delay)
                    delay_s = min(delay_s * backoff_multiplier, max_delay_s)

        return f_retry

    return decorate


API_KEY = os.getenv("OPENAI_API_KEY")
assert API_KEY, "Please set the OPENAI_API_KEY environment variable"
API_HTTP_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + API_KEY,
}
BASE_API_URL_OPENAI = "https://api.openai.com/v1"
BASE_API_URL_LLAMA = "http://localhost:8000/v1"

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
llama_path = "/home/ubuntu/nanogpt4crate/automated-interpretability/llama-2-7b-hf"
simulation_tokenizer = AutoTokenizer.from_pretrained(llama_path)
simulation_model = AutoModelForCausalLM.from_pretrained(llama_path, device_map='auto', quantization_config=quantization_config)

mistral_path = 'mistralai/Mistral-7B-Instruct-v0.2'
explanation_tokenizer = AutoTokenizer.from_pretrained(mistral_path)
explanation_model = AutoModelForCausalLM.from_pretrained(mistral_path, device_map='auto', quantization_config=quantization_config)

def mimic_api_explanation(kwargs, zeroablate=False):
    system_message = kwargs['messages'][0]['content']
    messages = kwargs['messages'][1:]
    messages[0]['content'] = system_message + " " + messages[0]['content']
    input_tokens = explanation_tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    generated_ids = explanation_model.generate(input_tokens, max_new_tokens=kwargs['max_tokens'], num_return_sequences=kwargs['n'], do_sample=True, temperature=kwargs['temperature'], top_p=kwargs['top_p'], pad_token_id=explanation_tokenizer.pad_token_id)
    output_tokens = explanation_tokenizer.decode(generated_ids[0]).split("[/INST]")[-1].split("</s>")[0]
    if zeroablate:
        output_tokens = "I can't determine what this neuron is about."
    
    response_json = {}
    response_json['id'] = ''
    response_json['object'] = ''
    response_json['created'] = ''
    response_json['model'] = 'mistral-7b'
    response_json['choices'] = [{'index': 0, 'message': {'role': 'assistant', 'content': output_tokens, 'logprobs': None, 'finish_reason': 'stop'}}]
    # response_json['usage'] = {'prompt_tokens': len(input_tokens[0]), 'total_tokens': len(generated_ids[0])}
    response_json['system_fingerprint'] = ''
    return response_json
    
    
def mimic_api_simulation(prompt):
    input_tokens = simulation_tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    logits = simulation_model(input_tokens).logits[0]
    top_logprobs = []
    tokens = []
    text_offset = 0
    text_start_id = 0
    with torch.no_grad():
        for i, token_logits in enumerate(logits):
            top5_token_id = torch.topk(token_logits, 5).indices
            top5_tokens = simulation_tokenizer.decode(top5_token_id)
            top5_token_logprobs = torch.topk(token_logits, 5).values
            top_logprobs.append({simulation_tokenizer.decode([top5_token_id[j]]): top5_token_logprobs[j].item() for j in range(5)})
            token = simulation_tokenizer.decode([input_tokens[0][i]])
            tokens.append(token)
            if token == "start" and simulation_tokenizer.decode([input_tokens[0][i-1]]) == "<" and simulation_tokenizer.decode([input_tokens[0][i+1]]) == ">":
                text_offset = len(simulation_tokenizer.decode(input_tokens[0][1:i+1]))
                text_start_id = i
    text_offsets = [0 if i < text_start_id else text_offset for i in range(len(tokens))]
    tokens = tokens[1:]
    top_logprobs = top_logprobs[:-1]
    top_logprobs[0] = None
    
    response_json = {}
    response_json['id'] = ''
    response_json['object'] = ''
    response_json['created'] = ''
    response_json['model'] = 'llama-2-7b'
    response_json['choices'] = [{'text': prompt, 'index': 0, 'finish_reason': 'length', 'logprobs': {'tokens': tokens, 'token_logprobs': None, 'top_logprobs': top_logprobs, 'text_offset': text_offsets}}]
    response_json['usage'] = {'prompt_tokens': len(tokens), 'total_tokens': len(tokens)}
    return response_json

class ApiClient:
    """Performs inference using the OpenAI API. Supports response caching and concurrency limits."""

    def __init__(
        self,
        model_name: str,
        # If set, no more than this number of HTTP requests will be made concurrently.
        max_concurrent: Optional[int] = None,
        # Whether to cache request/response pairs in memory to avoid duplicating requests.
        cache: bool = False,
    ):
        self.model_name = model_name

        if max_concurrent is not None:
            self._concurrency_check: Optional[Semaphore] = Semaphore(max_concurrent)
        else:
            self._concurrency_check = None

        if cache:
            self._cache: Optional[dict[str, Any]] = {}
        else:
            self._cache = None

    @exponential_backoff(retry_on=is_api_error)
    async def make_request(
        self, timeout_seconds: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        if self._cache is not None:
            key = orjson.dumps(kwargs)
            if key in self._cache:
                return self._cache[key]
        async with contextlib.AsyncExitStack() as stack:
            if self._concurrency_check is not None:
                await stack.enter_async_context(self._concurrency_check)
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=timeout_seconds)
            )
            # If the request has a "messages" key, it should be sent to the /chat/completions
            # endpoint. Otherwise, it should be sent to the /completions endpoint.
            kwargs["model"] = self.model_name
            response = None
            if "messages" in kwargs and 'gpt' in self.model_name:
                base_api_url = BASE_API_URL_LLAMA if "llama" in self.model_name else BASE_API_URL_OPENAI
                url = base_api_url + ("/chat/completions" if "messages" in kwargs else "/completions")
                response = await http_client.post(url, headers=API_HTTP_HEADERS, json=kwargs)
                response_json = response.json()
                # save kwargs and response in this case
                # with open("kwargs.json", "w") as f:
                #     json.dump(kwargs, f)
            elif "messages" in kwargs and 'mistral' in self.model_name:
                response_json = mimic_api_explanation(kwargs)
            elif "messages" in kwargs and 'zeroablate' in self.model_name:
                response_json = mimic_api_explanation(kwargs, zeroablate=True)
            elif "messages" not in kwargs and 'llama' in self.model_name:
                response_json = mimic_api_simulation(kwargs["prompt"])
        # with open("response.json", "w") as f:
        #     import json
        #     json.dump(response_json, f)
        # import IPython; IPython.embed(); exit()
        # The response json has useful information but the exception doesn't include it, so print it
        # out then reraise.
        if response is not None:
            try:
                response.raise_for_status()
            except Exception as e:
                print(response.json())
                raise e
            if self._cache is not None:
                self._cache[key] = response_json
        # print(response_json)
        return response_json


if __name__ == "__main__":

    async def main() -> None:
        client = ApiClient(model_name="llama-2-70b", max_concurrent=1)
        print(await client.make_request(prompt="Why did the chicken cross the road?", max_tokens=9))

    asyncio.run(main())
