"""Own-text generation backends for the fixed fidelity recipe, not hardware drivers."""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request

from fidelity_windows import canonical


def token_ids(tokenizer, text):
    encoded = (tokenizer.encode(text) if getattr(tokenizer, "_fidelity_identity", None)
               else tokenizer.encode(text, add_special_tokens=False))
    return encoded.ids if hasattr(encoded, "ids") else list(encoded)


def normalized_messages(messages):
    messages = copy.deepcopy(messages)
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    return messages


class ChatRenderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.native = getattr(tokenizer, "_fidelity_identity", None)
        if self.native:
            self.identity = {**self.native, "chat_template_kind": "checkpoint_python",
                             "chat_template_sha256": self.native["tokenizer_sha256"]}
        else:
            template = tokenizer.get_chat_template()
            if not template:
                raise ValueError("generation requires the checkpoint's own chat template")
            self.identity = {"chat_template_sha256": hashlib.sha256(template.encode()).hexdigest()}

    def render(self, body, assistant=None):
        messages = normalized_messages(body["messages"])
        if assistant is not None:
            messages += normalized_messages([assistant])
        return self.tokenizer.apply_chat_template(
            messages, tools=body.get("tools"), tokenize=False,
            add_generation_prompt=assistant is None)

    def prompt(self, body):
        if self.native:
            return list(self.tokenizer.apply_chat_template(
                normalized_messages(body["messages"]), tools=body.get("tools"),
                tokenize=True, add_generation_prompt=True))
        return token_ids(self.tokenizer, self.render(body))

    def completion(self, body, assistant):
        prompt, full = self.render(body), self.render(body, assistant)
        if not full.startswith(prompt):
            raise ValueError("checkpoint template rewrites the generation prefix; cannot reconstruct API span")
        text = full[len(prompt):]
        prompt_ids = self.prompt(body)
        if self.native:
            full_ids = list(self.tokenizer.apply_chat_template(
                normalized_messages(body["messages"] + [assistant]), tools=body.get("tools"),
                tokenize=True, add_generation_prompt=False))
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("checkpoint tokenizer rewrites the generation prefix")
            generated = full_ids[len(prompt_ids):]
        else:
            full_ids = token_ids(self.tokenizer, full)
            generated = token_ids(self.tokenizer, text)
        decoded = (self.tokenizer.decode(generated) if self.native else
                   self.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        if decoded != text:
            raise ValueError("generated span does not re-tokenize losslessly")
        # Do not accept a token crossing the conditioning/generated boundary.
        if full_ids != prompt_ids + generated:
            raise ValueError("generated span crosses the prompt token boundary")
        return generated


class ResponseCache:
    def __init__(self, directory, identity):
        self.directory, self.identity = Path(directory), identity
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def binding_key(self, binding):
        backend = binding["backend"]
        if backend.get("text_source") == "api":
            # Hosted responses depend on wire inputs, not local rendering code.
            request = binding["request"]
            return {"endpoint": backend["endpoint"], "model": request["model"],
                    "provider": request.get("provider"), "request": request}
        return binding

    def find(self, body):
        binding = {"backend": self.identity, "request": body}
        expected = self.binding_key(binding)
        path = self.directory / (hashlib.sha256(canonical(expected)).hexdigest() + ".json")
        candidates = [path] if path.exists() else []
        if self.identity.get("text_source") == "api" and not candidates:
            # Read old source-bound records in place; never rewrite their dates or hashes.
            candidates = sorted(self.directory.glob("*.json"))
        for candidate in candidates:
            record = json.loads(candidate.read_bytes())
            if self.binding_key(record["binding"]) == expected:
                return record
            if candidate == path:
                raise ValueError("cached generation binding mismatch")
        return None

    def get(self, body, generate):
        binding = {"backend": self.identity, "request": body}
        key = hashlib.sha256(canonical(self.binding_key(binding))).hexdigest()
        path = self.directory / (key + ".json")
        record = self.find(body)
        if record is not None:
            return record
        response = generate()
        record = {"binding": binding, "response": response,
                  "date": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        # A killed write cannot become a valid cached response. Never persist a key/header.
        temporary = path.with_suffix(f".{os.getpid()}.pending")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(canonical(record) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        return record


class TextGenerator:
    def __init__(self, renderer, cache, *, checkpoint, family, identity):
        self.renderer, self.checkpoint, self.family = renderer, checkpoint, family
        self.identity = {**identity, **renderer.identity, "checkpoint": checkpoint}
        self.cache = ResponseCache(cache, self.identity)

    def prompt(self, body):
        return {"probe": {"prompt_ids": self.renderer.prompt(body)}}

    def result(self, body, generated, message, provenance):
        if not generated:
            raise ValueError("backend returned no generated tokens")
        return {"probe": {"prompt_ids": self.renderer.prompt(body), "generated": generated,
                          "engine": self.identity["text_source"]},
                "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
                "content": message.get("content") or "", "tool_calls": message.get("tool_calls") or [],
                "server": {"model": self.checkpoint, "family": self.family, **self.identity},
                "generation_provenance": provenance}


class APIClient(TextGenerator):
    def __init__(self, renderer, cache, *, checkpoint, family, base, model, key_file,
                 timeout=1800, retries=4, extra_body=None, identity=None):
        parts = urllib.parse.urlsplit(base)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("API base must not contain credentials, query or fragment")
        if parts.scheme != "https" and not (parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1")):
            raise ValueError("API generation requires HTTPS (except a local test server)")
        self.endpoint = base.rstrip("/") + "/chat/completions"
        self.model, self.key_file = model, Path(key_file)
        self.timeout, self.retries, self.extra_body = timeout, retries, extra_body or {}
        if set(self.extra_body) & {"model", "messages", "temperature", "max_tokens", "stream", "tools"}:
            raise ValueError("API options may not override the fixed generation recipe")
        super().__init__(renderer, cache, checkpoint=checkpoint, family=family,
                         identity={**(identity or {}), "text_source": "api", "endpoint": self.endpoint, "model_id": model})

    def request(self, body):
        if stat.S_IMODE(self.key_file.stat().st_mode) != 0o600:
            raise ValueError("API key file must have mode 0600")
        key = self.key_file.read_text().strip()
        if not key:
            raise ValueError("empty API key file")
        request = urllib.request.Request(self.endpoint, canonical(body),
            {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                error.close()
                if error.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    # Providers can echo credentials/request text in error bodies.
                    raise ValueError(f"generation API HTTP {error.code}") from None
                time.sleep(min(60, 2 ** attempt))
        raise AssertionError("unreachable")

    def __call__(self, body):
        request = copy.deepcopy(body)
        request["model"] = self.model
        request.update(self.extra_body)
        record = self.cache.find(request)
        if record is None:
            if urllib.parse.urlsplit(self.endpoint).hostname == "openrouter.ai":
                # Reuse old exact requests first; budget only newly purchased responses.
                reasoning_options = copy.deepcopy(request.get("reasoning") or {})
                reasoning_options.pop("effort", None)
                reasoning_options["max_tokens"] = body["max_tokens"]
                request["reasoning"] = reasoning_options
                request.pop("reasoning_effort", None)
            record = self.cache.get(request, lambda: self.request(request))
        raw = record["response"]
        if raw.get("error") or not raw.get("choices"):
            raise ValueError("API returned an error or no choices (raw response cached)")
        choice = raw["choices"][0]
        message = copy.deepcopy(choice["message"])
        message["role"] = "assistant"
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        message["reasoning_content"] = reasoning
        if not message.get("content") and not reasoning and not message.get("tool_calls"):
            raise ValueError("API returned no accessible text or tool calls")
        generated_full = self.renderer.completion(body, message)
        generated = generated_full[:body["max_tokens"]]
        visible_message = message
        if len(generated) < len(generated_full):
            # A cut response is unfinished; never execute calls from its discarded suffix.
            visible_message = {**message, "tool_calls": []}
        return self.result(body, generated, visible_message, {
            **self.identity, "date": record["date"], "provider": raw.get("provider"),
            "response_model": raw.get("model"), "response_id": raw.get("id"),
            "usage": raw.get("usage"), "finish_reason": choice.get("finish_reason"),
            "reasoning_content": reasoning, "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "recipe_options": {k: v for k, v in record["binding"]["request"].items()
                               if k in self.extra_body or k == "reasoning"},
            "cached_generator_sha256": record["binding"]["backend"].get("generator_sha256"),
            "current_generator_sha256": self.identity.get("generator_sha256"),
            "raw_response_sha256": hashlib.sha256(canonical(raw)).hexdigest(),
            "generated_ids_full": len(generated_full), "generated_ids_kept": len(generated),
            "reconstructed_tokens": len(generated), "retokenization": "lossless full completion; checkpoint chat template; recipe ID prefix"})


def parse_qwen_response(text, tools, *, thinking_prefix=False):
    """Parse the checkpoint's Qwen XML envelope; the fixture validates and executes calls."""
    reasoning = ""
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
    elif thinking_prefix:
        return {"role": "assistant", "reasoning_content": text, "content": "", "tool_calls": []}
    calls = []
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools or []}
    for index, match in enumerate(re.finditer(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", text, re.S)):
        name, parameters = match.groups()
        arguments = {}
        for key, value in re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", parameters, re.S):
            value = value.strip("\n")
            kind = schemas.get(name, {}).get("properties", {}).get(key, {}).get("type")
            if kind != "string":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Retain bad arguments for the fixture runner's normal error path.
            arguments[key] = value
        calls.append({"id": f"call_{index}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(arguments)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    return {"role": "assistant", "reasoning_content": reasoning, "content": content, "tool_calls": calls}


class HFClient(TextGenerator):
    def __init__(self, renderer, cache, *, checkpoint, family, snapshot, identity):
        if family != "qwen3_5":
            raise ValueError("local HF generation currently supports the dense Qwen reference only")
        self.snapshot, self.model = Path(snapshot), None
        super().__init__(renderer, cache, checkpoint=checkpoint, family=family,
                         identity={**identity, "text_source": "hf", "dtype": "bf16", "do_sample": False})

    def generate(self, body):
        import importlib.util
        import torch
        if self.model is None:
            path = Path(__file__).resolve().parents[2] / "scripts/qualify/qwen3_5/reference.py"
            spec = importlib.util.spec_from_file_location("qwen35_generation_reference", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            golden_path = Path(__file__).resolve().parent / "families/qwen3_5/golden.py"
            golden_spec = importlib.util.spec_from_file_location("qwen35_generation_golden", golden_path)
            golden = importlib.util.module_from_spec(golden_spec)
            golden_spec.loader.exec_module(golden)
            _, _, _, source = module.reference_identity(self.snapshot)
            golden.verify_reference_source(source)
            torch.manual_seed(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            from transformers.models.qwen3_5 import modeling_qwen3_5 as reference
            from shape_invariant import install_eager
            install_eager(reference)
            self.model, _ = module.load_reference(self.snapshot)
            self.model.eval()
        prompt = self.renderer.prompt(body)
        ids = torch.tensor([prompt], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         do_sample=False, max_new_tokens=body["max_tokens"], use_cache=True)
        return {"generated": output[0, len(prompt):].tolist()}

    def __call__(self, body):
        record = self.cache.get(body, lambda: self.generate(body))
        generated = record["response"]["generated"]
        text = self.renderer.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if token_ids(self.renderer.tokenizer, text) != generated:
            raise ValueError("HF generated span does not re-tokenize losslessly")
        # The stop token is scored, but is not assistant content on the next fixture turn.
        for stop in ("<|im_end|>", "<|endoftext|>"):
            text = text.removesuffix(stop)
        message = parse_qwen_response(text, body.get("tools"),
                                     thinking_prefix=self.renderer.render(body).endswith("<think>\n"))
        return self.result(body, generated, message, {
            **self.identity, "date": record["date"], "reasoning_content": message["reasoning_content"],
            "content": message["content"], "tool_calls": message["tool_calls"],
            "raw_response_sha256": hashlib.sha256(canonical(record["response"])).hexdigest(),
            "recipe_deviations": ["JSON response_format is an instruction, not a constrained decoder"]
                if body.get("response_format") else [], "retokenization": "lossless; native generated IDs"})
