#!/usr/bin/env python3
"""CPU-only MiniMax recursive XML matcher/mask qualification.

Uses the actual checkpoint tokenizer and unmodified template. Accepted strings
are checkpoint renderings, not another engine's MiniMax XML spelling. Emits
schema/wire/expected-argument rows for the Rust constrained round-trip test.
"""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

NS = "]<]minimax[>["
VOCAB = 200064


def obj(properties, required=None, **extra):
    return dict(type="object", properties=properties,
                required=list(properties) if required is None else required,
                additionalProperties=False, **extra)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-lib", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    tokenizer = Tokenizer.from_file(str(a.snapshot / "tokenizer.json"))
    template = _compile_jinja_template((a.snapshot / "chat_template.jinja").read_text())
    lib = c.CDLL(str(a.native_lib.resolve()))
    P = c.c_void_p
    signatures = {
        "compiler_create": [c.c_char_p, c.c_size_t, c.POINTER(c.c_int32), c.c_size_t, c.POINTER(P), P, c.c_size_t],
        "compile": [P, c.c_int, c.c_char_p, c.c_int, c.POINTER(P), P, c.c_size_t],
        "matcher_create": [P, c.POINTER(P), P, c.c_size_t],
        "matcher_accept_token": [P, c.c_uint32, c.POINTER(c.c_int), P, c.c_size_t],
        "matcher_is_completed": [P, c.POINTER(c.c_int), P, c.c_size_t],
        "matcher_fill_bitmask": [P, c.POINTER(c.c_uint32), c.c_size_t, c.POINTER(c.c_int), P, c.c_size_t],
        "matcher_destroy": [P], "grammar_destroy": [P], "compiler_destroy": [P],
    }
    for name, signature in signatures.items():
        fn = getattr(lib, "cuteafd_xgrammar_" + name)
        fn.argtypes = signature
        fn.restype = c.c_int

    def invoke(name, *args, error=True):
        buffer = c.create_string_buffer(8192)
        status = getattr(lib, "cuteafd_xgrammar_" + name)(*args, *([buffer, len(buffer)] if error else []))
        if status:
            raise RuntimeError((name, status, buffer.value.decode()))

    def wire(schema, values):
        tool = dict(type="function", function=dict(name="lookup", parameters=schema))
        rendered = template.render(messages=[dict(role="assistant", content=None, tool_calls=[
            dict(function=dict(name="lookup", arguments=values))])], tools=[tool], add_generation_prompt=False)
        start = rendered.rindex(f'{NS}<invoke name="lookup">') + len(f'{NS}<invoke name="lookup">')
        return rendered[start:rendered.index(f"{NS}</invoke>", start)]

    cases = []

    def case(name, schema, values, bad=()):
        cases.append((name, schema, values, list(bad)))

    case("string", obj({"s": {"type": "string"}}), {"s": '001 "quoted" <b> text'})
    case("integer", obj({"n": {"type": "integer", "minimum": 1, "maximum": 7}}), {"n": 3},
         [f"{NS}<n>8{NS}</n>", f"{NS}<n>three{NS}</n>"])
    case("number", obj({"n": {"type": "number", "minimum": -2, "maximum": 4}}), {"n": 1.25})
    case("boolean", obj({"flag": {"type": "boolean"}}), {"flag": True}, [f"{NS}<flag>yes{NS}</flag>"])
    case("control-markers-excluded", obj({"s": {"type": "string"}}), {"s": "safe"},
         [f"{NS}<s>a{marker}b{NS}</s>" for marker in ["[e~[", "]~b]", "]~!b[", "]!d~[", "]!p~[", "<mm:think>", "</mm:think>", NS]])
    case("optional-null-omitted", obj({"nil": {"type": "null"}}, required=[]), {"nil": None})
    case("nested-object", obj({"cfg": obj({"enabled": {"type": "boolean"}, "code": {"type": "string"}})}),
         {"cfg": {"enabled": False, "code": "001"}})
    case("nested-array", obj({"rows": {"type": "array", "items": obj({"n": {"type": "integer"}}), "minItems": 1, "maxItems": 2}}),
         {"rows": [{"n": 1}, {"n": 2}]})
    case("array-of-arrays", obj({"rows": {"type": "array", "items": {"type": "array", "items": {"type": "boolean"}}}}),
         {"rows": [[True], [False, True]]})
    case("empty-containers", obj({"a": {"type": "array", "items": {"type": "integer"}}, "o": obj({})}), {"a": [], "o": {}})
    case("same-name-recursion", obj({"x": obj({"x": {"type": "string"}})}), {"x": {"x": "yes"}})
    case("enum-string", obj({"mode": {"enum": ["yes", "no"]}}), {"mode": "yes"}, [f"{NS}<mode>other{NS}</mode>"])
    case("enum-number", obj({"mode": {"enum": [1, 2]}}), {"mode": 2}, [f"{NS}<mode>3{NS}</mode>"])
    case("string-union", obj({"v": {"anyOf": [{"type": "string", "pattern": "^[A-Z]{2}$"}, {"type": "integer"}]}}), {"v": "AB"},
         [f"{NS}<v>12{NS}</v>"])
    case("numeric-union", obj({"v": {"type": ["integer", "boolean"]}}), {"v": 2})
    case("nullable-integer", obj({"n": {"type": ["integer", "null"]}}), {"n": 4})
    case("coding-agent-nullable-anyof", obj({"command": {"type": "string"},
        "timeout_ms": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]}}),
        {"command": "git status", "timeout_ms": 1000})
    case("pattern", obj({"id": {"type": "string", "pattern": "^[A-Z]{2}[0-9]$"}}), {"id": "AB2"},
         [f"{NS}<id>bad{NS}</id>"])
    case("pattern-and-length", obj({"s": {"type": "string", "pattern": "^[a-z]+$", "minLength": 2, "maxLength": 4}}), {"s": "abcd"},
         [f"{NS}<s>a{NS}</s>", f"{NS}<s>abcde{NS}</s>", f"{NS}<s>AB{NS}</s>"])
    case("string-length", obj({"s": {"type": "string", "minLength": 2, "maxLength": 4}}), {"s": "abcd"},
         [f"{NS}<s>a{NS}</s>", f"{NS}<s>abcde{NS}</s>"])
    case("reference", obj({"v": {"$ref": "#/$defs/value"}}, **{"$defs": {"value": {"type": "integer"}}}), {"v": 7})
    case("additional-only-empty-subset", {"type": "object", "additionalProperties": True}, {})
    case("no-parameters", obj({}), {}, [f"{NS}<unknown>1{NS}</unknown>"])

    report = dict(passed=False, native_lib_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest(), cases=[], refusals=[])
    compiler = P()
    stops = (c.c_int32 * 1)(200020)
    invoke("compiler_create", str(a.snapshot / "tokenizer.json").encode(), VOCAB, stops, 1, c.byref(compiler))
    try:
        for name, schema, expected, bad in cases:
            grammar = P()
            spec = dict(type="structural_tag", format=dict(type="minimax_tool_schema", json_schema=schema, strict=True))
            invoke("compile", compiler, 3, json.dumps(spec).encode(), 1, c.byref(grammar))
            good = wire(schema, expected)
            record = dict(name=name, schema=schema, expected={k: v for k, v in expected.items() if v is not None}, arguments_wire=good, checks=[])
            report["cases"].append(record)
            # Every nonempty case also rejects a mismatched closing tag and a duplicate field.
            malformed = good.replace(f"{NS}</", f"{NS}</wrong", 1) if good else "x"
            if good:
                bad += [malformed, good + good]
            try:
                for valid, texts in [(True, [good]), (False, bad)]:
                    for text in texts:
                        matcher = P()
                        invoke("matcher_create", grammar, c.byref(matcher))
                        accepted = True
                        try:
                            for token in tokenizer.encode(text, add_special_tokens=False).ids:
                                mask = (c.c_uint32 * ((VOCAB + 31) // 32))()
                                needs = c.c_int()
                                invoke("matcher_fill_bitmask", matcher, mask, len(mask), c.byref(needs))
                                allowed = not needs.value or bool(mask[token // 32] & (1 << (token % 32)))
                                result = c.c_int()
                                invoke("matcher_accept_token", matcher, token, c.byref(result))
                                assert bool(result.value) == allowed, (name, token, "mask mismatch")
                                if not result.value:
                                    accepted = False
                                    break
                            complete = c.c_int()
                            invoke("matcher_is_completed", matcher, c.byref(complete))
                            actual = accepted and bool(complete.value)
                            record["checks"].append(dict(text=text, expected=valid, accepted_complete=actual))
                            assert actual == valid, (name, text, valid, actual)
                        finally:
                            invoke("matcher_destroy", matcher, error=False)
            finally:
                invoke("grammar_destroy", grammar, error=False)
            print("PASS", name, flush=True)
        for name, schema in [
            ("required-null", obj({"nil": {"type": "null"}})),
            ("required-undeclared", {"type": "object", "required": ["missing"], "additionalProperties": True}),
            ("additional-minimum", {"type": "object", "minProperties": 1, "additionalProperties": True}),
            ("nested-required-null", obj({"cfg": obj({"nil": {"type": "null"}})})),
            ("required-null-reference", obj({"nil": {"$ref": "#/$defs/nil"}}, **{"$defs": {"nil": {"type": "null"}}})),
            ("container-union", obj({"v": {"type": ["object", "string"]}})),
            ("unknown-dependent-required", {"type": "object", "additionalProperties": True, "dependentRequired": {"a": ["b"]}}),
        ]:
            grammar = P()
            spec = dict(type="structural_tag", format=dict(type="minimax_tool_schema", json_schema=schema, strict=True))
            try:
                invoke("compile", compiler, 3, json.dumps(spec).encode(), 1, c.byref(grammar))
            except RuntimeError as error:
                assert "MiniMax" in str(error)
                report["refusals"].append(dict(name=name, error=str(error)))
            else:
                invoke("grammar_destroy", grammar, error=False)
                raise AssertionError((name, "must refuse"))
        report["passed"] = True
    finally:
        invoke("compiler_destroy", compiler, error=False)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(cases)} cases, {len(report['refusals'])} build-time refusals")


if __name__ == "__main__":
    main()
