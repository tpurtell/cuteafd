#!/usr/bin/env python3
"""Check the native Muse atem tool-parameter grammar against the checkpoint tokenizer.

Each case compiles a `muse_tool_schema` structural tag (the arguments of one
<atem:invoke>) and walks token by token through texts that must be accepted
(the template goldens' calls, typed values, strict schemas) or rejected
(malformed tags, wrong types, undeclared or duplicated parameters). The token
mask and the matcher must agree at every step. CPU only:

  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --network none \\
    -v "$SRC:/work:ro" -v "$OUT:/out" -v "$SNAPSHOT:/snapshot:ro" cuteafd-coordinator-dev:latest \\
    python3 /work/scripts/qualify/muse_spark/qualify-tool-grammar.py \\
      --native-lib /out/libcuteafd_native.so --tokenizer /snapshot/tokenizer.json \\
      --goldens /work/rust/crates/cuteafd-api/src/openai/chat/muse/fixtures/muse_template_goldens.json \\
      --output /out/muse-tool-grammar.json
"""
import argparse
import ctypes as c
import hashlib
import json
import re
from pathlib import Path
from tokenizers import Tokenizer

VOCAB = 202048
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--native-lib', type=Path, required=True)
p.add_argument('--tokenizer', type=Path, required=True)
p.add_argument('--goldens', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
tokenizer = Tokenizer.from_file(str(a.tokenizer))
lib = c.CDLL(str(a.native_lib.resolve()))
P = c.c_void_p
functions = {
    'compiler_create': [c.c_char_p, c.c_size_t, c.POINTER(c.c_int32), c.c_size_t, c.POINTER(P), P, c.c_size_t],
    'compile': [P, c.c_int, c.c_char_p, c.c_int, c.POINTER(P), P, c.c_size_t],
    'matcher_create': [P, c.POINTER(P), P, c.c_size_t],
    'matcher_accept_token': [P, c.c_uint32, c.POINTER(c.c_int), P, c.c_size_t],
    'matcher_is_completed': [P, c.POINTER(c.c_int), P, c.c_size_t],
    'matcher_fill_bitmask': [P, c.POINTER(c.c_uint32), c.c_size_t, c.POINTER(c.c_int), P, c.c_size_t],
    'matcher_destroy': [P], 'grammar_destroy': [P], 'compiler_destroy': [P],
}
for name, args in functions.items():
    fn = getattr(lib, 'cuteafd_xgrammar_' + name); fn.argtypes = args; fn.restype = c.c_int


def invoke(name, *args, error=True):
    buffer = c.create_string_buffer(8192)
    status = getattr(lib, 'cuteafd_xgrammar_' + name)(*args, *([buffer, len(buffer)] if error else []))
    assert status == 0, (name, status, buffer.value.decode())


def esc_attr(text):
    return (text.replace('&', '&amp;').replace('"', '&quot;').replace("'", '&#39;')
            .replace('<', '&lt;').replace('>', '&gt;').replace('\r', ' ').replace('\n', ' '))


def value_text(value):
    """The template's rendering: booleans/null/containers as JSON, strings raw."""
    if isinstance(value, bool): return 'true' if value else 'false'
    if value is None: return 'null'
    if isinstance(value, (dict, list)): return json.dumps(value, ensure_ascii=False)
    return str(value)


def parameter(name, value): return f'<atem:parameter name="{esc_attr(name)}">{value_text(value)}</atem:parameter>'
def arguments(values): return '\n'.join(parameter(k, v) for k, v in values.items())
def obj(properties, required=None, extra=False):
    return dict(type='object', properties=properties, required=list(properties) if required is None else required,
                additionalProperties=extra)


cases = []
def case(name, schema, good, bad, strict=True): cases.append((name, schema, strict, good, bad))

# The template goldens' tool calls, exactly as rendered.
goldens = json.loads(a.goldens.read_text())
for golden in goldens:
    tools = {t['function']['name']: t['function'] for t in golden['context'].get('tools', [])}
    # Only the conversation's assistant tool messages (not the system prompt's examples).
    calls = re.findall(r'<\|start\|>assistant to=[^<]+<\|message\|><atem:function_calls>\n'
                       r'<atem:invoke name="([^"]+)">\n(.*?)</atem:invoke>', golden['expected'], re.S)
    for index, (tool, body) in enumerate(calls):
        schema = tools[tool].get('parameters') or {'type': 'object', 'properties': {}}
        case(f'golden-{golden["name"]}-{index}', schema, [body.rstrip('\n')], [], strict=False)

weather = obj({'city': {'type': 'string'}, 'days': {'type': 'integer', 'minimum': 1, 'maximum': 7},
               'units': {'type': 'string', 'enum': ['c', 'f']}}, required=['city'])
case('typed-values', weather,
     [arguments({'city': '台北'}), arguments({'city': 'Oslo', 'days': 3, 'units': 'f'}),
      arguments({'city': 'a "quoted" <b> city\nline two'})],
     ['', arguments({'days': 3}), arguments({'city': 'Oslo', 'days': 9}), arguments({'city': 'Oslo', 'days': 'three'}),
      arguments({'city': 'Oslo', 'units': 'k'}), arguments({'city': 'Oslo', 'other': 1}),
      parameter('city', 'a') + '\n' + parameter('city', 'b'),
      '<atem:parameter name="city">Oslo', '<atem:parameter name=city>Oslo</atem:parameter>'])
case('json-values', obj({'flag': {'type': 'boolean'}, 'n': {'type': 'number'}, 'nil': {'type': 'null'},
                          'tags': {'type': 'array', 'items': {'type': 'string'}},
                          'mode': obj({'append': {'type': 'boolean'}})}),
     [arguments({'flag': True, 'n': 1.5, 'nil': None, 'tags': ['x', 'y'], 'mode': {'append': True}})],
     [arguments({'flag': 'yes', 'n': 1.5, 'nil': None, 'tags': ['x'], 'mode': {'append': True}}),
      arguments({'flag': True, 'n': 1.5, 'nil': None, 'tags': [1], 'mode': {'append': True}}),
      arguments({'flag': True, 'n': 1.5, 'nil': None, 'tags': ['x'], 'mode': {'append': 'no'}})])
case('string-with-tags', obj({'text': {'type': 'string'}}),
     [arguments({'text': 'x <atem:invoke> y </atem:function_calls> z'})],
     [parameter('text', 'a') + '</atem:parameter>', '<atem:parameter name="text">a</atem:parameter>b</atem:parameter>'])
case('const-and-enum', obj({'k': {'const': 'yes'}, 'n': {'enum': [1, 2]}}),
     [arguments({'k': 'yes', 'n': 2})], [arguments({'k': 'no', 'n': 2}), arguments({'k': 'yes', 'n': 3})])
case('optional-and-additional', obj({'a': {'type': 'integer'}}, required=[], extra={'type': 'string'}),
     ['', arguments({'a': 2}), arguments({'a': 2, 'note': 'hi'}), arguments({'note': 'hi'})],
     [arguments({'a': 'x'}), parameter('a', 1) + '\n' + parameter('a', 2)])
case('escaped-names', obj({'a b"c<d>': {'type': 'integer'}}), [arguments({'a b"c<d>': 1})],
     ['<atem:parameter name="a b"c<d>">1</atem:parameter>'])
case('pattern-string', obj({'id': {'type': 'string', 'pattern': '^[A-Z]{2}[0-9]$'}}),
     [arguments({'id': 'AB2'})], [arguments({'id': 'ab2'}), arguments({'id': ' AB2'})])
case('no-parameters', {'type': 'object', 'properties': {}, 'additionalProperties': False}, [''],
     [arguments({'x': 1})])
case('non-strict-any', {'type': 'object'}, ['', arguments({'anything': 'goes', 'n': 1})], [], strict=False)
# Every raw-string branch (plain, length, pattern) excludes the closing tag and special tokens.
END = '</atem:parameter>'
for label, prop in [('raw', {'type': 'string'}), ('length', {'type': 'string', 'minLength': 1, 'maxLength': 64}),
                    ('pattern', {'type': 'string', 'pattern': '^[^\\n]*$'})]:
    case(f'raw-{label}-excludes-structure', obj({'s': prop}), [arguments({'s': 'a < b | c'})],
         [f'<atem:parameter name="s">x{END}y</atem:parameter>',
          f'<atem:parameter name="s">x<|eom|>y</atem:parameter>',
          f'<atem:parameter name="s">x<|start|>assistant</atem:parameter>',
          f'<atem:parameter name="s">x<|eot|></atem:parameter>'])
# JSON values: strings inside them never carry a raw '<' (the \u003c escape stays).
case('json-strings-without-raw-lt', obj({'cfg': {'type': 'object', 'additionalProperties': {'type': 'string'}},
                                        'tags': {'type': 'array', 'items': {'type': 'string', 'maxLength': 8}}}),
     ['<atem:parameter name="cfg">{"k": "a \\u003c b"}</atem:parameter>\n' + parameter('tags', ['x'])],
     ['<atem:parameter name="cfg">{"k": "a <|eom|> b"}</atem:parameter>\n' + parameter('tags', ['x']),
      parameter('cfg', {'k': 'v'}) + '\n<atem:parameter name="tags">["</atem:parameter>"]</atem:parameter>'])
# Names render through the template's esc pre-pass: a special token in a name is broken apart.
case('escaped-special-names', obj({'a<|b': {'type': 'integer'}}),
     ['<atem:parameter name="a&lt; |b">1</atem:parameter>'], ['<atem:parameter name="a&lt;|b">1</atem:parameter>'])

compiler = P(); stop = (c.c_int32 * 2)(200001, 200008)
invoke('compiler_create', str(a.tokenizer).encode(), VOCAB, stop, 2, c.byref(compiler))
report = dict(scope=__doc__.splitlines()[0], native_lib_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest(),
              tokenizer_sha256=hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(), cases=[], passed=False)
def save(): a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
try:
    for name, schema, strict, good, bad in cases:
        grammar = P()
        spec = dict(type='structural_tag', format=dict(type='muse_tool_schema', json_schema=schema, strict=strict))
        invoke('compile', compiler, 3, json.dumps(spec, ensure_ascii=False).encode(), 1, c.byref(grammar))
        record = dict(name=name, schema=schema, strict=strict, checks=[]); report['cases'].append(record); save()
        try:
            for expected, texts in [(True, good), (False, bad)]:
                for text in texts:
                    matcher = P(); invoke('matcher_create', grammar, c.byref(matcher))
                    ids = tokenizer.encode(text, add_special_tokens=False).ids
                    accepted = True
                    try:
                        for token in ids:
                            mask = (c.c_uint32 * (VOCAB // 32))(); needs = c.c_int()
                            invoke('matcher_fill_bitmask', matcher, mask, len(mask), c.byref(needs))
                            allowed = not needs.value or bool(mask[token // 32] & (1 << (token % 32)))
                            result = c.c_int(); invoke('matcher_accept_token', matcher, token, c.byref(result))
                            assert bool(result.value) == allowed, (name, token, 'mask/accept mismatch')
                            if not result.value: accepted = False; break
                        complete = c.c_int(); invoke('matcher_is_completed', matcher, c.byref(complete))
                        actual = accepted and bool(complete.value)
                        record['checks'].append(dict(text=text, expected=expected, accepted_complete=actual)); save()
                        assert actual == expected, (name, text, expected, actual)
                    finally: invoke('matcher_destroy', matcher, error=False)
        finally: invoke('grammar_destroy', grammar, error=False)
        print('PASS', name, flush=True)
    report['passed'] = True
    save()
    print(f'PASS all {len(cases)} cases')
finally:
    invoke('compiler_destroy', compiler, error=False)
