#include "xgrammar_muse_tools.h"
#include "regex_converter.h"
#include "xgrammar_regular.h"
#include <stdexcept>

namespace xgrammar {
namespace {
const std::string kOpen = "<atem:parameter name=\"";
const std::string kNameEnd = "\">";
const std::string kEnd = "</atem:parameter>";
const std::string kRaw = "muse_raw_string";
const std::string kKey = "muse_parameter_key";
std::string Literal(const std::string& text) { return EBNFScriptCreator::Str(text); }
// The template's esc_attr: names are XML-escaped (and newlines folded), so a
// declared name renders as this attribute text.
std::string Attribute(const std::string& name) {
  std::string out;
  for (char c : name) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&#39;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '\r': case '\n': out += ' '; break;
      default: out += c;
    }
  }
  return out;
}
picojson::value Parse(const std::string& text) {
  picojson::value value;
  auto error = picojson::parse(value, text);
  if (!error.empty()) throw std::invalid_argument(error);
  return value;
}
}  // namespace

MuseToolCallingConverter::MuseToolCallingConverter(RefResolver resolver)
    : JSONSchemaConverter(std::nullopt, std::nullopt, true, 16, std::move(resolver), false) {}

// Level 0 is the arguments object, level 1 a parameter's value, deeper levels
// are JSON inside a value. Rules differ by level, so the cache keys carry it.
std::string MuseToolCallingConverter::ContextKey(const std::string& key) const {
  return std::to_string(std::min(level_, 2)) + ":" + key;
}
void MuseToolCallingConverter::AddCache(const std::string& key, const std::string& value) {
  if (!key.empty()) cache_[ContextKey(key)] = value;
}
std::optional<std::string> MuseToolCallingConverter::GetCache(const std::string& key) const {
  auto found = cache_.find(ContextKey(key));
  return found == cache_.end() ? std::nullopt : std::optional<std::string>(found->second);
}
void MuseToolCallingConverter::AddBasicRules() {
  auto previous = level_;
  level_ = 2;
  JSONSchemaConverter::AddBasicRules();
  level_ = previous;
  // Raw text up to the closing tag (the template does not escape values).
  ebnf_script_creator_.AddRule(kRaw,
      "TagDispatch(loop_after_dispatch=false,excludes=(" + Literal(kEnd) + "))");
  // An undeclared name: attribute text without quotes, markup or newlines.
  ebnf_script_creator_.AddRule(kKey, R"ebnf([^"<>\r\n]+)ebnf");
}
std::string MuseToolCallingConverter::GenerateString(const StringSpec& spec, const std::string& name) {
  if (level_ != 1) return JSONSchemaConverter::GenerateString(spec, name);
  if (spec.format) {
    if (auto pattern = JSONFormatToRegexPattern(*spec.format)) return RegexToEBNF(*pattern, false);
  }
  if (spec.pattern) return RegexToEBNF(*spec.pattern, false);
  if (spec.min_length != 0 || spec.max_length != -1) {
    return "[^]{" + std::to_string(spec.min_length) + "," +
        (spec.max_length == -1 ? "" : std::to_string(spec.max_length)) + "}";
  }
  return kRaw;
}
std::string MuseToolCallingConverter::GenerateArray(const ArraySpec& spec, const std::string& name) {
  ++level_;
  auto value = JSONSchemaConverter::GenerateArray(spec, name);
  --level_;
  return value;
}
std::string MuseToolCallingConverter::GenerateObject(const ObjectSpec& spec, const std::string& name, bool) {
  const bool root = level_ == 0;
  ++level_;
  auto value = JSONSchemaConverter::GenerateObject(spec, name, !root);
  --level_;
  return value;
}
std::string MuseToolCallingConverter::GenerateAny(const AnySpec& spec, const std::string& name) {
  if (level_ == 0) {
    ObjectSpec object;
    object.allow_additional_properties = true;
    object.additional_properties_schema = SchemaSpec::Make(AnySpec{});
    return GenerateObject(object, name);
  }
  // A value of any type: raw text (a string), or a JSON scalar/container. The
  // raw branch already admits every scalar's text; the parser's typing rule
  // (no declared string type: JSON when it parses) recovers the type.
  if (level_ == 1) return kRaw;
  return JSONSchemaConverter::GenerateAny(spec, name);
}
std::string MuseToolCallingConverter::GenerateConst(const ConstSpec& spec, const std::string& name) {
  if (level_ != 1) return JSONSchemaConverter::GenerateConst(spec, name);
  auto value = Parse(spec.json_value);
  // A string constant renders raw; anything else as its JSON text.
  if (value.is<std::string>()) {
    const auto& text = value.get<std::string>();
    if (text.find(kEnd) != std::string::npos)
      throw std::invalid_argument("string constant contains the closing parameter tag");
    return Literal(text);
  }
  return Literal(value.serialize(false));
}
std::string MuseToolCallingConverter::GenerateEnum(const EnumSpec& spec, const std::string& name) {
  if (level_ != 1) return JSONSchemaConverter::GenerateEnum(spec, name);
  std::vector<std::string> alternatives;
  for (const auto& value : spec.json_values) alternatives.push_back(GenerateConst({value}, name));
  return EBNFScriptCreator::Or(alternatives);
}
std::string MuseToolCallingConverter::FormatProperty(const std::string& key, const std::string& value,
    const std::string& name, int64_t index) {
  if (level_ != 1) return JSONSchemaConverter::FormatProperty(key, value, name, index);
  return Literal(kOpen + Attribute(key) + kNameEnd) + " (" + value + ") " + Literal(kEnd);
}
std::string MuseToolCallingConverter::FormatOtherProperty(const std::string& key, const std::string& value,
    const std::string& name, const std::string& suffix) {
  if (level_ != 1) return JSONSchemaConverter::FormatOtherProperty(key, value, name, suffix);
  return Literal(kOpen) + " " + key + " " + Literal(kNameEnd) + " (" + value + ") " + Literal(kEnd);
}
std::string MuseToolCallingConverter::GetKeyPattern() const {
  return level_ == 1 ? kKey : JSONSchemaConverter::GetKeyPattern();
}
std::string MuseToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& name) {
  // An undeclared name may not reuse a declared one (with weaker constraints).
  if (level_ != 1) return JSONSchemaConverter::GetKeyPatternExcluding(properties, name);
  if (properties.empty()) return kKey;
  auto allowed = V41RegularFSM(Grammar::FromEBNF("root ::= [^\"<>\\r\\n]+"));
  std::vector<std::string> names;
  for (const auto& property : properties) names.push_back(Literal(Attribute(property.name)));
  auto declared = V41RegularFSM(Grammar::FromEBNF("root ::= " + EBNFScriptCreator::Or(names)));
  return V41EmitFSM(V41Subtract(allowed, declared), ebnf_script_creator_);
}
std::string MuseToolCallingConverter::NextSeparator(bool end) {
  // Parameters are newline-separated in the template; bounded whitespace as for V4.1.
  return level_ == 1 ? GetWhitespacePattern() : JSONSchemaConverter::NextSeparator(end);
}
std::string MuseToolSchemaToEBNF(const picojson::value& schema, bool strict) {
  return JSONSchemaToEBNF(schema, true, std::nullopt, std::nullopt, strict, 16,
      static_cast<JSONFormat>(6), false);
}
}  // namespace xgrammar
