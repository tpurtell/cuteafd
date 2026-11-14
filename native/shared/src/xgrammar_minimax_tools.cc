#include "xgrammar_minimax_tools.h"
#include "regex_converter.h"
#include "xgrammar_regular.h"
#include <algorithm>
#include <stdexcept>

namespace xgrammar {
namespace {
const std::string kNS = "]<]minimax[>[";
const std::string kRaw = "minimax_raw";
std::string Literal(const std::string& text) { return EBNFScriptCreator::Str(text); }
void Name(const std::string& name) {
  if (name.empty() || name.find_first_of("<>/\" \r\n\t") != std::string::npos || name.find(kNS) != std::string::npos)
    throw std::invalid_argument("MiniMax XML cannot represent property name: " + name);
}
std::string Wrap(const std::string& name, const std::string& body) {
  Name(name);
  return Literal(kNS + "<" + name + ">") + " (" + body + ") " + Literal(kNS + "</" + name + ">");
}
bool String(const SchemaSpecPtr& spec) {
  if (std::holds_alternative<StringSpec>(spec->spec) || std::holds_alternative<AnySpec>(spec->spec)) return true;
  if (auto value = std::get_if<ConstSpec>(&spec->spec)) {
    picojson::value parsed;
    if (!picojson::parse(parsed, value->json_value).empty()) throw std::invalid_argument("invalid MiniMax const");
    return parsed.is<std::string>();
  }
  if (auto values = std::get_if<EnumSpec>(&spec->spec)) {
    for (const auto& value : values->json_values) {
      picojson::value parsed;
      if (!picojson::parse(parsed, value).empty()) throw std::invalid_argument("invalid MiniMax enum");
      if (parsed.is<std::string>()) return true;
    }
  }
  if (auto values = std::get_if<AnyOfSpec>(&spec->spec))
    return std::any_of(values->options.begin(), values->options.end(), String);
  if (auto values = std::get_if<TypeArraySpec>(&spec->spec))
    return std::any_of(values->type_schemas.begin(), values->type_schemas.end(), String);
  return false;
}
bool NullOnly(const SchemaSpecPtr& spec) {
  if (std::holds_alternative<NullSpec>(spec->spec)) return true;
  if (auto value = std::get_if<ConstSpec>(&spec->spec)) return value->json_value == "null";
  if (auto values = std::get_if<EnumSpec>(&spec->spec))
    return std::all_of(values->json_values.begin(), values->json_values.end(), [](const std::string& v) { return v == "null"; });
  return false;
}
// Validate representability independently of XGrammar's strict flag. Arbitrary
// XML opening/closing names cannot be paired by a CFG: MiniMax generates only
// declared property names, recursively. Unlike Muse's attribute syntax (which
// permits unknown names), additionalProperties is a valid-subset restriction:
// unknown names are never emitted, and an unconstrained object emits empty.
// Required undeclared/null-only fields and impossible minProperties refuse.
void Validate(const picojson::value& value, const picojson::value& root, size_t depth = 0) {
  if (depth > 64) throw std::invalid_argument("MiniMax schema nesting or reference recursion exceeds 64");
  if (!value.is<picojson::object>()) return;
  const auto& object = value.get<picojson::object>();
  if (auto ref = object.find("$ref"); ref != object.end()) {
    if (!ref->second.is<std::string>() || ref->second.get<std::string>().empty() || ref->second.get<std::string>()[0] != '#')
      throw std::invalid_argument("MiniMax tool schema supports local references only");
    picojson::value target = root;
    auto pointer = ref->second.get<std::string>().substr(1);
    while (!pointer.empty()) {
      if (pointer[0] != '/') throw std::invalid_argument("invalid MiniMax schema reference");
      auto slash = pointer.find('/', 1);
      auto key = pointer.substr(1, slash == std::string::npos ? slash : slash - 1);
      for (const auto& replacement : {std::make_pair(std::string("~1"), std::string("/")), std::make_pair(std::string("~0"), std::string("~"))}) {
        size_t at = 0;
        while ((at = key.find(replacement.first, at)) != std::string::npos) { key.replace(at, 2, replacement.second); ++at; }
      }
      if (!target.is<picojson::object>() || !target.get<picojson::object>().count(key))
        throw std::invalid_argument("unresolved MiniMax schema reference");
      auto next = target.get<picojson::object>().at(key);
      target = std::move(next);
      pointer = slash == std::string::npos ? "" : pointer.substr(slash);
    }
    Validate(target, root, depth + 1);
  }
  for (const auto& key : {"patternProperties", "propertyNames", "dependentRequired", "dependentSchemas", "if", "then", "else", "not", "contains", "minContains", "maxContains", "uniqueItems", "oneOf", "allOf", "multipleOf", "unevaluatedProperties"})
    if (object.count(key)) throw std::invalid_argument(std::string("MiniMax constrained XML does not support schema keyword: ") + key);
  auto properties = object.find("properties");
  const picojson::object empty;
  const auto& props = properties != object.end() && properties->second.is<picojson::object>() ? properties->second.get<picojson::object>() : empty;
  if (auto required = object.find("required"); required != object.end() && required->second.is<picojson::array>()) {
    for (const auto& name : required->second.get<picojson::array>()) {
      if (!name.is<std::string>() || !props.count(name.get<std::string>()))
        throw std::invalid_argument("MiniMax constrained XML requires every required property to be declared");
      const auto& property = props.at(name.get<std::string>());
      if (property.is<picojson::object>()) {
        const auto& p = property.get<picojson::object>();
        auto type = p.find("type"), constant = p.find("const");
        if ((type != p.end() && type->second.is<std::string>() && type->second.get<std::string>() == "null") ||
            (constant != p.end() && constant->second.is<picojson::null>()))
          throw std::invalid_argument("MiniMax constrained XML cannot represent a required-null field");
      }
    }
  }
  if (auto minimum = object.find("minProperties"); minimum != object.end() && minimum->second.is<double>() && minimum->second.get<double>() > props.size())
    throw std::invalid_argument("MiniMax constrained XML minProperties requires undeclared properties");
  for (const auto& property : props) { Name(property.first); Validate(property.second, root, depth + 1); }
  for (const auto& key : {"items", "additionalProperties"}) {
    auto nested = object.find(key);
    if (nested != object.end()) Validate(nested->second, root, depth + 1);
  }
  for (const auto& key : {"anyOf", "oneOf"}) {
    auto union_ = object.find(key);
    if (union_ != object.end() && union_->second.is<picojson::array>()) {
      for (const auto& branch : union_->second.get<picojson::array>()) {
        if (!branch.is<picojson::object>()) continue;
        const auto& b = branch.get<picojson::object>();
        auto type = b.find("type");
        if (b.count("$ref") || (type != b.end() && type->second.is<std::string>() &&
            (type->second.get<std::string>() == "object" || type->second.get<std::string>() == "array")))
          throw std::invalid_argument("MiniMax constrained XML supports scalar union branches without references only");
      }
    }
  }
  if (auto types = object.find("type"); types != object.end() && types->second.is<picojson::array>())
    for (const auto& type : types->second.get<picojson::array>())
      if (type.is<std::string>() && (type.get<std::string>() == "object" || type.get<std::string>() == "array"))
        throw std::invalid_argument("MiniMax constrained XML container type unions are not representable unambiguously");
  for (const auto& key : {"anyOf", "oneOf", "allOf", "prefixItems"}) {
    auto nested = object.find(key);
    if (nested != object.end() && nested->second.is<picojson::array>())
      for (const auto& branch : nested->second.get<picojson::array>()) Validate(branch, root, depth + 1);
  }
}
}  // namespace

MiniMaxToolCallingConverter::MiniMaxToolCallingConverter(RefResolver resolver)
    : JSONSchemaConverter(std::nullopt, std::nullopt, false, 16, std::move(resolver), false) {}
std::string MiniMaxToolCallingConverter::ContextKey(const std::string& key) const {
  return (level_ == 0 ? "root:" : "value:") + key;
}
void MiniMaxToolCallingConverter::AddCache(const std::string& key, const std::string& value) {
  if (!key.empty()) cache_[ContextKey(key)] = value;
}
std::optional<std::string> MiniMaxToolCallingConverter::GetCache(const std::string& key) const {
  auto found = cache_.find(ContextKey(key));
  return found == cache_.end() ? std::nullopt : std::optional<std::string>(found->second);
}
void MiniMaxToolCallingConverter::AddBasicRules() {
  // Base initialization hardcodes JSON's quoted string; install native rules
  // explicitly so every recursive depth remains XML, never JSON in a value.
  ebnf_script_creator_.AddRule(kRaw, "TagDispatch(loop_after_dispatch=false,excludes=(" + Literal(kNS) + ", " + Literal("[e~[") + ", " + Literal("]~b]") + ", " + Literal("]~!b[") + ", " + Literal("]!d~[") + ", " + Literal("]!p~[") + ", " + Literal("<mm:think>") + ", " + Literal("</mm:think>") + "))");
  for (const auto& item : std::vector<std::pair<std::string, std::string>>{
      {kBasicInteger, "(\"-\")? (\"0\" | [1-9] [0-9]*)"},
      {kBasicNumber, "(\"-\")? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?"},
      {kBasicBoolean, "\"true\" | \"false\""}, {kBasicNull, "\"\""},
      {kBasicString, kRaw}, {kBasicAny, kRaw}}) ebnf_script_creator_.AddRule(item.first, item.second);
  for (const auto& item : std::vector<std::pair<std::string, std::string>>{
      {"integer", kBasicInteger}, {"number", kBasicNumber}, {"boolean", kBasicBoolean}, {"string", kBasicString}})
    cache_["value:{\"type\":\"" + item.first + "\"}"] = item.second;
  cache_["value:{}"] = kBasicAny;
}
std::string MiniMaxToolCallingConverter::GenerateString(const StringSpec& spec, const std::string&) {
  if (!spec.format && !spec.pattern && spec.min_length == 0 && spec.max_length == -1) return kRaw;
  std::string body;
  if (spec.format) if (auto pattern = JSONFormatToRegexPattern(*spec.format)) body = RegexToEBNF(*pattern, false);
  if (spec.format && spec.pattern) throw std::invalid_argument("MiniMax constrained XML does not combine format and pattern");
  if (body.empty() && spec.pattern) body = RegexToEBNF(*spec.pattern, false);
  if (body.empty()) body = "[^]{" + std::to_string(spec.min_length) + "," + (spec.max_length == -1 ? "" : std::to_string(spec.max_length)) + "}";
  auto allowed = V41RegularFSM(Grammar::FromEBNF("root ::= " + body));
  if ((spec.pattern || spec.format) && (spec.min_length != 0 || spec.max_length != -1)) {
    auto length = "[^]{" + std::to_string(spec.min_length) + "," + (spec.max_length == -1 ? "" : std::to_string(spec.max_length)) + "}";
    allowed = V41Intersect(allowed, V41RegularFSM(Grammar::FromEBNF("root ::= " + length)));
  }
  // Exclude the namespace and all output control markers, even under pattern.
  std::string forbidden;
  for (const auto& token : {kNS, std::string("[e~["), std::string("]~b]"), std::string("]~!b["), std::string("]!d~["), std::string("]!p~["), std::string("<mm:think>"), std::string("</mm:think>")}) {
    if (!forbidden.empty()) forbidden += "|";
    for (char ch : token) {
      if (std::string("\\^$.|?*+()[]{}/").find(ch) != std::string::npos) forbidden += '\\';
      forbidden += ch;
    }
  }
  auto safe = V41Subtract(allowed, V41RegularFSM(Grammar::FromRegex("[^]*(" + forbidden + ")[^]*")));
  if (V41Empty(safe)) throw std::invalid_argument("MiniMax string schema contains only unrepresentable control markers");
  return V41EmitFSM(safe, ebnf_script_creator_);
}
std::string MiniMaxToolCallingConverter::GenerateNull(const NullSpec&, const std::string&) {
  throw std::invalid_argument("MiniMax constrained XML cannot represent null values; object fields are omitted by the checkpoint");
}
std::string MiniMaxToolCallingConverter::GenerateObject(const ObjectSpec& spec, const std::string& name, bool) {
  ObjectSpec declared = spec;
  declared.allow_additional_properties = false;
  declared.additional_properties_schema = nullptr;
  declared.allow_unevaluated_properties = false;
  declared.unevaluated_properties_schema = nullptr;
  declared.pattern_properties.clear();
  declared.property_names = nullptr;
  declared.properties.erase(std::remove_if(declared.properties.begin(), declared.properties.end(), [&](const ObjectSpec::Property& property) {
    if (!NullOnly(property.schema)) return false;
    if (declared.required.count(property.name)) throw std::invalid_argument("MiniMax constrained XML cannot represent a required-null field");
    return true;
  }), declared.properties.end());
  if (declared.min_properties > static_cast<int>(declared.properties.size()))
    throw std::invalid_argument("MiniMax minProperties cannot be satisfied by representable declared properties");
  if (declared.properties.empty()) return Literal("");
  ++level_;
  auto result = JSONSchemaConverter::GenerateObject(declared, name, false);
  --level_;
  // The base converter admits whitespace for an empty object. In recursive
  // XML that whitespace would instead be a scalar and would not round-trip.
  const auto whitespace = GetWhitespacePattern();
  size_t at = 0;
  while ((at = result.find(whitespace, at)) != std::string::npos) {
    result.replace(at, whitespace.size(), Literal(""));
    at += 2;
  }
  return result;
}
std::string MiniMaxToolCallingConverter::GenerateArray(const ArraySpec& spec, const std::string& name) {
  ++level_;
  std::vector<std::string> prefix;
  for (size_t i = 0; i < spec.prefix_items.size(); ++i)
    prefix.push_back(Wrap("item", CreateRule(spec.prefix_items[i], name + "_item_" + std::to_string(i))));
  auto body = EBNFScriptCreator::Concat(prefix);
  const int64_t start = static_cast<int64_t>(prefix.size());
  if (spec.max_items != -1 && spec.max_items < start) throw std::invalid_argument("MiniMax array prefix exceeds maxItems");
  if (spec.allow_additional_items && spec.additional_items) {
    auto item = Wrap("item", CreateRule(spec.additional_items, name + "_item"));
    body += " " + EBNFScriptCreator::Repeat("(" + item + ")", static_cast<int>(std::max<int64_t>(0, spec.min_items - start)),
        spec.max_items == -1 ? -1 : static_cast<int>(spec.max_items - start));
  } else if (spec.min_items > start) throw std::invalid_argument("MiniMax array minItems cannot be satisfied");
  --level_;
  return body.empty() ? Literal("") : body;
}
std::string MiniMaxToolCallingConverter::GenerateAny(const AnySpec&, const std::string&) {
  return level_ == 0 ? Literal("") : kRaw;
}
std::string MiniMaxToolCallingConverter::GenerateConst(const ConstSpec& spec, const std::string& name) {
  picojson::value value;
  if (!picojson::parse(value, spec.json_value).empty()) throw std::invalid_argument("invalid MiniMax constant");
  if (value.is<picojson::null>()) return GenerateNull({}, name);
  if (value.is<std::string>()) {
    const auto& text = value.get<std::string>();
    for (const auto& token : {kNS, std::string("[e~["), std::string("]~b]"), std::string("]~!b["), std::string("]!d~["), std::string("]!p~["), std::string("<mm:think>"), std::string("</mm:think>")})
      if (text.find(token) != std::string::npos) throw std::invalid_argument("MiniMax string constant contains a control marker");
    return Literal(text);
  }
  if (value.is<picojson::array>() || value.is<picojson::object>())
    throw std::invalid_argument("MiniMax constrained XML container const/enum is not supported");
  return Literal(spec.json_value);
}
std::string MiniMaxToolCallingConverter::GenerateEnum(const EnumSpec& spec, const std::string& name) {
  const bool strings = String(SchemaSpec::Make(spec));
  std::vector<std::string> alternatives;
  for (const auto& text : spec.json_values) {
    picojson::value value;
    if (!picojson::parse(value, text).empty()) throw std::invalid_argument("invalid MiniMax enum");
    if (value.is<picojson::null>() || (strings && !value.is<std::string>())) continue;
    alternatives.push_back(GenerateConst({text}, name));
  }
  if (alternatives.empty()) throw std::invalid_argument("MiniMax enum has no representable values");
  return EBNFScriptCreator::Or(alternatives);
}
std::string MiniMaxToolCallingConverter::Alternatives(const std::vector<SchemaSpecPtr>& options, const std::string& name) {
  if (level_ == 0) throw std::invalid_argument("MiniMax tool arguments must have a concrete object schema");
  const bool strings = std::any_of(options.begin(), options.end(), String);
  std::vector<std::string> alternatives;
  for (const auto& option : options) {
    if (NullOnly(option) || (strings && !String(option))) continue;
    alternatives.push_back(CreateRule(option, name));
  }
  if (alternatives.empty()) throw std::invalid_argument("MiniMax union has no representable values");
  return EBNFScriptCreator::Or(alternatives);
}
std::string MiniMaxToolCallingConverter::GenerateAnyOf(const AnyOfSpec& spec, const std::string& name) { return Alternatives(spec.options, name); }
std::string MiniMaxToolCallingConverter::GenerateTypeArray(const TypeArraySpec& spec, const std::string& name) { return Alternatives(spec.type_schemas, name); }
std::string MiniMaxToolCallingConverter::GenerateAllOf(const AllOfSpec& spec, const std::string& name) {
  if (spec.schemas.size() == 1) return CreateRule(spec.schemas.front(), name);
  throw std::invalid_argument("MiniMax constrained XML requires allOf to normalize to a single schema");
}
std::string MiniMaxToolCallingConverter::FormatProperty(const std::string& key, const std::string& value, const std::string&, int64_t) { return Wrap(key, value); }
std::string MiniMaxToolCallingConverter::FormatOtherProperty(const std::string&, const std::string&, const std::string&, const std::string&) {
  throw std::invalid_argument("MiniMax XML generates declared property names only");
}
std::string MiniMaxToolCallingConverter::NextSeparator(bool) { return Literal(""); }
std::string MiniMaxToolSchemaToEBNF(const picojson::value& schema, bool strict) {
  if (!schema.is<picojson::object>()) throw std::invalid_argument("MiniMax tool arguments require an object schema");
  const auto& object = schema.get<picojson::object>();
  if (auto type = object.find("type"); type != object.end() && (!type->second.is<std::string>() || type->second.get<std::string>() != "object"))
    throw std::invalid_argument("MiniMax tool arguments require an object schema");
  Validate(schema, schema);
  return JSONSchemaToEBNF(schema, true, std::nullopt, std::nullopt, strict, 16, static_cast<JSONFormat>(7), false);
}
}  // namespace xgrammar
