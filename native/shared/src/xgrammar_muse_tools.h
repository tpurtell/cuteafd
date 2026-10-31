#pragma once
#include "json_schema_converter.h"

namespace xgrammar {
// Project extension of the pinned schema IR for Muse Spark's atem tool calls:
//
//   <atem:parameter name="KEY">VALUE</atem:parameter>
//
// one per argument, as the checkpoint's chat template renders them. A value
// whose schema admits a string is the raw text up to the closing tag (the
// template prints strings unquoted); every other value is JSON per its schema
// (the template prints booleans, null, numbers, objects and arrays as JSON).
// The build adds a factory case in the generated converter copy, as for V4.1.
class MuseToolCallingConverter : public JSONSchemaConverter {
 public:
  explicit MuseToolCallingConverter(RefResolver resolver);
 protected:
  std::string GenerateString(const StringSpec&, const std::string&) override;
  std::string GenerateObject(const ObjectSpec&, const std::string&, bool = true) override;
  std::string GenerateArray(const ArraySpec&, const std::string&) override;
  std::string GenerateAny(const AnySpec&, const std::string&) override;
  std::string GenerateConst(const ConstSpec&, const std::string&) override;
  std::string GenerateEnum(const EnumSpec&, const std::string&) override;
  std::string FormatProperty(const std::string&, const std::string&, const std::string&, int64_t) override;
  std::string FormatOtherProperty(const std::string&, const std::string&, const std::string&, const std::string&) override;
  std::string GetKeyPattern() const override;
  std::string GetKeyPatternExcluding(const std::vector<ObjectSpec::Property>&, const std::string&) override;
  std::string NextSeparator(bool = false) override;
  void AddBasicRules() override;
  void AddCache(const std::string&, const std::string&) override;
  std::optional<std::string> GetCache(const std::string&) const override;
 private:
  int level_ = 0;
  std::unordered_map<std::string, std::string> cache_;
  std::string ContextKey(const std::string&) const;
  std::string Constrained(const StringSpec&, bool raw);
};
std::string MuseToolSchemaToEBNF(const picojson::value& schema, bool strict);
}  // namespace xgrammar
