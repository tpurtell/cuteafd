#pragma once
#include "json_schema_converter.h"

namespace xgrammar {
// M3's namespaced XML is recursive at every depth, unlike upstream
// kMiniMaxXML and Muse's JSON-valued parameter bodies. Only declared property
// names are generated: arbitrary matching open/close names are not context-free.
// additionalProperties admits a valid subset (empty for no declared names),
// whereas Muse/V4.1 carry unknown names in single opening attributes and allow
// them. Required undeclared/null-only fields and unsatisfiable minima refuse.
// Strings win scalar unions, matching the API parser; container unions refuse.
class MiniMaxToolCallingConverter : public JSONSchemaConverter {
 public:
  explicit MiniMaxToolCallingConverter(RefResolver resolver);
 protected:
  void AddBasicRules() override;
  std::string GenerateString(const StringSpec&, const std::string&) override;
  std::string GenerateNull(const NullSpec&, const std::string&) override;
  std::string GenerateObject(const ObjectSpec&, const std::string&, bool = true) override;
  std::string GenerateArray(const ArraySpec&, const std::string&) override;
  std::string GenerateAny(const AnySpec&, const std::string&) override;
  std::string GenerateConst(const ConstSpec&, const std::string&) override;
  std::string GenerateEnum(const EnumSpec&, const std::string&) override;
  std::string GenerateAnyOf(const AnyOfSpec&, const std::string&) override;
  std::string GenerateTypeArray(const TypeArraySpec&, const std::string&) override;
  std::string GenerateAllOf(const AllOfSpec&, const std::string&) override;
  std::string FormatProperty(const std::string&, const std::string&, const std::string&, int64_t) override;
  std::string FormatOtherProperty(const std::string&, const std::string&, const std::string&, const std::string&) override;
  std::string NextSeparator(bool = false) override;
  void AddCache(const std::string&, const std::string&) override;
  std::optional<std::string> GetCache(const std::string&) const override;
 private:
  size_t level_ = 0;
  std::unordered_map<std::string, std::string> cache_;
  std::string ContextKey(const std::string&) const;
  std::string Alternatives(const std::vector<SchemaSpecPtr>&, const std::string&);
};
std::string MiniMaxToolSchemaToEBNF(const picojson::value& schema, bool strict);
}  // namespace xgrammar
