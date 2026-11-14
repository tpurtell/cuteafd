#include "xgrammar_regular.h"
#include "xgrammar_minimax_tools.h"
#include <picojson.h>
#include <xgrammar/xgrammar.h>
#include <iostream>
#include <stdexcept>

using namespace xgrammar;

static bool Accept(const CompiledGrammar& grammar, const std::string& text) {
  GrammarMatcher matcher(grammar);
  return matcher.AcceptString(text) && matcher.IsCompleted();
}

int main() {
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);
  std::vector<std::string> texts{"", "北京", "台北", "北台", "😀", "\x7f",
      "\xc2\x80", "\xdf\xbf", "\xe0\xa0\x80", "\xed\x9f\xbf",
      "\xee\x80\x80", "\xef\xbf\xbf", "\xf0\x90\x80\x80",
      "\xf4\x8f\xbf\xbf", std::string(1, '\0')};
  std::vector<std::string> frontier{""};
  for (int length = 0; length < 4; ++length) {
    std::vector<std::string> next;
    for (const auto& prefix : frontier)
      for (const auto& letter : {"a", "b", "x", "台", "😀"}) {
        next.push_back(prefix + letter);
        texts.push_back(next.back());
      }
    frontier = std::move(next);
  }
  const std::vector<std::string> patterns{
      "^(ab)*x(ab)*$", "^[a-z]{2,5}$", "^(a|bc)+$",
      "^([^台]|ab){1,4}$", "^[^]*$", "^[台-😀]*$",
      "^(a?b){0,3}$", "^.{0,0}$", "^[^a]*$"};
  size_t checked = 0;
  auto emitted = [&](const FSMWithStartEnd& fsm) {
    EBNFScriptCreator script;
    auto root = script.AllocateRuleName("root");
    auto body = V41EmitFSM(fsm, script);
    script.AddRuleWithAllocatedName(root, body);
    return compiler.CompileGrammar(Grammar::FromEBNF(script.GetScript()));
  };
  for (const auto& lhs_pattern : patterns) {
    auto lhs_grammar = Grammar::FromRegex(lhs_pattern);
    auto lhs = compiler.CompileGrammar(lhs_grammar);
    auto lhs_fsm = V41RegularFSM(lhs_grammar);
    auto roundtrip = emitted(lhs_fsm);
    for (const auto& text : texts) {
      if (Accept(lhs, text) != Accept(roundtrip, text))
        throw std::runtime_error("regular roundtrip mismatch: " + lhs_pattern);
      ++checked;
    }
    for (const auto& rhs_pattern : patterns) {
      auto rhs_grammar = Grammar::FromRegex(rhs_pattern);
      auto rhs = compiler.CompileGrammar(rhs_grammar);
      auto rhs_fsm = V41RegularFSM(rhs_grammar);
      auto intersection = emitted(V41Intersect(lhs_fsm, rhs_fsm));
      auto subtraction = emitted(V41Subtract(lhs_fsm, rhs_fsm));
      if (!V41Empty(V41Subtract(lhs_fsm, lhs_fsm)))
        throw std::runtime_error("self subtraction is not empty");
      for (const auto& text : texts) {
        bool a = Accept(lhs, text), b = Accept(rhs, text);
        if (Accept(intersection, text) != (a && b) ||
            Accept(subtraction, text) != (a && !b))
          throw std::runtime_error("regular product mismatch: " + lhs_pattern + " / " + rhs_pattern);
        checked += 2;
      }
    }
  }
  // Shared rule references must not share their callers' continuations.
  auto shared = Grammar::FromEBNF("root ::= part \"x\" part\npart ::= \"a\" | \"b\"\n");
  auto shared_result = emitted(V41RegularFSM(shared));
  if (!Accept(shared_result, "axb") || Accept(shared_result, "axaxb") || Accept(shared_result, "a"))
    throw std::runtime_error("shared continuation leaked");
  bool rejected = false;
  try {
    V41RegularFSM(Grammar::FromEBNF("root ::= \"\" | \"a\" root \"b\"\n"));
  } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("non-regular recursion accepted");
  // MiniMax M3 recursively serializes XML, not upstream kMiniMaxXML.
  picojson::value schema;
  auto error = picojson::parse(schema, R"({"type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer","minimum":1,"maximum":7},"nested":{"type":"object","properties":{"flag":{"type":"boolean"}},"required":["flag"],"additionalProperties":false},"rows":{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}},"required":["s","n","nested","rows"],"additionalProperties":false})");
  if (!error.empty()) throw std::runtime_error(error);
  auto minimax = compiler.CompileGrammar(Grammar::FromEBNF(MiniMaxToolSchemaToEBNF(schema, true)));
  const std::string ns = "]<]minimax[>[";
  const std::string valid = ns + "<s>001" + ns + "</s>" + ns + "<n>3" + ns + "</n>" + ns + "<nested>" + ns + "<flag>true" + ns + "</flag>" + ns + "</nested>" + ns + "<rows>" + ns + "<item>1" + ns + "</item>" + ns + "<item>2" + ns + "</item>" + ns + "</rows>";
  if (!Accept(minimax, valid)) throw std::runtime_error("MiniMax template XML rejected");
  auto bad = valid;
  bad.replace(bad.find("<n>3"), 4, "<n>9");
  if (Accept(minimax, bad) || Accept(minimax, valid + valid)) throw std::runtime_error("invalid MiniMax XML accepted");
  auto refuse = [&](const std::string& text) {
    picojson::value input;
    if (!picojson::parse(input, text).empty()) throw std::runtime_error("invalid test schema");
    bool refused = false;
    try { MiniMaxToolSchemaToEBNF(input, true); } catch (const std::invalid_argument&) { refused = true; }
    if (!refused) throw std::runtime_error("unrepresentable MiniMax schema accepted");
  };
  refuse(R"({"type":"object","properties":{"nil":{"type":"null"}},"required":["nil"]})");
  refuse(R"({"type":"object","required":["unknown"],"additionalProperties":true})");
  refuse(R"({"type":"object","minProperties":1,"additionalProperties":true})");
  std::cout << "regular grammar checks passed: " << checked << " language comparisons\n";
}
