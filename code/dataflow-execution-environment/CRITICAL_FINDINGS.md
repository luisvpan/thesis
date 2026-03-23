# CRITICAL FINDINGS - 2026-03-09

## Overview

Comprehensive analysis of the codebase reveals several CRITICAL issues that block key functionality. The most critical is the set literal parsing issue which prevents all set operations from being tested end-to-end.

---

## CRITICAL BLOCKERS (P0)

### 1. Set Literal Parsing Issue - COMPLETE BLOCKER

**Status:** ACTIVE CRITICAL ISSUE
**Discovered:** 2026-03-09

**Problem:**
The grammar spec (GRAMMAR_SPEC.md line 168) defines set literal syntax using `{}`:
```
set_literal ::= "{" ( value ( "," value )* )? "}"
```

**BUT the implementation:**
- Parser uses `arrayLiteral` rule which expects `[]` brackets (LBracket/RBracket)
- NO `setLiteral` rule exists
- Cannot parse `{1, 2, 3}` syntax for sets
- AST builder only handles `arrayLiteral`, not set literals

**Evidence:**
1. Parser (dataflow-parser.ts lines 170-177): `arrayLiteral` uses LBracket/RBracket
2. Grammar spec (GRAMMAR_SPEC.md line 168): set_literal uses `{}` syntax
3. AST builder (ast-builder.ts lines 76-78): Only handles `arrayLiteral`

**Impact:**
- **COMPLETE BLOCKER:** Set literals cannot be parsed
- All programs using set literals fail to compile
- Set operations (UNION, INTERSECTION, DIFFERENCE, COMPLEMENT) cannot be tested
- Integration tests for sets (Test 4.1, Test 4.2) are stubbed
- Curriculum types with sets cannot be tested end-to-end

**Root Cause:**
Grammar spec defines `{}` for sets, but parser uses `[]` from arrayLiteral

**Required Fix:**
1. Add `setLiteral` rule to parser using LBrace/RBrace tokens
2. Update AST builder to handle `setLiteral` CST
3. Add set literal to `value` rule alternatives
4. Update integration tests to use `{}` syntax

**Estimated Time:** 1.5 hours

---

## HIGH PRIORITY ISSUES (P1)

### 2. COMPARE Operation Type Error

**Status:** ACTIVE ISSUE
**Discovered:** 2026-03-09

**Problem:**
COMPARE operation has wrong output type in registry (registry.ts line 77):
- Registry shows: `outputType: "integer"`
- Spec requires: `outputType: "boolean"` (LANGUAGE_SPEC.md lines 53, 70, 89, 107, 137)

**Impact:**
- Type checking for COMPARE is incorrect
- Programs using COMPARE may fail type validation incorrectly
- Spec violation

**Required Fix:**
Update registry.ts line 77: change `outputType: "integer"` to `outputType: "boolean"`

**Estimated Time:** 5 minutes

---

### 3. Missing Stream Literal Tokens

**Status:** ACTIVE ISSUE
**Discovered:** 2026-03-09

**Problem:**
Grammar spec defines stream source keywords (GRAMMAR_SPEC.md lines 172-174):
- SENSOR
- GENERATOR
- EXTERNAL

But these tokens are missing from the lexer.

**Impact:**
- Cannot parse stream literals
- Temporal operations cannot be tested end-to-end
- Stream functionality cannot be used

**Required Fix:**
Add missing tokens to lexer (tokens.ts)

**Estimated Time:** 15 minutes

---

### 4. Missing Stream Literal Parser Rule

**Status:** ACTIVE ISSUE
**Discovered:** 2026-03-09

**Problem:**
Grammar spec defines stream_literal syntax (GRAMMAR_SPEC.md line 170):
```
stream_literal ::= "stream" "<" type ">" "(" stream_source ")"
stream_source ::= "sensor" "(" text_literal ")" | "generator" "(" identifier ")" | "external" "(" text_literal ")"
```

But no `streamLiteral` rule exists in parser.

**Impact:**
- Cannot parse stream literals
- Temporal operations cannot be tested
- Stream functionality blocked

**Required Fix:**
Add `streamLiteral` rule to parser

**Estimated Time:** 30 minutes

---

### 5. Missing Fraction Literal Support

**Status:** ACTIVE ISSUE
**Discovered:** 2026-03-09

**Problem:**
Grammar spec defines fraction literal (GRAMMAR_SPEC.md line 108):
```
fraction_literal ::= integer_literal "/" natural_literal
```

But:
- No Fraction literal token in lexer
- No fractionLiteral rule in parser
- No fractionLiteral method in AST builder

**Impact:**
- Cannot parse fraction literals
- Fraction type defined but unusable

**Required Fix:**
1. Add fraction literal parsing support
2. Add AST builder method
3. Add tests

**Estimated Time:** 1 hour

---

## MEDIUM PRIORITY ISSUES (P2)

### 6. Stubbed Integration Tests

**Status:** ACTIVE ISSUE
**Discovered:** 2026-03-09

**Problem:**
Many integration tests are stubbed with TODO comments:
- Set operations tests (union.test.ts, filter-sort.test.ts)
- Temporal tests (fby.test.ts, streams.test.ts)
- Some curriculum and type tests

**Impact:**
- Cannot verify end-to-end functionality
- Integration test coverage is low (30%)

**Required Fix:**
Implement all stubbed tests after parsing issues resolved

**Estimated Time:** 4 hours

---

## SUMMARY TABLE

| Issue | Priority | Status | Impact Area | Estimated Time |
|-------|----------|--------|-------------|----------------|
| Set literal parsing | P0 CRITICAL | Blocks all set functionality | 1.5 hours |
| COMPARE type error | P1 HIGH | Type checking | 5 minutes |
| Stream source tokens | P1 HIGH | Stream parsing | 15 minutes |
| Stream literal parser | P1 HIGH | Stream parsing | 30 minutes |
| Fraction literal support | P1 HIGH | Fraction types | 1 hour |
| Stubbed integration tests | P2 MEDIUM | Test coverage | 4 hours |

---

## IMMEDIATE NEXT STEPS (Priority Order)

1. **Fix COMPARE operation type** (5 minutes) - Quick fix
2. **Add setLiteral rule to parser** (45 minutes) - CRITICAL
3. **Add setLiteral method to AST builder** (30 minutes) - CRITICAL
4. **Implement set integration tests** (30 minutes) - Verify fix
5. **Add stream source tokens** (15 minutes)
6. **Add stream literal parser rule** (30 minutes)
7. **Implement remaining integration tests** (4 hours)

**Total Estimated Time:** ~7.5 hours

---

## AFFECTED INTEGRATION TESTS

The following tests cannot be implemented or are failing due to these issues:

- **Test 4.1: Union of Shape Sets** - Blocked by set literal parsing
- **Test 4.2: Filter and Sort Combined** - Blocked by set literal parsing
- **Test 5.1: FBY Counter** - Blocked by stream literal parsing
- **Test 5.2: FIRST from Stream** - Blocked by stream literal parsing
- **All tests using COMPARE** - May fail due to type error

---

## CONCLUSION

The set literal parsing issue is a COMPLETE BLOCKER that must be fixed before any set operations can be tested or used. This is the highest priority issue that needs immediate attention.

The analysis confirms the user's observation: "sets and set operations are not correctly tested, and there is no 'setLiteral' defined in lexer nor parser" - this assessment is accurate and represents a critical gap in the implementation.

---

**Document Created:** 2026-03-09
**Analysis Method:** Manual code review against specifications
**Analysis Scope:** Compiler, Runtime, Shared packages; all specifications
