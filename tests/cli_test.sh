#!/bin/bash
# CLI smoke test suite for Goth
# Tests all major CLI modes of the goth interpreter binary.

set -e

GOTH="${GOTH:-./crates/target/release/goth}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASS=0
FAIL=0

pass() {
    echo -e "${GREEN}✓ $1${NC}"
    PASS=$((PASS + 1))
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    FAIL=$((FAIL + 1))
}

# Test exact output match
run_test() {
    local name="$1"
    local expected="$2"
    shift 2

    local actual
    actual=$("$@" 2>&1) || true

    if [ "$actual" = "$expected" ]; then
        pass "$name"
    else
        fail "$name"
        echo -e "${YELLOW}  Expected:${NC} $(echo "$expected" | head -1)"
        echo -e "${YELLOW}  Got:${NC} $(echo "$actual" | head -1)"
    fi
}

# Test that output contains a substring (case-sensitive)
run_test_contains() {
    local name="$1"
    local substring="$2"
    shift 2

    local actual
    actual=$("$@" 2>&1) || true

    if echo "$actual" | grep -qF "$substring"; then
        pass "$name"
    else
        fail "$name"
        echo -e "${YELLOW}  Expected to contain:${NC} $substring"
        echo -e "${YELLOW}  Got:${NC} $(echo "$actual" | head -3)"
    fi
}

# Test that output contains a substring (case-insensitive)
run_test_contains_i() {
    local name="$1"
    local substring="$2"
    shift 2

    local actual
    actual=$("$@" 2>&1) || true

    if echo "$actual" | grep -qiF "$substring"; then
        pass "$name"
    else
        fail "$name"
        echo -e "${YELLOW}  Expected to contain (case-insensitive):${NC} $substring"
        echo -e "${YELLOW}  Got:${NC} $(echo "$actual" | head -3)"
    fi
}

# Test that stdout is empty
run_test_empty() {
    local name="$1"
    shift

    local actual
    actual=$("$@" 2>&1) || true

    if [ -z "$actual" ]; then
        pass "$name"
    else
        fail "$name"
        echo -e "${YELLOW}  Expected empty output${NC}"
        echo -e "${YELLOW}  Got:${NC} $(echo "$actual" | head -1)"
    fi
}

echo "=== Goth CLI Smoke Tests ==="
echo ""

# --- Section 1: Expression evaluation (-e) ---
echo "Section: Expression evaluation (-e)"
run_test "integer arithmetic" "7" $GOTH -e "1 + 2 * 3"
run_test "negative result" "-7" $GOTH -e "3 - 10"
run_test "boolean true" "⊤" $GOTH -e "3 > 2"
run_test "boolean false" "⊥" $GOTH -e "3 < 2"
run_test "lambda application" "6" $GOTH -e "(λ→ ₀ + 1) 5"
run_test "array sum" "15" $GOTH -e "Σ [1, 2, 3, 4, 5]"
run_test "let binding" "15" $GOTH -e "let x ← 10 in x + 5"
echo ""

# --- Section 2: File execution ---
echo "Section: File execution"
run_test "identity" "42" $GOTH examples/basic/identity.goth 42
run_test "add_one" "100" $GOTH examples/basic/add_one.goth 99
run_test "square" "49" $GOTH examples/basic/square.goth 7
run_test "factorial" "120" $GOTH examples/recursion/factorial.goth 5
run_test "fibonacci" "55" $GOTH examples/recursion/fibonacci.goth 10
run_test "isPrime" "⊤" $GOTH examples/algorithms/isPrime.goth 17
echo ""

# --- Section 3: Multi-argument programs ---
echo "Section: Multi-argument programs"
run_test "gcd" "4" $GOTH examples/recursion/gcd.goth 12 8
run_test "compose" "36" $GOTH examples/higher-order/compose.goth 3
echo ""

# --- Section 4: Parse-only mode (-p) ---
echo "Section: Parse-only mode (-p)"
run_test_contains "parse file" "Parsed:" $GOTH -p examples/basic/identity.goth
run_test_empty "parse expression" $GOTH -p -e "1 + 2"
echo ""

# --- Section 5: AST mode (-a) ---
echo "Section: AST mode (-a)"
run_test_contains "ast expression" "AST:" $GOTH -a -e "1 + 2"
run_test_contains "ast file" "Parsed AST:" $GOTH -a examples/basic/identity.goth
echo ""

# --- Section 6: Type check mode (-c) ---
echo "Section: Type check mode (-c)"
run_test_contains "type check expression" "Type:" $GOTH -c -e "1 + 2"
run_test "type check file with args" "100" $GOTH -c examples/basic/add_one.goth 99
echo ""

# --- Section 7: JSON workflow ---
echo "Section: JSON workflow"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

run_test_contains "to-json produces JSON" '"decls"' $GOTH --to-json examples/basic/identity.goth

# Compact JSON should be a single line
COMPACT_OUTPUT=$($GOTH --compact --to-json examples/basic/identity.goth 2>&1) || true
COMPACT_LINES=$(echo "$COMPACT_OUTPUT" | wc -l)
if [ "$COMPACT_LINES" -eq 1 ] && echo "$COMPACT_OUTPUT" | grep -qF '"decls"'; then
    pass "compact to-json is single line"
else
    fail "compact to-json is single line"
    echo -e "${YELLOW}  Lines: $COMPACT_LINES${NC}"
fi

# JSON round-trip
$GOTH --to-json examples/basic/add_one.goth > "$TMPDIR/add_one.json" 2>&1
ROUNDTRIP=$($GOTH --from-json "$TMPDIR/add_one.json" 99 2>&1) || true
if echo "$ROUNDTRIP" | grep -qF "100"; then
    pass "json round-trip preserves semantics"
else
    fail "json round-trip preserves semantics"
    echo -e "${YELLOW}  Expected output to contain: 100${NC}"
    echo -e "${YELLOW}  Got:${NC} $ROUNDTRIP"
fi
echo ""

# --- Section 8: No-main mode ---
echo "Section: No-main mode (--no-main)"
run_test_contains "no-main shows declarations" "fn main" $GOTH --no-main examples/basic/identity.goth
echo ""

# --- Section 9: Error handling ---
echo "Section: Error handling"
run_test_contains_i "nonexistent file" "error" $GOTH nonexistent_file.goth
run_test_contains_i "syntax error" "error" $GOTH -e "1 + + 2"

# Render from JSON
$GOTH --to-json examples/basic/add_one.goth > "$TMPDIR/render_test.json" 2>&1
RENDER_OUTPUT=$($GOTH --render --from-json "$TMPDIR/render_test.json" 2>&1) || true
if echo "$RENDER_OUTPUT" | grep -qF "module" && echo "$RENDER_OUTPUT" | grep -qF "main"; then
    pass "render from json produces source"
else
    fail "render from json produces source"
    echo -e "${YELLOW}  Expected 'module' and 'main' in output${NC}"
    echo -e "${YELLOW}  Got:${NC} $(echo "$RENDER_OUTPUT" | head -3)"
fi
echo ""

# --- Summary ---
TOTAL=$((PASS + FAIL))
echo "Results: $PASS/$TOTAL passed"
if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}$FAIL test(s) failed${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
fi
