#!/bin/bash
# Self-compilation test for Goth
# Tests that the compiler can compile and run Goth programs

set -e

GOTHC="${GOTHC:-./crates/target/release/gothc}"
GOTH="${GOTH:-./crates/target/release/goth}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✓ $1${NC}"
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

echo "=== Goth Self-Compilation Tests ==="
echo ""

# Test 1: Simple expression evaluation
echo "Test 1: Expression evaluation"
result=$($GOTH -e "1 + 2 * 3")
if [ "$result" = "7" ]; then
    pass "Expression: 1 + 2 * 3 = 7"
else
    fail "Expected 7, got $result"
fi

# Test 2: Lambda evaluation
echo "Test 2: Lambda evaluation"
result=$($GOTH -e "(λ→ ₀ + 1) 5")
if [ "$result" = "6" ]; then
    pass "Lambda: (λ→ ₀ + 1) 5 = 6"
else
    fail "Expected 6, got $result"
fi

# Test 3: Array sum
echo "Test 3: Array sum"
result=$($GOTH -e "Σ [1, 2, 3, 4, 5]")
if [ "$result" = "15" ]; then
    pass "Sum: Σ [1, 2, 3, 4, 5] = 15"
else
    fail "Expected 15, got $result"
fi

# Test 4: Map operation
echo "Test 4: Map operation"
result=$($GOTH -e "Σ ([1, 2, 3] ↦ λ→ ₀ × 2)")
if [ "$result" = "12" ]; then
    pass "Map: Σ ([1, 2, 3] ↦ λ→ ₀ × 2) = 12"
else
    fail "Expected 12, got $result"
fi

# Test 5: Compile hello world
echo "Test 5: Compile and run hello world"
cat > /tmp/hello.goth << 'EOF'
╭─ main : () → I
╰─ 42
EOF

$GOTHC /tmp/hello.goth -o /tmp/hello 2>/dev/null
result=$(/tmp/hello)
if [ "$result" = "42" ]; then
    pass "Compile: hello world returns 42"
else
    fail "Expected 42, got $result"
fi

# Test 6: Compile arithmetic
echo "Test 6: Compile arithmetic"
cat > /tmp/arith.goth << 'EOF'
╭─ main : () → I
╰─ (3 + 4) × 5
EOF

$GOTHC /tmp/arith.goth -o /tmp/arith 2>/dev/null
result=$(/tmp/arith)
if [ "$result" = "35" ]; then
    pass "Compile: (3 + 4) × 5 = 35"
else
    fail "Expected 35, got $result"
fi

# Test 7: Compile with lambda
echo "Test 7: Compile lambda"
cat > /tmp/lambda.goth << 'EOF'
╭─ main : I → I
╰─ (λ→ ₀ × ₀) ₀
EOF

$GOTHC /tmp/lambda.goth -o /tmp/lambda 2>/dev/null
result=$(/tmp/lambda 7)
if [ "$result" = "49" ]; then
    pass "Compile: lambda square 7 = 49"
else
    fail "Expected 49, got $result"
fi

# Test 8: Compile cross-function call
echo "Test 8: Cross-function call"
cat > /tmp/cross_fn.goth << 'EOF'
╭─ square : I → I
╰─ ₀ × ₀

╭─ main : () → I
╰─ square 9
EOF

$GOTHC /tmp/cross_fn.goth -o /tmp/cross_fn 2>/dev/null
result=$(/tmp/cross_fn)
if [ "$result" = "81" ]; then
    pass "Compile: cross-function call square 9 = 81"
else
    fail "Expected 81, got $result"
fi

# Test 9: Compile enum/pattern matching
echo "Test 9: Enum pattern matching"
cat > /tmp/enum_test.goth << 'EOF'
enum Maybe τ where Just τ | Nothing

╭─ main : () → I
╰─ match Just 100 {
    Just x → x;
    Nothing → 0
  }
EOF

$GOTHC /tmp/enum_test.goth -o /tmp/enum_test 2>/dev/null
result=$(/tmp/enum_test)
if [ "$result" = "100" ]; then
    pass "Compile: enum pattern match = 100"
else
    fail "Expected 100, got $result"
fi

# Test 10: Compile filter operation
echo "Test 10: Filter operation"
cat > /tmp/filter_test.goth << 'EOF'
╭─ main : () → I
╰─ Σ (ι 10 ▸ λ→ ₀ > 5)
EOF

$GOTHC /tmp/filter_test.goth -o /tmp/filter_test 2>/dev/null
result=$(/tmp/filter_test)
if [ "$result" = "30" ]; then
    pass "Compile: filter ι 10 (x > 5) sum = 30"
else
    fail "Expected 30, got $result"
fi

# Test 11: Parse standard library files
echo "Test 11: Parse standard library"
for file in stdlib/*.goth; do
    $GOTH -p "$file" > /dev/null || fail "Failed to parse $file"
done
pass "All stdlib files parse successfully"

# Cleanup
rm -f /tmp/hello.goth /tmp/hello /tmp/arith.goth /tmp/arith /tmp/lambda.goth /tmp/lambda
rm -f /tmp/cross_fn.goth /tmp/cross_fn /tmp/enum_test.goth /tmp/enum_test
rm -f /tmp/filter_test.goth /tmp/filter_test

echo ""
echo -e "${GREEN}All tests passed!${NC}"
