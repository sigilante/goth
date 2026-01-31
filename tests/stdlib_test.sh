#!/bin/bash
# Stdlib test suite for Goth
# Tests the standard library modules by running .goth test files
# and comparing stdout against expected output.

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

run_test() {
    local name="$1"
    local file="$2"
    local expected="$3"

    local actual
    actual=$($GOTH "$file" 2>&1) || true

    if [ "$actual" = "$expected" ]; then
        pass "$name"
    else
        fail "$name"
        echo -e "${YELLOW}  Expected:${NC}"
        echo "$expected" | head -5
        echo -e "${YELLOW}  Got:${NC}"
        echo "$actual" | head -5
        if [ "$(echo "$expected" | wc -l)" -gt 5 ]; then
            echo "  ... (truncated)"
        fi
    fi
}

echo "=== Goth Stdlib Tests ==="
echo ""

# --- Test: prelude ---
echo "Module: prelude"
EXPECTED_PRELUDE=$(cat <<'EOF'
42
10
7
12
⊥
⊤
⊥
⊤
⊥
⊤
⊥
⊤
⊥
⊤
⊤
⊥
1
0
⊥
⊤
⊤
⊥
⊤
⊤
⊤
⊤
11
9
5
-1
1
0
0
3
7
5
16
-5
10
20
⟨2, 1⟩
1
2
3
4
EOF
)
run_test "prelude" "tests/stdlib/test_prelude.goth" "$EXPECTED_PRELUDE"

# --- Test: list ---
echo "Module: list"
EXPECTED_LIST=$(cat <<'EOF'
10
30
[20 30]
[10 20]
20
99
[0 10 20 30]
[10 20 30 40]
[1 2 3 4]
[7 7 7]
[1 2 3 4 5]
[10 20 30]
[4 5]
[3 2 1]
6
24
⊤
⊥
⊤
2
⊥
3
EOF
)
run_test "list" "tests/stdlib/test_list.goth" "$EXPECTED_LIST"

# --- Test: math ---
echo "Module: math"
EXPECTED_MATH=$(cat <<'EOF'
314
272
1618
1414
7
3
2.5
-1
0
1
-1
0
1
25
27
4
5
272
100
200
300
⊤
⊤
⊤
⊥
0
3
3
1
3
4
3
7
5
2
8
5
EOF
)
run_test "math" "tests/stdlib/test_math.goth" "$EXPECTED_MATH"

# --- Test: option ---
echo "Module: option"
EXPECTED_OPTION=$(cat <<'EOF'
⟨⊤, 42⟩
⟨⊤, 3.14⟩
⊤
⊥
42
3.14
⟨⊤, 10⟩
⟨⊤, 0⟩
⟨⊤, 5⟩
⟨⊥, 0⟩
⟨⊤, 1⟩
⟨⊤, 5⟩
⟨⊥, 0⟩
⟨⊥, 0⟩
⊤
⊥
Some(42)
Some(3.14)
EOF
)
run_test "option" "tests/stdlib/test_option.goth" "$EXPECTED_OPTION"

# --- Test: result ---
echo "Module: result"
EXPECTED_RESULT=$(cat <<'EOF'
⟨⊤, 42, 0⟩
⟨⊥, 0, 1⟩
⟨⊤, 10, ""⟩
⟨⊥, 0, "bad"⟩
⊤
⊥
⊤
⊥
42
0
42
0
⟨⊤, 10, 0⟩
⟨⊥, 0, 1⟩
⟨⊤, 5, 0⟩
⟨⊥, 0, -1⟩
⊤
⊥
⊥
Ok(42)
Err(1)
EOF
)
run_test "result" "tests/stdlib/test_result.goth" "$EXPECTED_RESULT"

# --- Test: string ---
echo "Module: string"
EXPECTED_STRING=$(cat <<'EOF'
5
⊤
⊥
⊤
⊤
⊥
⊤
⊤
⊤
⊤
⊤
⊥
⊤
⊤
42
true
false
0
3
hello
foo
EOF
)
run_test "string" "tests/stdlib/test_string.goth" "$EXPECTED_STRING"

# --- Summary ---
echo ""
TOTAL=$((PASS + FAIL))
echo "Results: $PASS/$TOTAL passed"
if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}$FAIL test(s) failed${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
fi
