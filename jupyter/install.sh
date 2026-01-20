#!/bin/bash
# Install Goth Jupyter Kernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/goth"
VENV_DIR="${HOME}/.local/share/goth-jupyter-venv"

echo "Installing Goth Jupyter Kernel..."

# Find goth binary
GOTH_PATH=""
if command -v goth &> /dev/null; then
    GOTH_PATH="$(which goth)"
elif [ -x "$PROJECT_DIR/crates/target/release/goth" ]; then
    GOTH_PATH="$PROJECT_DIR/crates/target/release/goth"
elif [ -x "$PROJECT_DIR/crates/target/debug/goth" ]; then
    GOTH_PATH="$PROJECT_DIR/crates/target/debug/goth"
else
    echo "Building goth..."
    (cd "$PROJECT_DIR/crates" && cargo build --release -p goth-cli --bin goth)
    GOTH_PATH="$PROJECT_DIR/crates/target/release/goth"
fi

if [ ! -x "$GOTH_PATH" ]; then
    echo "Error: Could not find or build goth binary"
    exit 1
fi

echo "Using goth at: $GOTH_PATH"

# Check Python dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Create virtual environment if ipykernel not available
if ! python3 -c "import ipykernel" 2>/dev/null; then
    echo "Creating virtual environment for kernel..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet ipykernel
    PYTHON_PATH="$VENV_DIR/bin/python3"
else
    PYTHON_PATH="$(which python3)"
fi

# Create kernel directory
mkdir -p "$KERNEL_DIR"

# Copy kernel files and embed goth path
sed "s|self.goth_path = self._find_goth()|self.goth_path = '$GOTH_PATH'|" \
    "$SCRIPT_DIR/goth_kernel.py" > "$KERNEL_DIR/goth_kernel.py"

# Update kernel.json with correct path
cat > "$KERNEL_DIR/kernel.json" << EOF
{
    "argv": [
        "$PYTHON_PATH",
        "$KERNEL_DIR/goth_kernel.py",
        "-f",
        "{connection_file}"
    ],
    "display_name": "Goth",
    "language": "goth",
    "metadata": {
        "debugger": false
    }
}
EOF

echo ""
echo "Kernel installed to: $KERNEL_DIR"
echo "Goth binary: $GOTH_PATH"
echo ""
echo "To verify installation:"
echo "  jupyter kernelspec list"
echo ""
echo "To start Jupyter with Goth kernel:"
echo "  jupyter notebook"
echo "  # Then select 'Goth' from New menu"
