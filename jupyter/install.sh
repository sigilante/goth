#!/bin/bash
# Install Goth Jupyter Kernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/goth"
VENV_DIR="${HOME}/.local/share/goth-jupyter-venv"

echo "Installing Goth Jupyter Kernel..."

# Check dependencies
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

# Copy kernel files
cp "$SCRIPT_DIR/goth_kernel.py" "$KERNEL_DIR/"

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

echo "Kernel installed to: $KERNEL_DIR"
echo ""
echo "Make sure 'goth' is in your PATH, or build it first:"
echo "  cd crates && cargo build --release"
echo "  export PATH=\"\$PATH:\$(pwd)/target/release\""
echo ""
echo "To verify installation:"
echo "  jupyter kernelspec list"
echo ""
echo "To start Jupyter with Goth kernel:"
echo "  jupyter notebook"
echo "  # Then select 'Goth' from New menu"
