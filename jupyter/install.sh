#!/bin/bash
# Install Goth Jupyter Kernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/goth"

echo "Installing Goth Jupyter Kernel..."

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

if ! python3 -c "import ipykernel" 2>/dev/null; then
    echo "Installing ipykernel..."
    pip3 install ipykernel
fi

# Create kernel directory
mkdir -p "$KERNEL_DIR"

# Copy kernel files
cp "$SCRIPT_DIR/goth_kernel.py" "$KERNEL_DIR/"
cp "$SCRIPT_DIR/kernel.json" "$KERNEL_DIR/"

# Update kernel.json with correct path
cat > "$KERNEL_DIR/kernel.json" << EOF
{
    "argv": [
        "python3",
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
echo "Make sure 'goth' is in your PATH, or set GOTH_PATH environment variable."
echo ""
echo "To verify installation:"
echo "  jupyter kernelspec list"
echo ""
echo "To start Jupyter with Goth kernel:"
echo "  jupyter notebook"
echo "  # Then select 'Goth' from New menu"
