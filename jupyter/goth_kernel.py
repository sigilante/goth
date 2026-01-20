#!/usr/bin/env python3
"""Goth Jupyter Kernel - A Jupyter kernel for the Goth programming language."""

import subprocess
import re
from ipykernel.kernelbase import Kernel

class GothKernel(Kernel):
    implementation = 'goth'
    implementation_version = '0.1.0'
    language = 'goth'
    language_version = '0.1.0'
    language_info = {
        'name': 'goth',
        'mimetype': 'text/x-goth',
        'file_extension': '.goth',
    }
    banner = """
   ╔═══════════════════════════════════════╗
   ║             𝖌𝖔𝖙𝖍  v0.1.0              ║
   ║   Functional • Tensors • Refinements  ║
   ╚═══════════════════════════════════════╝
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Find the goth binary
        self.goth_path = self._find_goth()

    def _find_goth(self):
        """Find the goth interpreter binary."""
        import shutil
        # Check common locations
        locations = [
            shutil.which('goth'),
            './target/release/goth',
            './target/debug/goth',
            '../target/release/goth',
            '../target/debug/goth',
        ]
        for loc in locations:
            if loc and subprocess.run(['test', '-x', loc or ''], capture_output=True).returncode == 0:
                return loc
        # Default - assume it's in PATH
        return 'goth'

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        """Execute Goth code."""
        if not code.strip():
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
            }

        # Handle magic commands
        if code.strip().startswith('%'):
            return self._handle_magic(code.strip())

        try:
            # Run goth with the expression
            result = subprocess.run(
                [self.goth_path, '-e', code],
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout.strip()
            error = result.stderr.strip()

            # Filter out warnings from stderr
            error_lines = [
                line for line in error.split('\n')
                if not line.strip().startswith('warning:')
                and 'note:' not in line
                and line.strip()
            ]
            filtered_error = '\n'.join(error_lines)

            if result.returncode != 0 and filtered_error:
                # Error occurred
                if not silent:
                    self.send_response(self.iopub_socket, 'stream', {
                        'name': 'stderr',
                        'text': filtered_error + '\n',
                    })
                return {
                    'status': 'error',
                    'execution_count': self.execution_count,
                    'ename': 'GothError',
                    'evalue': filtered_error,
                    'traceback': [],
                }
            else:
                # Success
                if not silent and output:
                    # Format output nicely
                    self.send_response(self.iopub_socket, 'execute_result', {
                        'execution_count': self.execution_count,
                        'data': {'text/plain': output},
                        'metadata': {},
                    })

                return {
                    'status': 'ok',
                    'execution_count': self.execution_count,
                    'payload': [],
                    'user_expressions': {},
                }

        except subprocess.TimeoutExpired:
            if not silent:
                self.send_response(self.iopub_socket, 'stream', {
                    'name': 'stderr',
                    'text': 'Execution timed out (30s limit)\n',
                })
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': 'TimeoutError',
                'evalue': 'Execution timed out',
                'traceback': [],
            }
        except Exception as e:
            if not silent:
                self.send_response(self.iopub_socket, 'stream', {
                    'name': 'stderr',
                    'text': f'Kernel error: {e}\n',
                })
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': [],
            }

    def _handle_magic(self, code):
        """Handle magic commands."""
        if code == '%help':
            help_text = """
Goth Jupyter Kernel Magic Commands:
  %help     - Show this help
  %version  - Show version info
  %ast      - Show AST for expression (use: %ast <expr>)
  %type     - Show type of expression (use: %type <expr>)

Goth Syntax Quick Reference:
  λ→ ₀ + 1       Lambda (or \\-> _0 + 1)
  [1, 2, 3]      Array
  ⟨x, y⟩         Tuple
  Σ arr          Sum (or +/ arr)
  arr ↦ f        Map (or arr -: f)
  arr ▸ p        Filter
"""
            self.send_response(self.iopub_socket, 'stream', {
                'name': 'stdout',
                'text': help_text,
            })
        elif code == '%version':
            self.send_response(self.iopub_socket, 'stream', {
                'name': 'stdout',
                'text': f'Goth Kernel {self.implementation_version}\n',
            })
        elif code.startswith('%ast '):
            expr = code[5:]
            result = subprocess.run(
                [self.goth_path, '-e', expr, '--ast'],
                capture_output=True, text=True
            )
            self.send_response(self.iopub_socket, 'stream', {
                'name': 'stdout',
                'text': result.stdout or result.stderr,
            })
        elif code.startswith('%type '):
            expr = code[6:]
            result = subprocess.run(
                [self.goth_path, '-e', expr, '--check'],
                capture_output=True, text=True
            )
            output = result.stdout + result.stderr
            # Extract just the type line
            for line in output.split('\n'):
                if line.startswith('Type:'):
                    self.send_response(self.iopub_socket, 'stream', {
                        'name': 'stdout',
                        'text': line + '\n',
                    })
                    break
            else:
                self.send_response(self.iopub_socket, 'stream', {
                    'name': 'stdout',
                    'text': output,
                })
        else:
            self.send_response(self.iopub_socket, 'stream', {
                'name': 'stderr',
                'text': f'Unknown magic command: {code.split()[0]}\n',
            })

        return {
            'status': 'ok',
            'execution_count': self.execution_count,
            'payload': [],
            'user_expressions': {},
        }

    def do_is_complete(self, code):
        """Check if code is complete (for multi-line input)."""
        # Simple heuristic: check balanced brackets
        opens = code.count('[') + code.count('(') + code.count('{') + code.count('⟨')
        closes = code.count(']') + code.count(')') + code.count('}') + code.count('⟩')

        if opens > closes:
            return {'status': 'incomplete', 'indent': '  '}

        # Check for function declaration
        if '╭─' in code and '╰─' not in code:
            return {'status': 'incomplete', 'indent': ''}
        if '/-' in code and '\\-' not in code:
            return {'status': 'incomplete', 'indent': ''}

        # Check for let without in
        if 'let ' in code and ' in ' not in code and '←' in code:
            return {'status': 'incomplete', 'indent': '   '}

        return {'status': 'complete'}

    def do_complete(self, code, cursor_pos):
        """Provide code completion."""
        # Extract word at cursor
        text_before = code[:cursor_pos]
        match = re.search(r'[\w₀₁₂₃₄₅₆₇₈₉]+$', text_before)

        if not match:
            return {
                'status': 'ok',
                'matches': [],
                'cursor_start': cursor_pos,
                'cursor_end': cursor_pos,
                'metadata': {},
            }

        word = match.group()
        start = cursor_pos - len(word)

        # Built-in completions
        completions = [
            # Keywords
            'let', 'in', 'if', 'then', 'else', 'match', 'enum', 'where',
            # Types
            'I64', 'I32', 'F64', 'F32', 'Bool', 'Char', 'String',
            # Primitives
            'sqrt', 'exp', 'ln', 'sin', 'cos', 'tan',
            'floor', 'ceil', 'round', 'abs', 'length',
            'sum', 'prod', 'min', 'max',
            'iota', 'range', 'reverse',
            'dot', 'matmul', 'transpose', 'norm',
            'print', 'true', 'false',
            # Unicode
            'λ', 'Σ', 'Π', '↦', '▸', '∘', '⊤', '⊥',
            '₀', '₁', '₂', '₃', '₄', '₅',
        ]

        matches = [c for c in completions if c.startswith(word)]

        return {
            'status': 'ok',
            'matches': matches,
            'cursor_start': start,
            'cursor_end': cursor_pos,
            'metadata': {},
        }


if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=GothKernel)
