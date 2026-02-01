# nockchain-bridge

A standalone Rust CLI that translates between a simple line-based text protocol
(suitable for Goth's `readLine`/`print`) and Nockchain's gRPC API.

## Architecture

```
Goth program  <──stdin/stdout──>  nockchain-bridge  <──gRPC──>  Nockchain node
  (print/readLine)                 (tonic client)               (localhost:50051)
```

The bridge reads commands from stdin (or from a spawned Goth child process),
calls the corresponding gRPC method, and writes space-delimited responses that
Goth can parse with `words`.

## Building

```sh
cd tools/nockchain-bridge
cargo build --release
```

This is a standalone project (not part of the main `crates/` workspace) to keep
heavy async/network dependencies out of the language toolchain.

## Usage

### Interactive mode

```sh
# Start the bridge (connects to a local Nockchain node)
./target/release/nockchain-bridge

# Type commands:
BALANCE 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY
# → OK 1000000

METRICS
# → OK 150 200 0.950000

QUIT
```

### Spawning a Goth program

```sh
# The bridge spawns goth, pipes the text protocol over stdin/stdout
nockchain-bridge examples/nockchain/balance.goth myWalletAddress

# Custom endpoint
nockchain-bridge --endpoint http://node.example.com:50051 examples/nockchain/blocks.goth 0
```

## Protocol

Commands are one per line, space-separated. Responses start with `OK` or `ERROR`.

| Command | Args | Response |
|---------|------|----------|
| `BALANCE` | `<address>` | `OK <total_nicks>` |
| `SEND_TX` | `<tx_id> <raw_tx_hex>` | `OK` |
| `TX_ACCEPTED` | `<tx_id>` | `OK true\|false` |
| `GET_BLOCKS` | `[page_token]` | Multi-line: `OK <height> <block_id> <timestamp>` per block, blank line terminates |
| `BLOCK_DETAIL` | `<height_or_id>` | `OK <height> <block_id> <parent> <timestamp> <tx_count>` |
| `TX_DETAIL` | `<tx_id>` | `OK <tx_id> <height> <timestamp> <total_input> <total_output>` |
| `METRICS` | | `OK <cache_height> <heaviest_height> <coverage_ratio>` |
| `QUIT` | | Bridge exits |

## Goth examples

See `examples/nockchain/` for companion Goth programs:
- `balance.goth` — query wallet balance
- `blocks.goth` — list recent blocks
- `tx_status.goth` — check transaction acceptance
