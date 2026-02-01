mod client;
mod protocol;

use std::io::{self, BufRead, Write};
use std::process::Stdio;

use clap::Parser;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command as TokioCommand;

use client::NockchainClient;
use protocol::{format_response, parse_command, Command};

#[derive(Parser)]
#[command(name = "nockchain-bridge")]
#[command(about = "gRPC bridge between a line-based text protocol and a Nockchain node")]
struct Cli {
    /// Nockchain gRPC endpoint
    #[arg(long, default_value = "http://localhost:50051")]
    endpoint: String,

    /// Optional: path to a Goth program to spawn as a child process.
    /// The bridge will pipe its text protocol over the child's stdin/stdout.
    #[arg(trailing_var_arg = true)]
    goth_args: Vec<String>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if cli.goth_args.is_empty() {
        // Interactive / piped mode — read from own stdin, write to own stdout.
        run_interactive(&cli.endpoint).await;
    } else {
        // Spawn a Goth child process and mediate its I/O.
        if let Err(e) = run_child(&cli.endpoint, &cli.goth_args).await {
            eprintln!("nockchain-bridge: {e}");
            std::process::exit(1);
        }
    }
}

/// Interactive mode: read commands from stdin, call gRPC, print responses.
async fn run_interactive(endpoint: &str) {
    let mut client = match NockchainClient::connect(endpoint).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("nockchain-bridge: {e}");
            // Fall back to offline mode — still parse commands, but always
            // return the connection error so Goth programs can handle it.
            run_offline(&e);
            return;
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // EOF
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let cmd = match parse_command(&line) {
            Ok(c) => c,
            Err(e) => {
                let _ = writeln!(out, "{}", format_response(Err(e)));
                let _ = out.flush();
                continue;
            }
        };

        if matches!(cmd, Command::Quit) {
            break;
        }

        let result = dispatch(&mut client, &cmd).await;
        let formatted = format_response(result);
        let _ = writeln!(out, "{formatted}");
        let _ = out.flush();
    }
}

/// Offline fallback when the node is unreachable — still parses commands so
/// the Goth program gets well-formed ERROR responses instead of hanging.
fn run_offline(error: &str) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        match parse_command(&line) {
            Ok(Command::Quit) => break,
            Ok(_) => {
                let _ = writeln!(out, "{}", format_response(Err(error.to_string())));
                let _ = out.flush();
            }
            Err(e) => {
                let _ = writeln!(out, "{}", format_response(Err(e)));
                let _ = out.flush();
            }
        }
    }
}

/// Spawn a Goth program as a child process and mediate its I/O through the
/// text protocol.
async fn run_child(endpoint: &str, goth_args: &[String]) -> Result<(), String> {
    let mut child = TokioCommand::new("goth")
        .args(goth_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("failed to spawn goth: {e}"))?;

    let child_stdout = child.stdout.take().ok_or("no stdout from child")?;
    let mut child_stdin = child.stdin.take().ok_or("no stdin from child")?;

    let mut client = NockchainClient::connect(endpoint).await.ok();

    let mut reader = BufReader::new(child_stdout).lines();

    while let Ok(Some(line)) = reader.next_line().await {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let cmd = match parse_command(&line) {
            Ok(c) => c,
            Err(e) => {
                let resp = format_response(Err(e));
                let _ = child_stdin.write_all(format!("{resp}\n").as_bytes()).await;
                let _ = child_stdin.flush().await;
                continue;
            }
        };

        if matches!(cmd, Command::Quit) {
            break;
        }

        let result = match &mut client {
            Some(c) => dispatch(c, &cmd).await,
            None => Err("not connected to nockchain node".into()),
        };

        let resp = format_response(result);
        let _ = child_stdin.write_all(format!("{resp}\n").as_bytes()).await;
        let _ = child_stdin.flush().await;
    }

    let _ = child.wait().await;
    Ok(())
}

/// Dispatch a parsed command to the appropriate gRPC call.
async fn dispatch(
    client: &mut NockchainClient,
    cmd: &Command,
) -> Result<protocol::ResponseData, String> {
    match cmd {
        Command::Balance { address } => client.get_balance(address).await,
        Command::SendTx { tx_id, raw_tx_hex } => {
            client.send_transaction(tx_id, raw_tx_hex).await
        }
        Command::TxAccepted { tx_id } => client.tx_accepted(tx_id).await,
        Command::GetBlocks { page_token } => {
            client.get_blocks(page_token.as_deref()).await
        }
        Command::BlockDetail { selector } => client.get_block_details(selector).await,
        Command::TxDetail { tx_id } => client.get_tx_details(tx_id).await,
        Command::Metrics => client.get_metrics().await,
        Command::Quit => unreachable!("QUIT handled before dispatch"),
    }
}
