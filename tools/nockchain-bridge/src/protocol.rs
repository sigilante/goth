//! Text protocol: parse commands from stdin lines, format responses for stdout.
//!
//! Commands are space-delimited, one per line.  Responses start with `OK` or
//! `ERROR` so a Goth program can branch on `(words resp)[0]`.

/// Parsed command from a single input line.
#[derive(Debug, Clone)]
pub enum Command {
    Balance { address: String },
    SendTx { tx_id: String, raw_tx_hex: String },
    TxAccepted { tx_id: String },
    GetBlocks { page_token: Option<String> },
    BlockDetail { selector: String },
    TxDetail { tx_id: String },
    Metrics,
    Quit,
}

/// Typed response data returned by the gRPC client layer.
#[derive(Debug)]
pub enum ResponseData {
    /// Single-line value (BALANCE, SEND_TX, TX_ACCEPTED, BLOCK_DETAIL, TX_DETAIL, METRICS).
    Line(String),
    /// Multi-line value terminated by a blank line (GET_BLOCKS).
    Lines(Vec<String>),
}

/// Parse a single input line into a `Command`.
pub fn parse_command(line: &str) -> Result<Command, String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Err("empty command".into());
    }
    match parts[0].to_uppercase().as_str() {
        "BALANCE" => {
            let addr = parts.get(1).ok_or("BALANCE requires <address>")?;
            Ok(Command::Balance {
                address: (*addr).to_string(),
            })
        }
        "SEND_TX" => {
            let tx_id = parts.get(1).ok_or("SEND_TX requires <tx_id> <raw_tx_hex>")?;
            let raw = parts.get(2).ok_or("SEND_TX requires <raw_tx_hex>")?;
            Ok(Command::SendTx {
                tx_id: (*tx_id).to_string(),
                raw_tx_hex: (*raw).to_string(),
            })
        }
        "TX_ACCEPTED" => {
            let tx_id = parts.get(1).ok_or("TX_ACCEPTED requires <tx_id>")?;
            Ok(Command::TxAccepted {
                tx_id: (*tx_id).to_string(),
            })
        }
        "GET_BLOCKS" => {
            let page_token = parts.get(1).map(|s| (*s).to_string());
            Ok(Command::GetBlocks { page_token })
        }
        "BLOCK_DETAIL" => {
            let sel = parts.get(1).ok_or("BLOCK_DETAIL requires <height_or_id>")?;
            Ok(Command::BlockDetail {
                selector: (*sel).to_string(),
            })
        }
        "TX_DETAIL" => {
            let tx_id = parts.get(1).ok_or("TX_DETAIL requires <tx_id>")?;
            Ok(Command::TxDetail {
                tx_id: (*tx_id).to_string(),
            })
        }
        "METRICS" => Ok(Command::Metrics),
        "QUIT" => Ok(Command::Quit),
        other => Err(format!("unknown command: {other}")),
    }
}

/// Format a result into one or more output lines.
pub fn format_response(result: Result<ResponseData, String>) -> String {
    match result {
        Ok(ResponseData::Line(data)) => format!("OK {data}"),
        Ok(ResponseData::Lines(lines)) => {
            let mut out = String::new();
            for line in &lines {
                out.push_str(&format!("OK {line}\n"));
            }
            // Blank line terminates the multi-line block.
            out.push('\n');
            out
        }
        Err(msg) => {
            // Collapse whitespace so the error is a single parseable line.
            let clean: String = msg.split_whitespace().collect::<Vec<_>>().join(" ");
            format!("ERROR {clean}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_balance() {
        let cmd = parse_command("BALANCE abc123").unwrap();
        assert!(matches!(cmd, Command::Balance { address } if address == "abc123"));
    }

    #[test]
    fn parse_quit() {
        assert!(matches!(parse_command("QUIT").unwrap(), Command::Quit));
    }

    #[test]
    fn parse_empty() {
        assert!(parse_command("").is_err());
    }

    #[test]
    fn format_ok_line() {
        let resp = format_response(Ok(ResponseData::Line("42".into())));
        assert_eq!(resp, "OK 42");
    }

    #[test]
    fn format_error() {
        let resp = format_response(Err("connection  refused".into()));
        assert_eq!(resp, "ERROR connection refused");
    }

    #[test]
    fn parse_get_blocks_no_page() {
        let cmd = parse_command("GET_BLOCKS").unwrap();
        assert!(matches!(cmd, Command::GetBlocks { page_token: None }));
    }

    #[test]
    fn parse_get_blocks_with_page() {
        let cmd = parse_command("GET_BLOCKS tok123").unwrap();
        assert!(
            matches!(cmd, Command::GetBlocks { page_token: Some(ref t) } if t == "tok123")
        );
    }

    #[test]
    fn parse_case_insensitive() {
        let cmd = parse_command("balance MyAddr").unwrap();
        assert!(matches!(cmd, Command::Balance { address } if address == "MyAddr"));
    }
}
