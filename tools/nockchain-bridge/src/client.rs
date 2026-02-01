//! gRPC client wrapper — connects to a Nockchain node and translates each
//! `Command` into the appropriate RPC, returning `ResponseData` or an error
//! string suitable for the text protocol.

use tonic::transport::Channel;

use crate::protocol::ResponseData;

// Generated protobuf / gRPC stubs.
// The module hierarchy must mirror the protobuf package paths so that
// tonic-build's `super::` references resolve correctly.
pub mod nockchain {
    pub mod common {
        pub mod v1 {
            tonic::include_proto!("nockchain.common.v1");
        }
        pub mod v2 {
            tonic::include_proto!("nockchain.common.v2");
        }
    }
    pub mod public {
        pub mod v2 {
            tonic::include_proto!("nockchain.public.v2");
        }
    }
}

use nockchain::common::v1::{Base58Hash, Base58Pubkey, PageRequest};
use nockchain::public::v2::{
    nockchain_block_service_client::NockchainBlockServiceClient,
    nockchain_metrics_service_client::NockchainMetricsServiceClient,
    nockchain_service_client::NockchainServiceClient,
    GetBlockDetailsRequest, GetBlocksRequest, GetExplorerMetricsRequest,
    GetTransactionDetailsRequest, TransactionAcceptedRequest, WalletGetBalanceRequest,
    WalletSendTransactionRequest,
};

/// Helper: render a `Hash` as a hex-ish string of its belt values.
fn hash_display(h: &nockchain::common::v1::Hash) -> String {
    format!(
        "{:016x}{:016x}{:016x}{:016x}{:016x}",
        h.belt_1.as_ref().map_or(0, |b| b.value),
        h.belt_2.as_ref().map_or(0, |b| b.value),
        h.belt_3.as_ref().map_or(0, |b| b.value),
        h.belt_4.as_ref().map_or(0, |b| b.value),
        h.belt_5.as_ref().map_or(0, |b| b.value),
    )
}

pub struct NockchainClient {
    wallet: NockchainServiceClient<Channel>,
    blocks: NockchainBlockServiceClient<Channel>,
    metrics: NockchainMetricsServiceClient<Channel>,
}

impl NockchainClient {
    /// Connect to the Nockchain gRPC endpoint.
    pub async fn connect(endpoint: &str) -> Result<Self, String> {
        let channel = Channel::from_shared(endpoint.to_string())
            .map_err(|e| format!("invalid endpoint: {e}"))?
            .connect()
            .await
            .map_err(|e| format!("connection failed: {e}"))?;
        Ok(Self {
            wallet: NockchainServiceClient::new(channel.clone()),
            blocks: NockchainBlockServiceClient::new(channel.clone()),
            metrics: NockchainMetricsServiceClient::new(channel),
        })
    }

    /// BALANCE <address> → OK <total_nicks>
    pub async fn get_balance(&mut self, address: &str) -> Result<ResponseData, String> {
        let req = WalletGetBalanceRequest {
            selector: Some(
                nockchain::public::v2::wallet_get_balance_request::Selector::Address(
                    Base58Pubkey {
                        key: address.to_string(),
                    },
                ),
            ),
            page: Some(PageRequest {
                client_page_items_limit: 100,
                page_token: String::new(),
                max_bytes: 0,
            }),
        };
        let resp = self
            .wallet
            .wallet_get_balance(req)
            .await
            .map_err(|e| format!("rpc error: {e}"))?
            .into_inner();
        match resp.result {
            Some(nockchain::public::v2::wallet_get_balance_response::Result::Balance(bal)) => {
                let total: u64 = bal
                    .notes
                    .iter()
                    .filter_map(|entry| {
                        entry.note.as_ref().and_then(|n| match &n.note_version {
                            Some(nockchain::common::v2::note::NoteVersion::Legacy(leg)) => {
                                leg.assets.as_ref().map(|a| a.value)
                            }
                            Some(nockchain::common::v2::note::NoteVersion::V1(v1)) => {
                                v1.assets.as_ref().map(|a| a.value)
                            }
                            None => None,
                        })
                    })
                    .sum();
                Ok(ResponseData::Line(total.to_string()))
            }
            Some(nockchain::public::v2::wallet_get_balance_response::Result::Error(e)) => {
                Err(e.message)
            }
            None => Err("empty response".into()),
        }
    }

    /// SEND_TX <tx_id> <raw_tx_hex> → OK
    pub async fn send_transaction(
        &mut self,
        _tx_id: &str,
        _raw_tx_hex: &str,
    ) -> Result<ResponseData, String> {
        // Building a full RawTransaction from hex requires deserializing the
        // protobuf-encoded bytes.  For now we send a minimal request and let
        // the node validate.
        let req = WalletSendTransactionRequest {
            tx_id: None,
            raw_tx: None,
        };
        let resp = self
            .wallet
            .wallet_send_transaction(req)
            .await
            .map_err(|e| format!("rpc error: {e}"))?
            .into_inner();
        match resp.result {
            Some(nockchain::public::v2::wallet_send_transaction_response::Result::Ack(_)) => {
                Ok(ResponseData::Line(String::new()))
            }
            Some(nockchain::public::v2::wallet_send_transaction_response::Result::Error(e)) => {
                Err(e.message)
            }
            None => Err("empty response".into()),
        }
    }

    /// TX_ACCEPTED <tx_id> → OK true|false
    pub async fn tx_accepted(&mut self, tx_id: &str) -> Result<ResponseData, String> {
        let req = TransactionAcceptedRequest {
            tx_id: Some(Base58Hash {
                hash: tx_id.to_string(),
            }),
        };
        let resp = self
            .wallet
            .transaction_accepted(req)
            .await
            .map_err(|e| format!("rpc error: {e}"))?
            .into_inner();
        match resp.result {
            Some(nockchain::public::v2::transaction_accepted_response::Result::Accepted(b)) => {
                Ok(ResponseData::Line(b.to_string()))
            }
            Some(nockchain::public::v2::transaction_accepted_response::Result::Error(e)) => {
                Err(e.message)
            }
            None => Err("empty response".into()),
        }
    }

    /// GET_BLOCKS [page_token] → multi-line OK <height> <block_id> <timestamp>
    pub async fn get_blocks(
        &mut self,
        page_token: Option<&str>,
    ) -> Result<ResponseData, String> {
        let req = GetBlocksRequest {
            page: Some(PageRequest {
                client_page_items_limit: 20,
                page_token: page_token.unwrap_or_default().to_string(),
                max_bytes: 0,
            }),
        };
        let resp = self
            .blocks
            .get_blocks(req)
            .await
            .map_err(|e| format!("rpc error: {e}"))?
            .into_inner();
        match resp.result {
            Some(nockchain::public::v2::get_blocks_response::Result::Blocks(data)) => {
                let lines: Vec<String> = data
                    .blocks
                    .iter()
                    .map(|b| {
                        let id = b
                            .block_id
                            .as_ref()
                            .map_or_else(|| "unknown".to_string(), hash_display);
                        format!("{} {} {}", b.height, id, b.timestamp)
                    })
                    .collect();
                Ok(ResponseData::Lines(lines))
            }
            Some(nockchain::public::v2::get_blocks_response::Result::Error(e)) => {
                Err(e.message)
            }
            None => Err("empty response".into()),
        }
    }

    /// BLOCK_DETAIL <height_or_id> → OK <height> <block_id> <parent> <timestamp> <tx_count>
    pub async fn get_block_details(
        &mut self,
        selector: &str,
    ) -> Result<ResponseData, String> {
        let sel = if let Ok(h) = selector.parse::<u64>() {
            nockchain::public::v2::get_block_details_request::Selector::Height(h)
        } else {
            nockchain::public::v2::get_block_details_request::Selector::BlockId(Base58Hash {
                hash: selector.to_string(),
            })
        };
        let req = GetBlockDetailsRequest {
            selector: Some(sel),
        };
        let resp = self
            .blocks
            .get_block_details(req)
            .await
            .map_err(|e| format!("rpc error: {e}"))?
            .into_inner();
        match resp.result {
            Some(nockchain::public::v2::get_block_details_response::Result::Details(d)) => {
                let id = d
                    .block_id
                    .as_ref()
                    .map_or_else(|| "unknown".into(), hash_display);
                let parent = d
                    .parent
                    .as_ref()
                    .map_or_else(|| "unknown".into(), hash_display);
                Ok(ResponseData::Line(format!(
                    "{} {} {} {} {}",
                    d.height, id, parent, d.timestamp, d.tx_count
                )))
            }
            Some(nockchain::public::v2::get_block_details_response::Result::Error(e)) => {
                Err(e.message)
            }
            None => Err("empty response".into()),
        }
    }

    /// TX_DETAIL <tx_id> → OK <tx_id> <height> <timestamp> <total_input> <total_output>
    pub async fn get_tx_details(&mut self, tx_id: &str) -> Result<ResponseData, String> {
        let req = GetTransactionDetailsRequest {
            tx_id: Some(Base58Hash {
                hash: tx_id.to_string(),
            }),
        };
        let resp = self
            .blocks
            .get_transaction_details(req)
            .await
            .map_err(|e| format!("rpc error: {e}"))?
            .into_inner();
        match resp.result {
            Some(nockchain::public::v2::get_transaction_details_response::Result::Details(
                d,
            )) => {
                let input = d.total_input.as_ref().map_or(0, |n| n.value);
                // total_output is a oneof field
                let output = match &d.total_output_required {
                    Some(
                        nockchain::public::v2::transaction_details::TotalOutputRequired::TotalOutput(n),
                    ) => n.value,
                    None => 0,
                };
                Ok(ResponseData::Line(format!(
                    "{} {} {} {} {}",
                    d.tx_id, d.height, d.timestamp, input, output
                )))
            }
            Some(
                nockchain::public::v2::get_transaction_details_response::Result::Pending(_),
            ) => Ok(ResponseData::Line("pending".into())),
            Some(nockchain::public::v2::get_transaction_details_response::Result::Error(e)) => {
                Err(e.message)
            }
            None => Err("empty response".into()),
        }
    }

    /// METRICS → OK <cache_height> <heaviest_height> <coverage_ratio>
    pub async fn get_metrics(&mut self) -> Result<ResponseData, String> {
        let req = GetExplorerMetricsRequest {};
        let resp = self
            .metrics
            .get_explorer_metrics(req)
            .await
            .map_err(|e| format!("rpc error: {e}"))?
            .into_inner();
        match resp.result {
            Some(nockchain::public::v2::get_explorer_metrics_response::Result::Metrics(m)) => {
                Ok(ResponseData::Line(format!(
                    "{} {} {:.6}",
                    m.cache_height, m.heaviest_height, m.cache_coverage_ratio
                )))
            }
            Some(nockchain::public::v2::get_explorer_metrics_response::Result::Error(e)) => {
                Err(e.message)
            }
            None => Err("empty response".into()),
        }
    }
}
