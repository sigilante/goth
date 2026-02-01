fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)
        .compile_protos(
            &[
                "proto/nockchain/common/v1/primitives.proto",
                "proto/nockchain/common/v1/blockchain.proto",
                "proto/nockchain/common/v1/pagination.proto",
                "proto/nockchain/common/v2/blockchain.proto",
                "proto/nockchain/public/v2/nockchain.proto",
            ],
            &["proto"],
        )?;
    Ok(())
}
