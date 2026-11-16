#!/bin/sh
# Builds public/waves.wasm (the response tables of waves.js). Needs the wasm32-unknown-unknown
# target (Arch: pacman -S rust-wasm; rustup: rustup target add wasm32-unknown-unknown).
set -e
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/waves_response.wasm ../public/waves.wasm
ls -l ../public/waves.wasm
