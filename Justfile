set dotenv-load := false

default:
    @just --list

# Install the locked Python development environment.
bootstrap:
    uv sync --frozen

# Build the Rust library and Python distributions.
build: build-rust build-python

build-rust:
    cargo build --all-targets --all-features

build-python:
    uv build

# Run the complete Rust and Python test suites.
test: test-rust test-python

test-rust:
    cargo test --all-targets --all-features

test-python:
    uv run pytest

# Run all static checks without modifying source files.
lint: lint-rust lint-python lint-workflows

lint-rust:
    cargo clippy --all-targets --all-features -- -D warnings
    cargo fmt --all -- --check

lint-python:
    uv run ruff check .
    uv run ruff format --check .
    uv run ty check

lint-workflows:
    uv run yamloom check --file .yamloom.py

# Format Rust and Python source files.
fmt:
    cargo fmt --all
    uv run ruff format .
    uv run ruff check --fix .

# Run the Python example.
examples:
    MPLBACKEND=Agg uv run python examples/bump.py

# Remove generated build and test artifacts.
clean:
    cargo clean
    rm -rf build dist .pytest_cache .ruff_cache

sync-actions:
    yamloom sync
