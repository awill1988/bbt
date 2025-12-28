{
  description = "Document ingestion and RAG system";

  nixConfig = {
    cores = 0;  # use all available cores
    max-jobs = "auto";
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        lib = nixpkgs.lib;
        llvm-overlay = final: prev: {
          llvmPackages = prev.llvmPackages_21.override {
            doCheck = false;
            buildDocs = false;
          };
        };
        rustc-overlay = final: prev: {
          rustc = prev.rustc.overrideAttrs (old: {
            doCheck = false;
          });
        };
        overlays = [ rust-overlay.overlays.default llvm-overlay rustc-overlay ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config = {
            allowUnfreePredicate = pkg:
              let
                name = lib.getName pkg;
                hasNvidiaPrefix = lib.hasPrefix "cuda" name
                  || lib.hasPrefix "libcu" name || lib.hasPrefix "libnv" name
                  || lib.hasPrefix "libnpp" name;
              in hasNvidiaPrefix;
          };
        };
        isDarwin = pkgs.stdenv.isDarwin;

        rustToolchain = pkgs.rust-bin.nightly.latest.default.override {
          extensions = [ "rust-src" ];
          targets = [ "aarch64-unknown-linux-gnu" "x86_64-unknown-linux-gnu" ];
        };

        docker-build-script = pkgs.writeShellScriptBin "docker-build" ''
          set -e
          IMAGE_TAG=''${1:-bookmarks:latest}
          echo "building docker image: $IMAGE_TAG"
          ${pkgs.docker}/bin/docker build -t "$IMAGE_TAG" .
          echo "image built and tagged: $IMAGE_TAG"
        '';

        bbt-script = pkgs.writeShellScriptBin "bbt" ''
          set -e
          exec ${rustToolchain}/bin/cargo run --bin bbt -- "$@"
        '';

        nuke-script = pkgs.writeShellScriptBin "nuke" ''
          set -e

          echo "🔥 NUKE: resetting all data..."
          echo ""

          # stop docker compose
          if [ -f docker-compose.yml ]; then
            echo "stopping docker compose services..."
            ${pkgs.docker}/bin/docker compose down -v
            echo "✓ docker services stopped and volumes removed"
          fi

          # clear local data directories
          if [ -d ./data ]; then
            echo "clearing ./data directory..."
            rm -rf ./data/*
            echo "✓ local data cleared"
          fi

          # clear cache (optional, preserves models)
          if [ "$1" = "--clear-cache" ]; then
            echo "clearing ./cache directory..."
            rm -rf ./.cache/*
            echo "✓ cache cleared"
          else
            echo "ℹ  cache preserved (use --clear-cache to remove models)"
          fi

          # recreate directories
          mkdir -p .cache data

          echo ""
          echo "✅ nuke complete! all data reset"
          echo ""
          echo "to restart services:"
          echo "  docker compose up -d"
          echo ""
          echo "to re-index:"
          echo "  bbt sync <paths>"
          echo "  bbt sync --commits <paths>"
        '';

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; ([
            cmake
            pkg-config
            openssl
            zlib
          ] ++ lib.optionals isDarwin [
            libiconv
          ] ++ lib.optionals (!isDarwin) [
            gcc
            cudatoolkit
          ]);

          packages = [
            rustToolchain
            pkgs.rust-analyzer
            pkgs.clippy
            pkgs.docker
            docker-build-script
            bbt-script
            nuke-script
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.openssl.out}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.zlib.dev}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
            export LOG_LEVEL="''${LOG_LEVEL:-info}"
            export BBT_ENABLE_TRACING="''${BBT_ENABLE_TRACING:-true}"
            export BBT_VECTOR_STORE_TYPE="''${BBT_VECTOR_STORE_TYPE:-qdrant}"
            export BBT_QDRANT_URL="''${BBT_QDRANT_URL:-http://localhost:6334}"
            export BBT_STATE_STORE_PATH="''${BBT_STATE_STORE_PATH:-./data/state.db}"
            export BBT_RETRIEVAL_MODE="''${BBT_RETRIEVAL_MODE:-hybrid}"
            export BBT_TOP_K="''${BBT_TOP_K:-5}"
            export BBT_VECTOR_WEIGHT="''${BBT_VECTOR_WEIGHT:-0.5}"
            export BBT_BM25_WEIGHT="''${BBT_BM25_WEIGHT:-0.5}"
            export BBT_MIN_SCORE="''${BBT_MIN_SCORE:-0.5}"
            export BBT_CHUNK_SIZE="''${BBT_CHUNK_SIZE:-512}"
            export BBT_CHUNK_OVERLAP="''${BBT_CHUNK_OVERLAP:-128}"
            export BBT_FORCE_SYNC="''${BBT_FORCE_SYNC:-false}"
            export BBT_RESET_STATE="''${BBT_RESET_STATE:-false}"

            # enable wsl nvidia gpu driver libraries when running under wsl
            if [ -d /usr/lib/wsl/lib ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
            fi

            # ensure required directories exist
            mkdir -p .cache data
          '';
        };

        apps = {
          bbt = {
            type = "app";
            program = "${bbt-script}/bin/bbt";
          };
          docker-build = {
            type = "app";
            program = "${docker-build-script}/bin/docker-build";
          };
          nuke = {
            type = "app";
            program = "${nuke-script}/bin/nuke";
          };
        };
      });
}
