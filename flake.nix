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
                  || lib.hasPrefix "libnpp" name
                  || name == "cudnn";
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

        bbt-script = pkgs.writeShellScriptBin "bbt" (''
          set -e
        '' + lib.optionalString (!isDarwin) ''
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:${pkgs.cudaPackages.libcublas}/lib:${pkgs.cudaPackages.libcufft}/lib:${pkgs.cudaPackages.cudnn}/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
        '' + ''
          exec ${rustToolchain}/bin/cargo run --bin bbt --features ${if isDarwin then "coreml" else "cuda"} -- "$@"
        '');

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
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart
            cudaPackages.libcublas
            cudaPackages.libcufft
            cudaPackages.cudnn
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
            # avoid leaking nix libs into host binaries like /bin/ssh
            export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.zlib.dev}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
            export LOG_LEVEL="''${LOG_LEVEL:-info}"
            export ENABLE_TRACING="''${ENABLE_TRACING:-true}"
            export VECTOR_STORE_TYPE="''${VECTOR_STORE_TYPE:-qdrant}"
            export QDRANT_URL="''${QDRANT_URL:-http://localhost:6334}"
            export STATE_STORE_PATH="''${STATE_STORE_PATH:-./data/state.db}"
            export RETRIEVAL_MODE="''${RETRIEVAL_MODE:-hybrid}"
            export TOP_K="''${TOP_K:-5}"
            export VECTOR_WEIGHT="''${VECTOR_WEIGHT:-0.5}"
            export BM25_WEIGHT="''${BM25_WEIGHT:-0.5}"
            export MIN_SCORE="''${MIN_SCORE:-0.5}"
            export CHUNK_SIZE="''${CHUNK_SIZE:-512}"
            export CHUNK_OVERLAP="''${CHUNK_OVERLAP:-128}"
            export SYNC_FORCE="''${SYNC_FORCE:-false}"
            export SYNC_RESET_STATE="''${SYNC_RESET_STATE:-false}"

            # enable wsl nvidia gpu driver libraries when running under wsl
            if [ -d /usr/lib/wsl/lib ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
            fi
          '' + lib.optionalString (!isDarwin) ''
            # add cuda, cudnn, and c++ stdlib to path for onnxruntime
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:${pkgs.cudaPackages.libcublas}/lib:${pkgs.cudaPackages.libcufft}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH"
          '' + ''
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
