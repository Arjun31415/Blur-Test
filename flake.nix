{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    nixgl = {
      url = "github:nix-community/nixGL";
      # inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {nixpkgs, ...} @ inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {
        config,
        system,
        lib,
        ...
      }: let
        pkgs = import nixpkgs {
          inherit system;
        };
        # cudaPkg = pkgs.cudaPackages.cudatoolkit.override {cudaVersion = "12.2";};
        cudaPackageSet = pkgs.cudaPackages.override {cudaVersion = "12.2";};
        cudaPackages = [
          cudaPackageSet.cudnn
          cudaPackageSet.libcublas
          cudaPackageSet.cuda_nvcc
          cudaPackageSet.cuda_cudart
          cudaPackageSet.cuda_profiler_api
        ];
        opencv4cuda = pkgs.opencv4.override {
          enableCuda = true;
          enableGtk3 = true;
          # enableUnfree = true;
        };
        mkShell =
          pkgs
          .mkShell
          .override
          {stdenv = cudaPackageSet.backendStdenv;};
        stdenv = cudaPackageSet.backendStdenv;
      in {
        devShells.default = mkShell {
          inputsFrom = [config.packages.blur_test];

          packages = with pkgs; [
            inputs.nixgl.packages.${pkgs.system}.default
            inputs.nixgl.packages.${pkgs.system}.nixVulkanNvidia
            libGL
            wayland
            vulkan-tools
            vulkan-loader
            bear
            # flamegraph
            # samply
            valgrind
          ];
          shellHook = ''
            export CUDA_DIR=${lib.makeLibraryPath cudaPackages};
          '';
        };
        formatter = pkgs.alejandra;
        packages = rec {
          blur_test = with pkgs;
            stdenv
            .mkDerivation {
              name = "cvCuda_test";
              src = ./.;

              buildInputs =
                [opencv4cuda valgrind] ++ cudaPackages;
              nativeBuildInputs = with pkgs; [pkg-config libconfig cmake];
            };
          default = blur_test;
        };
      };
    };
}
