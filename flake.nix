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
        cudaPkg = pkgs.cudaPackages.cudatoolkit.override {cudaVersion = "12.1";};
      in {
        devShells.default = pkgs.mkShell rec {
          packages = with pkgs; [
            pkg-config
            openssl
            glxinfo
            vscode-extensions.llvm-org.lldb-vscode
            taplo
            mdbook
            glib-networking
            cudaPkg
            inputs.nixgl.packages.${pkgs.system}.default
            inputs.nixgl.packages.${pkgs.system}.nixVulkanNvidia
            cudaPackages.cudnn
            typos
            libxkbcommon
            libGL
            wayland
            vulkan-tools
            vulkan-loader
            # flamegraph
            # samply
          ];
          LD_LIBRARY_PATH = "${lib.makeLibraryPath packages}:/run/opengl-driver-32/lib:${pkgs.libGL}/lib:${cudaPkg}/lib:${pkgs.wayland}/lib";
        };
        formatter = pkgs.alejandra;
      };
    };
}
