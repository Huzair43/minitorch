from __future__ import annotations

from . import show_config


def main() -> None:
    config = show_config()
    print("MiniTorch Python")
    print(f"version: {config['version']}")
    if config["library_found"]:
        print(f"librairie C trouvée: {config['library_found']}")
    else:
        print("librairie C non trouvée")
        print("compile d'abord la cible partagée, puis définis MINITORCH_LIB si besoin")
    print("chemins testés:")
    for candidate in config["library_candidates"]:
        print(f"  - {candidate}")


if __name__ == "__main__":
    main()
