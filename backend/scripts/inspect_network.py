from pathlib import Path

import tsnet


INP_NAME = "Tnet1.inp"
INP_PATH = Path(__file__).resolve().parents[1] / "data" / "networks" / INP_NAME


def main() -> None:
    print(f"Using INP: {INP_PATH}")
    if not INP_PATH.exists():
        raise FileNotFoundError(INP_PATH)

    tm = tsnet.network.TransientModel(str(INP_PATH))

    print("\nPumps:")
    for pid, _ in tm.pumps():
        print("  ", pid)

    print("\nValves:")
    for vid, _ in tm.valves():
        print("  ", vid)

    print("\nSample junctions (first 20):")
    for i, (nid, _) in enumerate(tm.junctions()):
        if i >= 20:
            break
        print("  ", nid)


if __name__ == "__main__":
    main()
