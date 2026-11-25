"""Phase 5: compare two drivers' fastest laps, plus track maps.

Needs YOUR OWN F1 TV token:  export PITWALL_F1TV_TOKEN='...'
Run:  python examples/telemetry_demo.py 2024 Monza Q NOR PIA
"""
import logging
import sys

import matplotlib.pyplot as plt

import pitwall
from pitwall.plotting import delta_plot, gear_map, speed_map, speed_trace, use_pitwall_style


def main(year: int, event: int | str, kind: str, a: str, b: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # URLs only, never the token
    use_pitwall_style()
    print(pitwall.token_status())

    session = pitwall.get_session(year, event, kind).load()
    pitwall.load_telemetry(session, drivers=[a, b])  # big feeds: only these two cars
    for driver in (a, b):
        lap = pitwall.timing_lap(session, driver)
        print(f"{driver}: lap {lap['LapNumber']} in {lap['LapTimeSeconds']:.3f} s")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    speed_trace(session, [a, b], ax=top)
    delta_plot(session, a, b, ax=bottom)
    tag = f"{year}_r{session.round}_{kind}"
    for name, figure in [("speed_delta", fig),
                         ("gears", gear_map(session, a).figure),
                         ("speed_map", speed_map(session, a).figure)]:
        figure.tight_layout()
        figure.savefig(f"{name}_{tag}.png", dpi=150)
        print(f"Saved {name}_{tag}.png")
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    year = int(args[0]) if args else 2024
    event = args[1] if len(args) > 1 else "Monza"
    event = int(event) if str(event).isdigit() else event
    kind = args[2] if len(args) > 2 else "Q"
    a = args[3] if len(args) > 3 else "NOR"
    b = args[4] if len(args) > 4 else "PIA"
    main(year, event, kind, a, b)
