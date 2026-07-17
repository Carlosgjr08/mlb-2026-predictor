"""Command-line entry point: sample / fetch / train / predict / simulate / track."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlb_predictor",
        description="XGBoost run/score predictions for the 2026 MLB season.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sample", help="generate the offline sample dataset")

    fetch = sub.add_parser("fetch", help="fetch real data (MLB Stats API, needs network)")
    fetch.add_argument("--seasons", type=int, nargs="+",
                       default=[2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    fetch.add_argument("--include-live", action="store_true",
                       help="keep in-progress games as predictable fixtures "
                            "(the model only uses pre-game info)")

    train = sub.add_parser("train", help="train + evaluate the XGBoost models")
    train.add_argument("--test-season", type=int, default=2026)

    predict = sub.add_parser("predict", help="predict games")
    predict.add_argument("--home")
    predict.add_argument("--away")
    predict.add_argument("--upcoming", action="store_true")
    predict.add_argument("--limit", type=int, default=None,
                     help="max games printed (default 10, or all with --date)")
    predict.add_argument("--date", help="only games on this date (YYYY-MM-DD), "
                                        "with --upcoming")

    simulate = sub.add_parser("simulate", help="Monte Carlo the rest of the season")
    simulate.add_argument("--runs", type=int, default=5000)

    track = sub.add_parser("track", help="track predictions vs. real results")
    track_sub = track.add_subparsers(dest="track_command", required=True)
    track_sub.add_parser("record", help="log predictions for upcoming games")
    result = track_sub.add_parser("result", help="enter a final score")
    result.add_argument("--home", required=True)
    result.add_argument("--away", required=True)
    result.add_argument("--score", required=True, help="final score as home-away, e.g. 5-3")
    track_sub.add_parser("board", help="refresh the MLB_RESULTS.md scoreboard")

    args = parser.parse_args()

    if args.command == "sample":
        from .sample_data import main as run
        run()
    elif args.command == "fetch":
        from .fetch import main as run
        run(args.seasons, args.include_live)
    elif args.command == "train":
        from .train import main as run
        run(args.test_season)
    elif args.command == "predict":
        from .predict import predict_matchup, predict_upcoming
        from .train import load_bundle
        bundle = load_bundle()
        if args.upcoming:
            limit = args.limit if args.limit is not None else (None if args.date else 10)
            predict_upcoming(bundle, limit, args.date)
        elif args.home and args.away:
            predict_matchup(bundle, args.home, args.away)
        else:
            raise SystemExit("Use --home X --away Y, or --upcoming.")
    elif args.command == "simulate":
        from .simulate import main as run
        from .train import load_bundle
        run(load_bundle(), args.runs)
    elif args.command == "track":
        from . import track as tracker
        if args.track_command == "record":
            from .train import load_bundle
            tracker.record(load_bundle())
        elif args.track_command == "result":
            tracker.enter_result(args.home, args.away, args.score)
        elif args.track_command == "board":
            tracker.board()


if __name__ == "__main__":
    main()
