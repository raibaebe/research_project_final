"""CLI entry point for Sign Language Recognition."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_train(args):
    from src.train import train
    train(
        data_dir=args.data_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation,
    )


def cmd_evaluate(args):
    from src.evaluate import evaluate
    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        save_plots=not args.no_plots,
    )


def cmd_predict(args):
    from src.predict import run_webcam_prediction
    run_webcam_prediction(
        model_path=args.model_path,
        camera_index=args.camera,
        smoothing_window=args.smoothing,
    )


def cmd_predict_image(args):
    from src.predict import predict_single_image
    letter, confidence = predict_single_image(args.model_path, args.image)
    print(f"Predicted: {letter}  (confidence: {confidence:.1%})")


def cmd_collect(args):
    from src.collect import run_collection
    run_collection(
        output_dir=args.output_dir,
        camera_index=args.camera,
        frames_per_capture=args.frames,
    )


def main():
    parser = argparse.ArgumentParser(description="Sign Language Gesture Recognition")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_train = subparsers.add_parser("train", help="Train the CNN model")
    p_train.add_argument("--data-dir", default="data", help="Dataset directory (default: data)")
    p_train.add_argument("--model-path", default="models/trained_model.h5", help="Output model path")
    p_train.add_argument("--epochs", type=int, default=30, help="Max epochs (default: 30)")
    p_train.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    p_train.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained model")
    p_eval.add_argument("--data-dir", default="data", help="Dataset directory")
    p_eval.add_argument("--model-path", default="models/trained_model.h5", help="Model to evaluate")
    p_eval.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = subparsers.add_parser("predict", help="Real-time webcam prediction")
    p_pred.add_argument("--model-path", default="models/trained_model.h5", help="Model to use")
    p_pred.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p_pred.add_argument("--smoothing", type=int, default=7, help="Smoothing window size (default: 7)")
    p_pred.set_defaults(func=cmd_predict)

    p_img = subparsers.add_parser("predict-image", help="Predict from a single image")
    p_img.add_argument("image", help="Path to image file")
    p_img.add_argument("--model-path", default="models/trained_model.h5", help="Model to use")
    p_img.set_defaults(func=cmd_predict_image)

    p_coll = subparsers.add_parser("collect", help="Collect training data from webcam")
    p_coll.add_argument("--output-dir", default="data/collected", help="Output directory")
    p_coll.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p_coll.add_argument("--frames", type=int, default=5, help="Frames per key press (default: 5)")
    p_coll.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
