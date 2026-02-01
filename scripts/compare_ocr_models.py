# scripts/compare_ocr_models.py
import os
import sys
import argparse
import logging

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.comparison_engine import OCRComparisonEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('OCR-Comparison')


def print_comparison_results(results: dict) -> None:
    print("\n" + "-" * 10)
    print("VIETNAMESE OCR COMPARISON RESULTS")
    print(f"Image: {results['image_path']}")
    
    if results.get('ground_truth'):
        print(f"\nGround Truth: {results['ground_truth']}")
    
    print("\n" + "-" * 80)
    print(f"{'Model':<25} {'Prediction':<35} {'CER':>8} {'Time':>10}")
    print("-" * 80)
    
    for model_name in results.get('predictions', {}):
        pred = results['predictions'].get(model_name, '')
        timing = results['timing'].get(model_name, 0)
        metrics = results.get('metrics', {}).get(model_name, {})
        cer = metrics.get('cer', None)
        
        #truncate prediction for display
        pred_display = pred[:30] + '...' if len(pred) > 30 else pred
        cer_str = f"{cer*100:.1f}%" if cer is not None else "N/A"
        
        print(f"{model_name:<25} {pred_display:<35} {cer_str:>8} {timing*1000:>8.1f}ms")
    
    print("-" * 10)


def print_aggregate_results(aggregate_metrics: dict) -> None:
    """
    Print aggregate results across multiple images.
    
    Args:
        aggregate_metrics: Aggregate metrics dictionary
    """
    print("\n" + "-" * 10)
    print("AGGREGATE RESULTS")
    
    for model_name, metrics in sorted(aggregate_metrics.items(), key=lambda x: x[1].get('avg_cer', 1)):
        print(f"\n{model_name}:")
        print(f"  CER:  {metrics['avg_cer']*100:.2f}%")
        print(f"  WER:  {metrics['avg_wer']*100:.2f}%")
        print(f"  Acc:  {metrics['avg_accuracy']*100:.2f}%")
        print(f"  Time: {metrics['avg_time']*1000:.1f}ms")


def save_results_to_csv(all_results: list, csv_path: str) -> None:
    rows = []
    for r in all_results:
        row = {
            'image': r['image_path'],
            'ground_truth': r.get('ground_truth', '')
        }
        for model_name in r.get('predictions', {}):
            row[f'{model_name}_pred'] = r['predictions'].get(model_name, '')
            if model_name in r.get('metrics', {}):
                row[f'{model_name}_cer'] = r['metrics'][model_name].get('cer', None)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nResults saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare Vietnamese OCR Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Single image comparison
    python scripts/compare_ocr_models.py --image ./test.png
    
    # Folder comparison with ground truth
    python scripts/compare_ocr_models.py --folder ./data/test --ground_truth ./labels.tsv
    
    # Select specific models
    python scripts/compare_ocr_models.py --image ./test.png --models our,vietocr
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Single image to test')
    input_group.add_argument('--folder', type=str, help='Folder of images to test')
    
    parser.add_argument('--model_path', type=str, 
                        default='./ckpt/best_model_hf',
                        help='Path to our custom model')
    parser.add_argument('--vocab_path', type=str,
                        default='./ckpt/best_model_hf/vocabularies/combined_char_vocab.json',
                        help='Path to vocabulary')
    parser.add_argument('--ground_truth', type=str, default=None,
                        help='Ground truth text (for single image) or file (for folder)')
    parser.add_argument('--models', type=str, default='our,vietocr,tesseract,easyocr',
                        help='Comma-separated list of models: our,vietocr,vietocr_transformer,tesseract,easyocr')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for report (HTML)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Output path for CSV results')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    models_to_use = [m.strip() for m in args.models.split(',')]
    
    engine = OCRComparisonEngine(
        our_model_path=args.model_path,
        our_vocab_path=args.vocab_path,
        device=args.device,
        models_to_use=models_to_use
    )
    
    print("\nLoading OCR models...")
    load_results = engine.load_all_models()
    
    loaded_count = sum(1 for v in load_results.values() if v)
    print(f"\nLoaded {loaded_count}/{len(load_results)} models\n")
    
    if loaded_count == 0:
        print("No models were loaded successfully. Exiting.")
        return 1
    
    if args.image:
        results = engine.compare_single_image(args.image, args.ground_truth)
        print_comparison_results(results)
    
    elif args.folder:
        all_results, aggregate_metrics = engine.compare_folder(
            args.folder,
            args.ground_truth
        )
        
        print_aggregate_results(aggregate_metrics)
        
        if args.csv:
            save_results_to_csv(all_results, args.csv)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
