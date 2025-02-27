import csv
import argparse
import os
from collections import defaultdict

def evaluate(answers_path, predictions_path, languages_path, output_path):
    # Dictionary to store per-language statistics
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct = 0
    total_examples = 0
    
    # Open files and process line by line
    with open(answers_path, 'r', encoding='utf-8') as ans_file, \
         open(predictions_path, 'r', encoding='utf-8') as pred_file, \
         open(languages_path, 'r', encoding='utf-8') as lang_file:
        
        for ans, pred, lang in zip(ans_file, pred_file, lang_file):
            ans, pred, lang = ans.strip(), pred.strip(), lang.strip()
            stats[lang]['total'] += 1
            total_examples += 1
            
            if ans in pred:  # Check if the answer is in the prediction (top-3 characters)
                stats[lang]['correct'] += 1
                total_correct += 1
    
    # Compute accuracy and sort results by total number of examples descending
    results = [(lang, data['correct'], data['total'], round(data['correct'] / data['total'], 4)) 
               for lang, data in stats.items()]
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Add the overall results as the first row
    total_accuracy = round(total_correct / total_examples, 4) if total_examples > 0 else 0.0
    results.insert(0, ('all', total_correct, total_examples, total_accuracy))
    
    # Write results to CSV
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'evaluation_results.csv')
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['language', 'correct', 'total', 'accuracy'])
        writer.writerows(results)
    
    print(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against answers.")
    parser.add_argument("answers_path", type=str, help="Path to the answers file")
    parser.add_argument("predictions_path", type=str, help="Path to the predictions file")
    parser.add_argument("languages_path", type=str, help="Path to the languages file")
    parser.add_argument("output_path", type=str, help="Directory to save results")
    
    args = parser.parse_args()
    evaluate(args.answers_path, args.predictions_path, args.languages_path, args.output_path)
