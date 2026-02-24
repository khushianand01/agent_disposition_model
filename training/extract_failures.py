import json

test_file = '../data/splits/test_v6.json'
eval_file = 'eval_results_v6.json'
output_boost_file = '../data/failed_cases_boost.json'

def extract_failures():
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        test_data = json.load(f)
        
    print(f"Loading evaluation results from {eval_file}...")
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)
        
    failures = []
    # Key 'results' in eval_results_v6.json contains the list of {index, prediction, ground_truth}
    for item in eval_data.get('results', []):
        idx = item['index']
        pred_obj = item.get('prediction')
        pred = pred_obj.get('disposition') if pred_obj else "JSON_ERROR"
        gt = item['ground_truth']['disposition']
        
        if pred != gt:
            # Get the original sample using the index
            if idx < len(test_data):
                failures.append(test_data[idx])
                
    print(f"Extracted {len(failures)} failed cases.")
    
    with open(output_boost_file, 'w') as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    
    print(f"Saved boost samples to {output_boost_file}")

if __name__ == "__main__":
    extract_failures()
