import os
import json

def evaluate_jsonl_accuracy(input_dir):
    """
    Iterates over all JSONL files in a directory, reads each line,
    and computes the mean accuracy based on 'is_correct' values.
    """
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        # if not os.path.isfile(filepath) or not filename.endswith('.json'):
        #     continue  # Skip non-files or non-JSON files

        is_correct_values = []

        try:
            f = json.load(open(filepath))
        except:
            f = [json.loads(line) for line in open(filepath).readlines()]

        for entry in f:
            try:
                is_correct_values.append(entry['is_correct'])

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line in {filename} due to error: {e}")

        if is_correct_values:
            accuracy = sum(is_correct_values) / len(is_correct_values)
            print(f"File: {filename} | Accuracy: {accuracy:.3f}")
        else:
            print(f"File: {filename} | No valid entries.")

# Example usage
if __name__ == "__main__":
    input_dir = "src/evaluation/evaluation_logs" 
    evaluate_jsonl_accuracy(input_dir)
