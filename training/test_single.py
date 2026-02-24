import sys
import json
from inference.inference import get_model

def test_single_transcript(transcript=None):
    """
    Runs inference on a single transcript and prints results.
    """
    if not transcript:
        print("\n--- Interactive Transcript Tester ---")
        print("Enter/Paste your transcript below (Press Ctrl+D or Ctrl+Z to finish):")
        transcript = sys.stdin.read().strip()
    
    if not transcript:
        print("Error: No transcript provided.")
        return

    print("\n[1/2] Loading model and running inference...")
    model = get_model()
    result = model.predict(transcript)
    
    print("\n[2/2] Result Received:")
    print("=" * 50)
    print(json.dumps(result, indent=4))
    print("=" * 50)
    
    # Simple Production Logic Hint
    conf = result.get("confidence_score", 0.0)
    if conf < 0.85:
        print(f"Advice: Low confidence ({conf:.2%}). Hand over to Human.")
    else:
        print(f"Advice: High confidence ({conf:.2%}). Safe to automate.")

if __name__ == "__main__":
    # Can pass transcript as command line arg
    if len(sys.argv) > 1:
        test_single_transcript(" ".join(sys.argv[1:]))
    else:
        test_single_transcript()
