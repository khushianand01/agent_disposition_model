from inference.inference import get_model
import json

def production_pipeline(transcript, confidence_threshold=0.85):
    """
    Simulates a production workflow with a human-in-the-loop fallback.
    """
    model = get_model()
    
    # 1. Run model inference
    result = model.predict(transcript)
    
    # 2. Extract confidence
    confidence = result.get("confidence_score", 0.0)
    
    print(f"\n--- Production Processing ---")
    print(f"Confidence Score: {confidence:.2%}")
    print("\nModel Prediction (Full JSON):")
    print(json.dumps(result, indent=4))
    
    # 3. Decision Logic
    if confidence < confidence_threshold:
        print("\n⚠️ [RESULT]: LOW CONFIDENCE")
        print("ACTION: Routing to Human Review Queue.")
        return {
            "status": "human_review_required",
            "model_prediction": result,
            "reason": f"Confidence {confidence:.2%} is below threshold {confidence_threshold:.2%}"
        }
    else:
        print("\n✅ [RESULT]: HIGH CONFIDENCE")
        print("ACTION: Automating Disposition Update.")
        return {
            "status": "automated",
            "model_prediction": result
        }

if __name__ == "__main__":
    sample_transcript = """Agent: Hello, am I speaking with Rahul Sharma?
Customer: Yes, speaking.
Agent: This is regarding your overdue loan payment of ₹12,500.
Customer: Okay, I’m aware. I’ll check and get back.
"""
    final_output = production_pipeline(sample_transcript, confidence_threshold=0.85)
    
    print("\n--- Final System Output ---")
    print(json.dumps(final_output, indent=4))
