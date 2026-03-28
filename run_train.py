"""Wrapper to run training with error handling."""
import sys
import traceback

if __name__ == '__main__':
    try:
        print("Starting training wrapper...")
        print(f"Python version: {sys.version}")
        print("Importing train module...")
        
        # Import and run the main function
        from train import main
        print("Running main...")
        main()
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)
