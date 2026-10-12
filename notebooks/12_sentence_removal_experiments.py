"""Server entry file for the Week 3.1 three-sample sentence protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.black_hole.run_sentence_experiments import main

if __name__ == "__main__":
    main()
