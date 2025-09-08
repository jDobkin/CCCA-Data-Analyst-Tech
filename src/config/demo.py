"""
Demo: reading settings and printing a summary.
Run:
    python -m src.config.demo
"""
from .settings import summarize_settings, PATHS, australia_polygon

def main():
    summary = summarize_settings()
    print("=== Settings Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    poly = australia_polygon()
    print(f"Australia polygon bounds: {poly.bounds}")

if __name__ == "__main__":
    main()
