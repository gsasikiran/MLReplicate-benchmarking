import json
import os
from datetime import datetime

def load_cost_log(log_file="cost_log.json"):
    """Load the cost log file."""
    if not os.path.exists(log_file):
        print(f"No cost log found at: {log_file}")
        return None
    
    with open(log_file, 'r') as f:
        return json.load(f)

def display_statistics(logs):
    """Display cost statistics from the log."""
    if not logs:
        print("No cost data available.")
        return
    
    total_cost = sum(log['total_cost_usd'] for log in logs)
    total_tokens = sum(log['total_tokens'] for log in logs)
    total_requests = len(logs)
    
    avg_cost = total_cost / total_requests
    avg_tokens = total_tokens / total_requests
    
    print("=" * 70)
    print("COST STATISTICS SUMMARY")
    print("=" * 70)
    print(f"\nTotal API Requests: {total_requests}")
    print(f"Total Tokens Used:  {total_tokens:,}")
    print(f"Total Cost:         ${total_cost:.6f}")
    print(f"\nAverage per Request:")
    print(f"  Tokens: {avg_tokens:,.0f}")
    print(f"  Cost:   ${avg_cost:.6f}")
    print("\n" + "=" * 70)
    print("DETAILED LOG")
    print("=" * 70)
    
    for i, log in enumerate(logs, 1):
        timestamp = datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n#{i} - {timestamp}")
        print(f"  PDF: {log['pdf_file']}")
        print(f"  Title: {log['generated_title']}")
        print(f"  Tokens: {log['total_tokens']:,} ({log['input_tokens']:,} in + {log['output_tokens']:,} out)")
        print(f"  Cost: ${log['total_cost_usd']:.6f}")
    
    print("\n" + "=" * 70)

def main():
    """Main function."""
    logs = load_cost_log()
    
    if logs:
        display_statistics(logs)
    else:
        print("Run pdf_title_generator.py first to generate cost logs.")

if __name__ == "__main__":
    main()