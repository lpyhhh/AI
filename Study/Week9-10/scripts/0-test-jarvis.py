import argparse
from jarvis import analyze_cress_virus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Jarvis's CRESS Virus Tool")
    parser.add_argument("--seq", type=str, required=True, help="Amino acid sequence")
    args = parser.parse_args()
    
    print("\n--- Jarvis Tool Test ---")
    
    # 【关键修复点】：使用 .run() 方法调用 LangChain 工具
    try:
        output = analyze_cress_virus.run(args.seq) 
        print(output)
    except Exception as e:
        print(f"Call failed: {e}")