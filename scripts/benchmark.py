import time
import threading

def run_query(query):
    start = time.time()
    # 调用API...
    latency = time.time() - start
    print(f"Query: {query[:20]}... | Latency: {latency:.2f}s")

queries = ["混凝土强度", "钢筋焊接", "施工安全"] * 10

threads = []
for q in queries:
    t = threading.Thread(target=run_query, args=(q,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()