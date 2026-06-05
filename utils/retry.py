# utils/retry.py

import time
import random
from groq import RateLimitError


def llm_invoke_with_retry(llm, prompt: str, max_attempts: int = 4):
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(prompt)
        except RateLimitError:
            if attempt == max_attempts:
                raise
            sleep = (2 ** attempt) + random.uniform(0, 1)  # 3s, 5s, 9s max
            print(f"[RATE LIMIT] attempt {attempt} — sleeping {sleep:.1f}s")
            time.sleep(sleep)
        except Exception:
            raise
