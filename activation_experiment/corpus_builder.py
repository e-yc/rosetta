#!/usr/bin/env python3
"""
Activation Differential Experiment — Corpus Builder (Part 1)

Builds a 10,000-input corpus from HuggingFace datasets, tokenizes with both
models' tokenizers, computes alignment, and saves corpus.jsonl.

No GPU required.
"""

import json
import os
import random
import sys
import textwrap

random.seed(42)

import config
from tokenizer_translation import align_tokens, classify_alignment, get_activation_pairs


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_id, fallbacks, name):
    """Load tokenizer, trying fallbacks if primary is gated."""
    from transformers import AutoTokenizer
    token = os.environ.get("HF_TOKEN") or True
    for mid in [model_id] + fallbacks:
        try:
            print(f"  Loading {name} tokenizer from {mid} ...", end=" ", flush=True)
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, token=token)
            print("OK")
            return tok
        except Exception as e:
            err = str(e)[:100]
            print(f"FAILED ({err})")
    print(f"  *** Could not load tokenizer for {name} ***")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset loaders (with fallbacks)
# ---------------------------------------------------------------------------

def load_english_web(count):
    """Load english web paragraphs from wikipedia."""
    texts = []
    try:
        print("  Loading wikimedia/wikipedia (en) ...", flush=True)
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        for item in ds:
            text = item.get("text", "")
            # Split into paragraphs, take non-trivial ones
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 100 and not para.startswith(("==", "{{", "|", "*", "#")):
                    texts.append(para)
                    if len(texts) >= count:
                        break
            if len(texts) >= count:
                break
    except Exception as e:
        print(f"    Primary failed ({e}), trying wikitext fallback...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            for item in ds:
                text = item.get("text", "").strip()
                if len(text) > 100 and not text.startswith(("=", " =")):
                    texts.append(text)
                    if len(texts) >= count:
                        break
        except Exception as e2:
            print(f"    Wikitext also failed ({e2}), using hardcoded fallback")
            texts = _english_web_fallback(count)

    random.shuffle(texts)
    return texts[:count]


def load_code(count):
    """Load code snippets from HuggingFace datasets."""
    texts = []
    try:
        print("  Loading bigcode/starcoderdata ...", flush=True)
        from datasets import load_dataset
        ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True,
                          data_files={"train": ["python/*.parquet", "javascript/*.parquet"]})
        for item in ds:
            text = item.get("content", "")
            # Take first ~500 chars of each file
            snippet = text[:500].strip()
            if len(snippet) > 80:
                texts.append(snippet)
                if len(texts) >= count:
                    break
    except Exception:
        try:
            print("    Trying codeparrot/github-code ...", flush=True)
            from datasets import load_dataset
            ds = load_dataset("codeparrot/github-code", split="train", streaming=True,
                              languages=["Python", "JavaScript", "SQL", "TypeScript"])
            for item in ds:
                text = item.get("code", "")
                snippet = text[:500].strip()
                if len(snippet) > 80:
                    texts.append(snippet)
                    if len(texts) >= count:
                        break
        except Exception:
            print("    Using hardcoded code fallback")
            texts = _code_fallback(count)

    random.shuffle(texts)
    return texts[:count]


def load_conversational(count):
    """Load conversational text."""
    texts = []
    try:
        print("  Loading tatsu-lab/alpaca ...", flush=True)
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for item in ds:
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            output = item.get("output", "")
            # Combine instruction + input + output as a conversation turn
            combined = instruction
            if inp:
                combined += f"\n{inp}"
            combined += f"\n{output}"
            combined = combined.strip()
            if len(combined) > 60:
                texts.append(combined)
                if len(texts) >= count:
                    break
    except Exception as e:
        print(f"    Alpaca failed ({e}), using fallback")
        texts = _conversational_fallback(count)

    random.shuffle(texts)
    return texts[:count]


def load_math_reasoning(count):
    """Load math/reasoning problems."""
    texts = []
    try:
        print("  Loading gsm8k ...", flush=True)
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        for item in ds:
            q = item.get("question", "")
            a = item.get("answer", "")
            combined = f"Question: {q}\nAnswer: {a}"
            if len(combined) > 80:
                texts.append(combined)
                if len(texts) >= count:
                    break
    except Exception as e:
        print(f"    gsm8k failed ({e}), using fallback")

    if len(texts) < count:
        try:
            print("  Loading hendrycks/competition_math ...", flush=True)
            from datasets import load_dataset
            ds = load_dataset("hendrycks/competition_math", split="train")
            for item in ds:
                problem = item.get("problem", "")
                solution = item.get("solution", "")
                combined = f"Problem: {problem}\nSolution: {solution}"
                if len(combined) > 80:
                    texts.append(combined)
                    if len(texts) >= count:
                        break
        except Exception:
            pass

    if len(texts) < count:
        texts.extend(_math_fallback(count - len(texts)))

    random.shuffle(texts)
    return texts[:count]


def load_multilingual(count):
    """Load multilingual text (de, fr, es) from wikipedia."""
    texts = []
    per_lang = count // 3
    for lang, lang_code in [("German", "de"), ("French", "fr"), ("Spanish", "es")]:
        try:
            print(f"  Loading wikimedia/wikipedia ({lang_code}) ...", flush=True)
            from datasets import load_dataset
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang_code}",
                              split="train", streaming=True)
            lang_texts = []
            for item in ds:
                text = item.get("text", "")
                for para in text.split("\n\n"):
                    para = para.strip()
                    if len(para) > 100 and not para.startswith(("==", "{{", "|")):
                        lang_texts.append(para)
                        if len(lang_texts) >= per_lang:
                            break
                if len(lang_texts) >= per_lang:
                    break
            texts.extend(lang_texts[:per_lang])
        except Exception as e:
            print(f"    {lang} wikipedia failed ({e})")

    if len(texts) < count:
        texts.extend(_multilingual_fallback(count - len(texts)))

    random.shuffle(texts)
    return texts[:count]


def load_mixed_edge(count):
    """Generate synthetic mixed/edge-case inputs."""
    return _mixed_edge_cases(count)


# ---------------------------------------------------------------------------
# Hardcoded fallbacks
# ---------------------------------------------------------------------------

def _english_web_fallback(count):
    """Hardcoded English paragraphs as last-resort fallback."""
    base = [
        "The theory of plate tectonics describes the large-scale motion of seven large plates and the movements of a larger number of smaller plates of Earth's lithosphere. The model builds on the concept of continental drift, an idea developed during the first decades of the 20th century.",
        "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. Some of this chemical energy is stored in carbohydrate molecules.",
        "The Internet protocol suite, commonly known as TCP/IP, is the set of communication protocols used in the Internet and similar computer networks. The current foundational protocols in the suite are the Transmission Control Protocol and the Internet Protocol.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
        "The human brain is the central organ of the human nervous system, and with the spinal cord makes up the central nervous system. The brain consists of the cerebrum, the brainstem and the cerebellum. It controls most of the activities of the body.",
        "Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns. Though there have been previous periods of climatic change, since the mid-20th century humans have had an unprecedented impact.",
        "Quantum computing is a type of computation whose operations can harness the phenomena of quantum mechanics, such as superposition, interference, and entanglement. Devices that perform quantum computations are known as quantum computers.",
        "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. It occurred after the Crisis of the Late Middle Ages and was associated with great social change.",
        "DNA, or deoxyribonucleic acid, is a molecule composed of two polynucleotide chains that coil around each other to form a double helix. The molecule carries genetic instructions for the development, functioning, growth and reproduction of all known organisms.",
        "The global supply chain refers to the network created among different worldwide companies producing, handling, and distributing specific products or services. It encompasses every step from raw material sourcing to final delivery to the consumer.",
        "Blockchain technology is a decentralized, distributed ledger that records transactions across many computers. This ensures that any involved record cannot be altered retroactively, without the alteration of all subsequent blocks.",
        "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow in clouds, and falls again to the surface as precipitation.",
        "Artificial neural networks are computing systems inspired by the biological neural networks that constitute animal brains. These systems learn to perform tasks by considering examples, generally without being programmed with task-specific rules.",
        "The solar system consists of the Sun and the objects that orbit it, whether they orbit directly or by orbiting other objects. Of those that orbit the Sun directly, the largest are the eight planets, with the remainder being smaller objects.",
        "Cryptography is the practice and study of techniques for secure communication in the presence of adversarial behavior. More generally, cryptography is about constructing and analyzing protocols that prevent third parties from reading private messages.",
    ]
    # Repeat and vary to reach count
    result = []
    while len(result) < count:
        result.extend(base)
    return result[:count]


def _code_fallback(count):
    """Hardcoded code snippets."""
    base = [
        'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)',
        'class BinarySearchTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n    \n    def insert(self, val):\n        if val < self.value:\n            if self.left is None:\n                self.left = BinarySearchTree(val)\n            else:\n                self.left.insert(val)\n        else:\n            if self.right is None:\n                self.right = BinarySearchTree(val)\n            else:\n                self.right.insert(val)',
        'SELECT u.username, COUNT(o.id) as order_count, SUM(o.total) as total_spent\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at >= NOW() - INTERVAL \'30 days\'\nGROUP BY u.username\nHAVING COUNT(o.id) > 0\nORDER BY total_spent DESC\nLIMIT 100;',
        'async function fetchWithRetry(url, options = {}, maxRetries = 3) {\n  for (let attempt = 0; attempt < maxRetries; attempt++) {\n    try {\n      const response = await fetch(url, options);\n      if (!response.ok) throw new Error(`HTTP ${response.status}`);\n      return await response.json();\n    } catch (error) {\n      if (attempt === maxRetries - 1) throw error;\n      await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt)));\n    }\n  }\n}',
        'import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)\n        self.ff = nn.Sequential(\n            nn.Linear(d_model, d_ff),\n            nn.GELU(),\n            nn.Linear(d_ff, d_model),\n        )\n        self.norm1 = nn.LayerNorm(d_model)\n        self.norm2 = nn.LayerNorm(d_model)\n        self.dropout = nn.Dropout(dropout)',
        'from typing import TypeVar, Generic, Optional\nfrom dataclasses import dataclass\n\nT = TypeVar("T")\n\n@dataclass\nclass Result(Generic[T]):\n    value: Optional[T] = None\n    error: Optional[str] = None\n    \n    @property\n    def is_ok(self) -> bool:\n        return self.error is None\n    \n    @classmethod\n    def ok(cls, value: T) -> "Result[T]":\n        return cls(value=value)\n    \n    @classmethod\n    def err(cls, message: str) -> "Result[T]":\n        return cls(error=message)',
        'CREATE TABLE IF NOT EXISTS events (\n    id BIGSERIAL PRIMARY KEY,\n    user_id UUID NOT NULL REFERENCES users(id),\n    event_type VARCHAR(50) NOT NULL,\n    payload JSONB DEFAULT \'{}\',\n    created_at TIMESTAMPTZ DEFAULT NOW(),\n    processed_at TIMESTAMPTZ\n);\n\nCREATE INDEX CONCURRENTLY idx_events_user_type\n    ON events (user_id, event_type, created_at DESC);',
        'function debounce<T extends (...args: any[]) => any>(\n  fn: T,\n  delay: number\n): (...args: Parameters<T>) => void {\n  let timeoutId: ReturnType<typeof setTimeout>;\n  return function(this: any, ...args: Parameters<T>) {\n    clearTimeout(timeoutId);\n    timeoutId = setTimeout(() => fn.apply(this, args), delay);\n  };\n}',
        'import asyncio\nfrom aiohttp import ClientSession\n\nasync def fetch_all(urls: list[str]) -> list[dict]:\n    async with ClientSession() as session:\n        tasks = [fetch_one(session, url) for url in urls]\n        return await asyncio.gather(*tasks, return_exceptions=True)\n\nasync def fetch_one(session: ClientSession, url: str) -> dict:\n    async with session.get(url) as response:\n        response.raise_for_status()\n        return await response.json()',
        'WITH RECURSIVE tree AS (\n    SELECT id, name, parent_id, 0 AS depth, ARRAY[id] AS path\n    FROM categories\n    WHERE parent_id IS NULL\n    UNION ALL\n    SELECT c.id, c.name, c.parent_id, t.depth + 1, t.path || c.id\n    FROM categories c\n    JOIN tree t ON c.parent_id = t.id\n    WHERE NOT c.id = ANY(t.path)\n)\nSELECT * FROM tree ORDER BY path;',
        'def lru_cache(maxsize=128):\n    def decorator(func):\n        cache = {}\n        order = []\n        def wrapper(*args):\n            key = args\n            if key in cache:\n                order.remove(key)\n                order.append(key)\n                return cache[key]\n            result = func(*args)\n            cache[key] = result\n            order.append(key)\n            if len(order) > maxsize:\n                oldest = order.pop(0)\n                del cache[oldest]\n            return result\n        return wrapper\n    return decorator',
        'const express = require("express");\nconst rateLimit = require("express-rate-limit");\n\nconst app = express();\nconst limiter = rateLimit({\n  windowMs: 15 * 60 * 1000,\n  max: 100,\n  standardHeaders: true,\n  legacyHeaders: false,\n});\n\napp.use(limiter);\napp.use(express.json());\n\napp.get("/api/health", (req, res) => {\n  res.json({ status: "ok", timestamp: new Date().toISOString() });\n});',
        'import numpy as np\nfrom scipy.optimize import minimize\n\ndef fit_gaussian_mixture(data, n_components=3, max_iter=100):\n    n, d = data.shape\n    weights = np.ones(n_components) / n_components\n    means = data[np.random.choice(n, n_components, replace=False)]\n    covs = [np.eye(d) for _ in range(n_components)]\n    \n    for iteration in range(max_iter):\n        responsibilities = np.zeros((n, n_components))\n        for k in range(n_components):\n            responsibilities[:, k] = weights[k] * multivariate_normal_pdf(data, means[k], covs[k])',
        'interface Repository<T> {\n  findById(id: string): Promise<T | null>;\n  findAll(filter?: Partial<T>): Promise<T[]>;\n  create(entity: Omit<T, "id">): Promise<T>;\n  update(id: string, updates: Partial<T>): Promise<T>;\n  delete(id: string): Promise<boolean>;\n}\n\nclass UserRepository implements Repository<User> {\n  constructor(private db: Database) {}\n  \n  async findById(id: string): Promise<User | null> {\n    const row = await this.db.query("SELECT * FROM users WHERE id = $1", [id]);\n    return row ? this.mapToUser(row) : null;\n  }\n}',
        'package main\n\nimport (\n\t"context"\n\t"fmt"\n\t"sync"\n\t"time"\n)\n\nfunc fanOut(ctx context.Context, input <-chan int, workers int) []<-chan int {\n\tchannels := make([]<-chan int, workers)\n\tfor i := 0; i < workers; i++ {\n\t\tchannels[i] = process(ctx, input)\n\t}\n\treturn channels\n}\n\nfunc process(ctx context.Context, input <-chan int) <-chan int {\n\tout := make(chan int)\n\tgo func() {\n\t\tdefer close(out)\n\t\tfor n := range input {\n\t\t\tselect {\n\t\t\tcase out <- n * n:\n\t\t\tcase <-ctx.Done():\n\t\t\t\treturn\n\t\t\t}\n\t\t}\n\t}()\n\treturn out\n}',
    ]
    result = []
    while len(result) < count:
        result.extend(base)
    return result[:count]


def _conversational_fallback(count):
    """Hardcoded conversational text."""
    base = [
        "User: What is the best way to learn programming?\nAssistant: Start with Python — it has clean syntax and a huge ecosystem. Pick a project you care about and build it. You'll learn more from struggling through a real project than from following tutorials.",
        "User: Can you explain how neural networks work in simple terms?\nAssistant: Think of it like a chain of simple decisions. Each neuron takes numbers in, multiplies them by weights, adds them up, and passes the result through a function. Stack thousands of these together in layers, and the network can learn complex patterns.",
        "User: I'm getting a segfault in my C code, how do I debug it?\nAssistant: Run it under valgrind first: `valgrind ./your_program`. It'll tell you exactly where the invalid memory access happens. Common causes: dereferencing NULL, buffer overflow, use after free, or stack overflow from deep recursion.",
        "User: What's the difference between REST and GraphQL?\nAssistant: REST gives you fixed endpoints that return fixed data shapes. GraphQL gives you one endpoint where you ask for exactly the fields you need. REST is simpler to cache and reason about. GraphQL is better when your frontend needs flexible queries across many related entities.",
        "User: How do I handle authentication in a microservices architecture?\nAssistant: Use JWT tokens issued by a central auth service. Each microservice validates the token independently without calling back to auth. For service-to-service communication, use mutual TLS. Store refresh tokens server-side, never in localStorage.",
        "User: My database queries are getting slow, what should I check?\nAssistant: Run EXPLAIN ANALYZE on the slow queries. Look for sequential scans on large tables — add indexes. Check for N+1 query patterns in your ORM. Monitor connection pool saturation. Consider read replicas if you're read-heavy.",
        "User: What's the best way to handle errors in a Python API?\nAssistant: Use custom exception classes that map to HTTP status codes. Catch them in middleware that returns consistent JSON error responses. Log the full traceback server-side but only return safe messages to clients. Never expose internal details in production error responses.",
        "User: How do I set up CI/CD for a new project?\nAssistant: Start with GitHub Actions — it's free for public repos. Create a workflow that runs lint, tests, and build on every PR. Add a deploy step that triggers on merge to main. Use environment secrets for credentials. Keep pipelines under 10 minutes or developers will skip them.",
        "User: Can you explain Docker to me like I'm five?\nAssistant: Imagine you have a toy box. Everything your toy needs to work is inside that box — batteries, instructions, parts. You can carry that box anywhere and the toy works the same way. Docker is like that box but for software. It packages your code with everything it needs so it runs the same on any computer.",
        "User: What are the pros and cons of TypeScript vs JavaScript?\nAssistant: TypeScript catches bugs at compile time that JavaScript only catches at runtime. The type system is excellent for large codebases and refactoring. The downsides: slower build times, more verbose code, and you need to learn the type system. For small scripts, plain JS is fine. For anything with a team, use TypeScript.",
        "User: How should I structure a large React application?\nAssistant: Feature-based folders, not type-based. Group components, hooks, and utils by feature, not by what they are. Use a state management solution only when prop drilling gets painful. Keep components small. Extract business logic into custom hooks. Use React Query for server state.",
        "User: What's the difference between concurrency and parallelism?\nAssistant: Concurrency is about dealing with multiple things at once — like a chef switching between chopping and stirring. Parallelism is about doing multiple things at once — like two chefs each cooking different dishes. You can have concurrency without parallelism. Python's asyncio is concurrent but single-threaded.",
        "User: How do I choose between SQL and NoSQL databases?\nAssistant: If your data has clear relationships and you need ACID transactions, use SQL. If your data is document-shaped, varies in structure, or you need horizontal scaling above all else, consider NoSQL. Most applications are fine with PostgreSQL. Don't choose NoSQL just because it's trendy.",
        "User: What's the best approach to API versioning?\nAssistant: URL path versioning (/api/v1/users) is the simplest and most visible. Header versioning is cleaner but harder to test in a browser. Don't version unless you have breaking changes. When you do version, support the old version for at least 6 months with deprecation warnings.",
        "User: How do I improve the performance of my web application?\nAssistant: Measure first — don't guess. Use Chrome DevTools and Lighthouse. Common wins: lazy load images, code-split your JS bundles, add proper cache headers, use a CDN, compress responses with gzip/brotli. On the backend: add database indexes, use connection pooling, cache expensive computations.",
    ]
    result = []
    while len(result) < count:
        result.extend(base)
    return result[:count]


def _math_fallback(count):
    """Hardcoded math/reasoning problems."""
    base = [
        "Question: A store sells apples for $2 each and oranges for $3 each. If John buys 5 apples and 3 oranges, how much does he spend in total?\nAnswer: 5 × $2 = $10 for apples. 3 × $3 = $9 for oranges. Total = $10 + $9 = $19.",
        "Question: If a train travels at 60 mph for 2.5 hours, how far does it go?\nAnswer: Distance = speed × time = 60 × 2.5 = 150 miles.",
        "Problem: Find the derivative of f(x) = 3x^4 - 2x^3 + 5x - 7.\nSolution: f'(x) = 12x^3 - 6x^2 + 5. Using the power rule on each term.",
        "Question: A rectangle has length 12 cm and width 8 cm. What is its area and perimeter?\nAnswer: Area = 12 × 8 = 96 cm². Perimeter = 2(12 + 8) = 40 cm.",
        "Problem: Solve the system of equations: 2x + y = 7, x - y = 2.\nSolution: Adding the equations: 3x = 9, so x = 3. Then y = 7 - 2(3) = 1.",
        "Question: What is the probability of rolling a sum of 7 with two fair dice?\nAnswer: The favorable outcomes are (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) — that's 6 out of 36 total outcomes. P = 6/36 = 1/6.",
        "Problem: Find the eigenvalues of the matrix [[4, 1], [2, 3]].\nSolution: det(A - λI) = (4-λ)(3-λ) - 2 = λ² - 7λ + 10 = (λ-5)(λ-2). Eigenvalues: λ₁ = 5, λ₂ = 2.",
        "Question: A tank is being filled at 5 gallons per minute and drained at 2 gallons per minute. If the tank starts empty and holds 90 gallons, how long until it's full?\nAnswer: Net fill rate = 5 - 2 = 3 gallons/minute. Time = 90/3 = 30 minutes.",
        "Problem: Compute the integral of x·e^x dx.\nSolution: Using integration by parts with u = x, dv = e^x dx: ∫x·e^x dx = x·e^x - ∫e^x dx = x·e^x - e^x + C = e^x(x-1) + C.",
        "Question: In a class of 30 students, 18 play soccer, 15 play basketball, and 10 play both. How many play neither?\nAnswer: By inclusion-exclusion: soccer or basketball = 18 + 15 - 10 = 23. Neither = 30 - 23 = 7 students.",
    ]
    result = []
    while len(result) < count:
        result.extend(base)
    return result[:count]


def _multilingual_fallback(count):
    """Hardcoded multilingual sentences (German, French, Spanish)."""
    base = [
        "Die Quantenmechanik beschreibt das Verhalten von Teilchen auf subatomarer Ebene und stellt unser Verständnis der physikalischen Welt grundlegend in Frage.",
        "Le développement de l'intelligence artificielle transforme profondément notre société, de la médecine à l'éducation en passant par les transports.",
        "La biodiversidad de los ecosistemas tropicales es extraordinariamente rica y alberga millones de especies que aún no han sido catalogadas por la ciencia.",
        "Erneuerbare Energien wie Solar- und Windkraft spielen eine immer wichtigere Rolle bei der Bekämpfung des Klimawandels und der Energiewende.",
        "La révolution numérique a profondément changé la façon dont nous communiquons, travaillons et accédons à l'information au quotidien.",
        "El cambio climático representa uno de los mayores desafíos de nuestra generación y requiere una acción coordinada a nivel mundial.",
        "Die deutsche Automobilindustrie steht vor einem tiefgreifenden Wandel durch die Elektrifizierung und die Entwicklung autonomer Fahrsysteme.",
        "Les neurosciences computationnelles permettent de mieux comprendre le fonctionnement du cerveau humain grâce à des modèles mathématiques sophistiqués.",
        "La literatura latinoamericana del siglo veinte produjo obras maestras que revolucionaron las formas narrativas y expandieron los límites de la ficción.",
        "Künstliche Intelligenz und maschinelles Lernen revolutionieren die medizinische Diagnostik und ermöglichen präzisere Behandlungsmethoden.",
        "L'architecture gothique des cathédrales médiévales témoigne de l'extraordinaire maîtrise technique des bâtisseurs du Moyen Âge.",
        "Los avances en la edición genética mediante la tecnología CRISPR abren nuevas posibilidades para el tratamiento de enfermedades hereditarias.",
        "Die Philosophie Immanuel Kants hat das europäische Denken nachhaltig geprägt und bleibt ein zentraler Bezugspunkt der modernen Erkenntnistheorie.",
        "La gastronomie française est reconnue mondialement pour sa diversité, sa sophistication et son attachement aux produits du terroir.",
        "El telescopio espacial James Webb está revelando detalles sin precedentes sobre las primeras galaxias que se formaron después del Big Bang.",
    ]
    result = []
    while len(result) < count:
        result.extend(base)
    return result[:count]


def _mixed_edge_cases(count):
    """Generate synthetic mixed/edge-case inputs."""
    cases = [
        "Here is a JSON config:\n```json\n{\"model\": \"gpt-4\", \"temperature\": 0.7, \"max_tokens\": 2048, \"stop\": [\"\\n\\n\", \"END\"]}\n```\nMake sure to set temperature to 0 for deterministic outputs.",
        "Check out the docs at https://docs.example.com/api/v2/authentication#oauth2-pkce and https://github.com/org/repo/issues/1234 for more context on the auth flow changes.",
        "The LaTeX formula $\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\nabla^2 u$ describes wave propagation in a medium with speed $c$.",
        "🎉 We just hit 1M users! 🚀 Our DAU is up 340% MoM. Key metrics: p99 latency < 50ms, error rate 0.01%, uptime 99.99%. The team shipped 47 PRs this sprint 💪",
        "Error log from production:\n```\n2024-03-15T14:23:45.123Z ERROR [worker-7] Connection refused: redis://cache-primary:6379\n2024-03-15T14:23:45.456Z WARN  [worker-7] Falling back to cache-secondary:6380\n2024-03-15T14:23:46.789Z INFO  [worker-7] Reconnected to cache-primary:6379\n```",
        "# API Design Notes\n\n## Endpoints\n- `GET /api/v2/users/:id` → returns user profile\n- `POST /api/v2/users` → creates new user\n- `PATCH /api/v2/users/:id` → partial update\n\n## Auth\nAll endpoints require Bearer token in Authorization header.",
        "The CSS grid layout:\n```css\n.container {\n  display: grid;\n  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));\n  gap: 1.5rem;\n  padding: 2rem;\n}\n.card {\n  border-radius: 8px;\n  box-shadow: 0 2px 8px rgba(0,0,0,0.1);\n}\n```",
        "Meeting notes 2024-Q1 review: Revenue $4.2M (+18% YoY), EBITDA margin 23.5% (target was 22%), headcount 147 (+12 from Q4). Key risks: (1) supply chain delays for GPU orders, (2) competitor launched similar product, (3) key engineer departing in April.",
        "Terraform plan output:\n```\n# module.vpc.aws_subnet.private[0] will be created\n+ resource \"aws_subnet\" \"private\" {\n    + cidr_block = \"10.0.1.0/24\"\n    + vpc_id     = (known after apply)\n    + tags = { \"Name\" = \"private-us-east-1a\" }\n  }\nPlan: 23 to add, 0 to change, 0 to destroy.\n```",
        "The regex pattern `(?<=@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*` matches the domain part of an email address.",
        "Benchmark results:\n| Model | MMLU | HumanEval | MATH | MT-Bench |\n|-------|------|-----------|------|---------|\n| GPT-4o | 88.7 | 90.2 | 76.6 | 9.2 |\n| Claude 3.5 | 88.7 | 92.0 | 71.1 | 9.1 |\n| Llama 3.1 70B | 86.0 | 80.5 | 68.0 | 8.6 |",
        "Docker compose for local dev:\n```yaml\nservices:\n  app:\n    build: .\n    ports: [\"3000:3000\"]\n    environment:\n      DATABASE_URL: postgres://user:pass@db:5432/myapp\n      REDIS_URL: redis://cache:6379\n    depends_on: [db, cache]\n  db:\n    image: postgres:16\n    volumes: [pgdata:/var/lib/postgresql/data]\n```",
        "Patent claim 1: A computer-implemented method for training a neural network, comprising: (a) receiving a plurality of training samples; (b) computing a loss function using cross-entropy between predicted and target distributions; (c) backpropagating gradients through attention layers using scaled dot-product attention with O(n²) complexity.",
        "The protein sequence MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG begins with a methionine start codon and contains the characteristic binding domain for ATP hydrolysis.",
        "Git conflict resolution:\n```diff\n<<<<<<< HEAD\nconst MAX_RETRIES = 3;\nconst TIMEOUT_MS = 5000;\n=======\nconst MAX_RETRIES = 5;\nconst TIMEOUT_MS = 10000;\n>>>>>>> feature/increase-resilience\n```\nI'd go with the higher values from the feature branch.",
    ]
    result = []
    while len(result) < count:
        result.extend(cases)
    random.shuffle(result)
    return result[:count]


# ---------------------------------------------------------------------------
# Input processing and alignment
# ---------------------------------------------------------------------------

def process_input(text, tokenizer_a, tokenizer_b, max_tokens, min_tokens):
    """
    Tokenize with both models, truncate to max_tokens, align, and return metadata.
    Returns None if input should be filtered out.
    """
    # Tokenize both
    enc_a = tokenizer_a(text, add_special_tokens=False)
    enc_b = tokenizer_b(text, add_special_tokens=False)
    ids_a = enc_a["input_ids"]
    ids_b = enc_b["input_ids"]

    if len(ids_a) == 0 or len(ids_b) == 0:
        return None

    # Determine which model uses more tokens
    if len(ids_a) >= len(ids_b):
        longer_tok, longer_ids = tokenizer_a, ids_a
    else:
        longer_tok, longer_ids = tokenizer_b, ids_b

    # Truncate to max_tokens on the longer side
    if len(longer_ids) > max_tokens:
        # Find the character position where the max_tokens-th token ends
        truncated_ids = longer_ids[:max_tokens]
        truncated_text = longer_tok.decode(truncated_ids, skip_special_tokens=True)
        # Use this truncated text length to clip the original
        # Some decoded text may be slightly different, so use character count
        trunc_len = len(truncated_text)
        text = text[:trunc_len]

        # Re-tokenize both with truncated text
        enc_a = tokenizer_a(text, add_special_tokens=False)
        enc_b = tokenizer_b(text, add_special_tokens=False)
        ids_a = enc_a["input_ids"]
        ids_b = enc_b["input_ids"]

    # Filter too-short inputs
    if len(ids_a) < min_tokens or len(ids_b) < min_tokens:
        return None

    # Run alignment
    try:
        alignment = align_tokens(text, tokenizer_a, tokenizer_b,
                                  name_a=config.MODEL_A_NAME, name_b=config.MODEL_B_NAME)
        pairs = get_activation_pairs(alignment)
        buckets = classify_alignment(alignment)
    except Exception:
        return None

    # Only keep bucket-1 (clean) pairs for activation comparison
    clean_pairs = [p for p in pairs if p.bucket == 1]
    if len(clean_pairs) == 0:
        return None

    return {
        "raw_text": text,
        "model_a_token_count": len(ids_a),
        "model_b_token_count": len(ids_b),
        "model_a_positions": [p.model_a_positions[0] for p in clean_pairs],
        "model_b_positions": [p.model_b_positions[0] for p in clean_pairs],
        "num_clean_pairs": len(clean_pairs),
        "bucket_distribution": {
            "bucket_1": buckets["bucket_1_count"],
            "bucket_2": buckets["bucket_2_count"],
            "bucket_3": buckets["bucket_3_count"],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CORPUS BUILDER — Phase 2 Activation Experiment")
    print("=" * 70)

    # Load tokenizers
    print("\nLoading tokenizers...")
    tokenizer_a = load_tokenizer(config.MODEL_A_ID, config.MODEL_A_FALLBACKS, config.MODEL_A_NAME)
    tokenizer_b = load_tokenizer(config.MODEL_B_ID, config.MODEL_B_FALLBACKS, config.MODEL_B_NAME)

    # Load raw texts by category
    print("\nLoading datasets...")
    raw_by_category = {}
    loaders = {
        "english_web": load_english_web,
        "code": load_code,
        "conversational": load_conversational,
        "math_reasoning": load_math_reasoning,
        "multilingual": load_multilingual,
        "mixed_edge": load_mixed_edge,
    }

    for cat, target_count in config.CORPUS_CATEGORIES.items():
        print(f"\n--- {cat} (target: {target_count}) ---")
        raw_texts = loaders[cat](target_count * 2)  # load 2x to account for filtering
        raw_by_category[cat] = raw_texts
        print(f"  Loaded {len(raw_texts)} raw texts")

    # Process and align
    print("\n\nProcessing and aligning inputs...")
    corpus = []
    activation_index = {}  # maps (input_id, pair_index) -> row in memmap
    global_row = 0

    for cat, target_count in config.CORPUS_CATEGORIES.items():
        print(f"\n  {cat}: ", end="", flush=True)
        cat_count = 0
        for text in raw_by_category[cat]:
            if cat_count >= target_count:
                break

            result = process_input(text, tokenizer_a, tokenizer_b,
                                    config.MAX_TOKENS, config.MIN_TOKENS)
            if result is None:
                continue

            input_id = len(corpus)
            entry = {
                "input_id": input_id,
                "category": cat,
                **result,
            }
            corpus.append(entry)

            # Build activation index
            for pair_idx in range(result["num_clean_pairs"]):
                activation_index[f"{input_id}_{pair_idx}"] = global_row
                global_row += 1

            cat_count += 1
            if cat_count % 200 == 0:
                print(f"{cat_count}", end=" ", flush=True)

        print(f"→ {cat_count} inputs")

    # Save corpus
    print(f"\nTotal corpus size: {len(corpus)} inputs")
    print(f"Total aligned token pairs: {global_row}")

    with open(config.CORPUS_PATH, "w") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {config.CORPUS_PATH}")

    # Save activation index
    index_meta = {
        "total_rows": global_row,
        "total_inputs": len(corpus),
        "index": activation_index,
        "model_a_layers": config.MODEL_A_LAYERS + 1,  # +1 for embedding layer
        "model_b_layers": config.MODEL_B_LAYERS + 1,
        "hidden_dim": config.HIDDEN_DIM,
    }
    with open(config.ACTIVATION_INDEX_PATH, "w") as f:
        json.dump(index_meta, f, indent=2)
    print(f"Saved {config.ACTIVATION_INDEX_PATH}")

    # Print category breakdown
    print("\nCategory breakdown:")
    cat_counts = {}
    cat_pairs = {}
    for entry in corpus:
        cat = entry["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_pairs[cat] = cat_pairs.get(cat, 0) + entry["num_clean_pairs"]
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat:>20}: {cat_counts[cat]:>5} inputs, {cat_pairs[cat]:>7} aligned pairs")

    print(f"\nCorpus ready for activation extraction.")


if __name__ == "__main__":
    main()
