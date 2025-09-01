import argparse, json, requests, sys, shlex

DEFAULT_BASE = "http://127.0.0.1:8000"

# Pretty print JSON
def pp(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))

# Health check
def do_health(base):
    r = requests.get(f"{base}/health", timeout=30)
    pp(r.json())

# Get all books
def do_books(base):
    r = requests.get(f"{base}/books", timeout=30)
    pp(r.json())

# Get a specific book
def do_book(base, title):
    r = requests.get(f"{base}/books/{title}", timeout=30)
    if r.status_code == 404:
        print("Book not found")
    else:
        pp(r.json())

# Search for books
def do_search(base, q, theme=None, k=5):
    params = {"q": q, "k": k}
    if theme:
        params["theme"] = theme
    r = requests.get(f"{base}/search", params=params, timeout=60)
    pp(r.json())

# Recommend books with summaries from local tool
def do_recommend(base, query, theme=None, k=5):
    payload = {"query": query, "k": k}
    if theme:
        payload["theme"] = theme
    r = requests.post(f"{base}/recommend", json=payload, timeout=120)
    pp(r.json())

# Verify safety of text
def do_check(base, text):
    r = requests.get(f"{base}/safety/check", params={"q": text}, timeout=15)
    pp(r.json())

# Rebuild the index
def do_rebuild(base):
    r = requests.post(f"{base}/admin/rebuild", timeout=120)
    pp(r.json())

#chat pt endpoint-ul /ollama/chat
def do_chat(base, temperature=0.4):
    print("Chat Ollama (/exit pt a iesi).")
    history = []
    while True:
        try:
            msg = input("you> ").strip()
        except EOFError:
            print()
            break
        if not msg:
            continue
        if msg in ("/exit", "/quit"):
            break

        # verificare safety local
        s = requests.get(f"{base}/safety/check", params={"q": msg}, timeout=15).json()
        if s.get("bad"):
            print("Limbaj nepotrivit, reformuleaza. Termeni blocati:", ", ".join(s.get("hits", [])))
            continue

        # adaugare mesaj utilizator in istoric
        history.append({"role": "user", "content": msg})
        r = requests.post(f"{base}/ollama/chat",
                          json={"messages": history, "temperature": temperature},
                          timeout=180)
        data = r.json()
        if not data.get("safe", True):
            print("bot:", data.get("content", "Mesaj blocat din cauza limbajului"))
            continue
        content = data.get("content", "")
        print("bot:", content)
        history.append({"role": "assistant", "content": content})

# Rularea shell-ului
def run_shell(base):
    print("Smart Librarian")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in ("exit", "quit"):
            break
        if line == "help":
            print("""Comenzi:
  health
  books
  book "<title>"
  search -q "<text>" [-t "<theme>"] [-k N]
  ask -q "<text>" [-t "<theme>"] [-k N]
  check "<text>"
  rebuild
  chat
""")
            continue

        toks = shlex.split(line)
        cmd = toks[0]
        args = toks[1:]

        try:
            if cmd == "health":
                do_health(base)
            elif cmd == "books":
                do_books(base)
            elif cmd == "book":
                do_book(base, " ".join(args))
            elif cmd == "search":
                q, t, k = None, None, 5
                i = 0
                while i < len(args):
                    if args[i] == "-q":
                        q = args[i+1]; i += 2
                    elif args[i] == "-t":
                        t = args[i+1]; i += 2
                    elif args[i] == "-k":
                        k = int(args[i+1]); i += 2
                    else:
                        i += 1
                if not q: raise ValueError("Foloseste: search -q \"text\" [-t \"theme\"] [-k N]")
                do_search(base, q, t, k)
            elif cmd == "ask":
                q, t, k = None, None, 5
                i = 0
                while i < len(args):
                    if args[i] == "-q":
                        q = args[i+1]; i += 2
                    elif args[i] == "-t":
                        t = args[i+1]; i += 2
                    elif args[i] == "-k":
                        k = int(args[i+1]); i += 2
                    else:
                        i += 1
                if not q: raise ValueError("Foloseste: ask -q \"text\" [-t \"theme\"] [-k N]")
                do_recommend(base, q, t, k)
            elif cmd == "check":
                do_check(base, " ".join(args))
            elif cmd == "rebuild":
                do_rebuild(base)
            elif cmd == "chat":
                do_chat(base)
            else:
                print("Comanda necunoscuta")
        except Exception as e:
            print("Eroare:", e)

def main():
    p = argparse.ArgumentParser(description="Smart Librarian")
    p.add_argument("--base-url", default=DEFAULT_BASE, help="URL backend FastAPI")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("health")
    g = sub.add_parser("books")
    g = sub.add_parser("book"); g.add_argument("title")

    g = sub.add_parser("search")
    g.add_argument("-q", "--query", required=True)
    g.add_argument("-t", "--theme", default=None)
    g.add_argument("-k", "--k", type=int, default=5)

    g = sub.add_parser("ask")
    g.add_argument("-q", "--query", required=True)
    g.add_argument("-t", "--theme", default=None)
    g.add_argument("-k", "--k", type=int, default=5)

    g = sub.add_parser("check"); g.add_argument("text")
    g = sub.add_parser("rebuild")
    g = sub.add_parser("chat")
    g = sub.add_parser("shell")

    args = p.parse_args()
    base = args.base_url.rstrip("/")

    if args.cmd == "health": do_health(base)
    elif args.cmd == "books": do_books(base)
    elif args.cmd == "book": do_book(base, args.title)
    elif args.cmd == "search": do_search(base, args.query, args.theme, args.k)
    elif args.cmd == "ask": do_recommend(base, args.query, args.theme, args.k)
    elif args.cmd == "check": do_check(base, args.text)
    elif args.cmd == "rebuild": do_rebuild(base)
    elif args.cmd == "chat": do_chat(base)
    else:
        run_shell(base)

if __name__ == "__main__":
    main()
