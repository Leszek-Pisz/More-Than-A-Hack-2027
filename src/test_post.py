"""Quick check of post.py on made-up sequences (no match data).

    uv run python src/test_post.py
"""

import numpy as np

import post

H, A, D = 0, 1, 2
P = {"post": {"enabled": True, "min_poss_s": 1.0, "min_dead_s": 1.0}}   # 25 frames


def seq(*parts):
    return np.concatenate([np.full(n, lab, dtype=np.int8) for lab, n in parts])


def check(name, got, want):
    ok = np.array_equal(got, want)
    print(f"{'OK  ' if ok else 'FAIL'} {name}")
    return ok


results = [
    check("H-A(10)-H -> H", post.merge_short(seq((H, 100), (A, 10), (H, 100)), P), seq((H, 210),)),
    check("H-D(10)-H -> H", post.merge_short(seq((H, 100), (D, 10), (H, 100)), P), seq((H, 210),)),
    check("H-A(30)-H stays (long enough)",
          post.merge_short(seq((H, 100), (A, 30), (H, 100)), P), seq((H, 100), (A, 30), (H, 100))),
    check("H-A(10)-D stays (boundary not moved)",
          post.merge_short(seq((H, 100), (A, 10), (D, 100)), P), seq((H, 100), (A, 10), (D, 100))),
    check("blip at the edge stays", post.merge_short(seq((A, 5), (H, 100)), P), seq((A, 5), (H, 100))),
    check("H-A(3)-H(5)-A -> shortest first",
          post.merge_short(seq((H, 100), (A, 3), (H, 5), (A, 100)), P), seq((H, 108), (A, 100))),
    check("off = unchanged",
          post.merge_short(seq((H, 100), (A, 10), (H, 100)), {"post": {"enabled": False}}),
          seq((H, 100), (A, 10), (H, 100))),
]
print("all OK" if all(results) else "SOMETHING FAILED")
