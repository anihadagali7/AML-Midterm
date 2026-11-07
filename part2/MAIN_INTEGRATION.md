# Part Two integration onto main

The Part Two research branch inherited earlier Part One research commits. A
whole-branch merge would have included those changes. This integration replays
only `part2/` and the required ignore rules from each Part Two commit onto main
at `55cf05c`, retaining separate trial-phase and submission-record commits.
The root README now points to the leading Part Two submission. Part One's tree
is unchanged from that main baseline; ongoing worktrees are untouched.

Before integration documentation was added, the complete Part Two Git tree was
byte-identical to research head `9da0e1e`. No model implementation, frozen
protocol, benchmark result, raw input or submission CSV was edited during replay.
The notebook conflict was resolved to the research branch's Part Two notebook;
the old main notebook remains available in main's previous commit.

## Original and replayed commits

| Original commit | Main-integration commit | Phase/result |
| --- | --- | --- |
| `c12df86` | `87d2123` | Build Part Two experiments and reproducible Ani benchmark |
| `6f23dd5` | `e5369dc` | Compare both Part Two Ani winners with 24 improvement variants |
| `135765c` | `6931bf1` | Extend Ani outcome stack with recent clinical neighbors |
| `3a880e9` | `a9b2c81` | Benchmark every Part Two submission on identical fixed splits |
| `8d63e10` | `d5d634f` | Record file 10 Kaggle improvement to 0.58609 |
| `269630e` | `658d7b7` | Evaluate Part Two sprint candidates against file 10 on fixed splits |
| `b6f29a2` | `76223f6` | Record file 11 Kaggle first place at 0.58474 |
| `a250cb9` | `e7c5076` | Refine Part Two winner to 0.58465 and hold final submission slot |
| `a29bf1e` | `0a6eece` | Finalize Part Two sprint documentation and reserve status |
| `0346bbe` | `7a84798` | Record Part Two submission 03: Kaggle 0.59562 |
| `a5d3004` | `2240fbd` | Record Part Two submission 02: Kaggle 0.58993 |
| `6fa16c3` | `dd3b2f1` | Record Part Two submission 10: Kaggle 0.58609 |
| `341fd10` | `699b5ad` | Record Part Two submission 11: Kaggle 0.58474 |
| `9c009aa` | `d81d80e` | Record Part Two submission 13: Kaggle 0.58465 |
| `22e03c4` | `0135c64` | Record Part Two submission 15: Kaggle 0.58492 |
| `9da0e1e` | `71d64e1` | Document Part Two trial phases and verify scored submission tags |

The annotated score tags retain their original targets. They identify exact
submitted CSVs and original submission records, not new Kaggle uploads. Their
original commits remain reachable through the published tags; the table above
maps them to the equivalent record commits in the main-integration history.
See [submission history](submissions/HISTORY.md) and the machine-readable
[mapping](main_integration.json).

The winner remains file 13 at **0.58465**. File 16 is held and unsubmitted.
No additional candidate is promoted by this integration.

## Verification

All 16 regression tests pass. All six annotated score tags and original CSV
hashes verify. The common benchmark independently rescored 338 model/split
results. File 13 reproduces from native models with maximum probability error
1.11e-16, including after evaluation labels and identifiers are replaced. The
Part One tree hash matches the main baseline. See [verification](experiments/main_integration/verification.json).
