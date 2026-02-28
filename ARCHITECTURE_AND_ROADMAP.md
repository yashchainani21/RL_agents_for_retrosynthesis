# Architecture Overview & Future Roadmap

This document summarizes the current strengths of the RL Agents for Retrosynthesis codebase and provides recommendations for future improvements.

---

## Current Strengths

### 1. Dual-Agent Architecture
- **DORAnet Agent**: Performs retrosynthetic fragmentation using both enzymatic and synthetic transformations
- **RetroTide Agent**: Forward synthesis verification for PKS (polyketide synthase) pathway design
- **Hierarchical Search**: DORAnet fragments targets, then spawns RetroTide to verify PKS-synthesizable fragments

### 2. Parallel MCTS with Virtual Loss
- Thread-safe tree exploration using virtual loss technique (Chaslot et al., 2008)
- Configurable worker count with automatic CPU detection
- Lock-based synchronization for tree modifications
- Significant speedup on multi-core systems without sacrificing search diversity

### 3. Efficient Caching Strategy
- **SMILES Canonicalization Cache**: LRU cache (100K entries) for RDKit canonicalization calls
- **PKS Library Disk Cache**: Pickle-based caching for the 100K+ SMILES library (~841x speedup on reload)
- **Sink Compounds Cache**: Disk caching for 380K+ building block SMILES

### 4. Comprehensive Terminal Detection
- **Sink Compounds**: Chemical (~370K) and biological (~13K) building blocks that don't need further fragmentation
- **PKS Library Matching**: 100K+ known PKS products for quick feasibility checks
- **Prohibited Chemicals**: Filtering of hazardous/controlled substances

### 5. Flexible Configuration
- Configurable selection policies: UCB1 (standard) or depth-biased exploration
- Child downselection strategies: `first_N` (DORAnet order) or `hybrid` (sink > PKS > smaller MW)
- Tunable parameters: max_depth, iterations, virtual_loss, exploration constants

### 6. Rich Visualization
- Interactive HTML visualizations using Plotly
- Full tree view and pathways-only view
- Hover tooltips with molecule structures, reaction SMARTS, and node metadata
- Color-coded nodes by provenance (enzymatic/synthetic) and terminal status

### 7. Detailed Diagnostics
- Node selection vs. creation tracking (created_at, expanded_at, selected_at)
- Pathway extraction with success criteria validation
- Comprehensive results files with iteration-level diagnostics

### 8. Clean Package Structure
- Pip-installable with `setup.py` and `pyproject.toml`
- Modular agent packages: `DORAnet_agent`, `RetroTide_agent`, `utils`
- Optional dependencies for DORAnet and RetroTide integrations
- Unit tests for core MCTS functionality

---

## Areas for Improvement

### Algorithm & Search Enhancements

#### 1. Learned Value Function
**Current State**: Rewards are binary heuristics (PKS match = 1.0, sink = 1.0)

**Recommendation**: Train a neural network to predict synthesis feasibility scores based on:
- Molecular descriptors (fingerprints, physicochemical properties)
- Reaction type and complexity
- Historical success rates from literature

**Impact**: More accurate node value estimates would improve search efficiency and pathway quality.

#### 2. True Rollout/Simulation Phase
**Current State**: MCTS performs selection + expansion but no random rollout

**Recommendation**: Implement lightweight rollout policy:
- Random reaction selection to leaf nodes
- Or use a trained policy network for guided rollouts
- Average rollout outcomes for more robust value estimates

**Impact**: Better exploration of deep pathways; reduced sensitivity to early expansion choices.

#### 3. Adaptive Exploration Constants
**Current State**: Fixed UCB1 exploration constant (sqrt(2))

**Recommendation**:
- Implement UCB1-Tuned or PUCT (Predictor + UCB for Trees)
- Decay exploration over time as tree matures
- Per-node exploration based on uncertainty estimates

**Impact**: Better exploration/exploitation balance throughout search.

#### 4. Iterative Deepening / Beam Search
**Current State**: Breadth-first expansion limited by early termination

**Recommendation**:
- Iterative deepening: progressively increase max_depth
- Beam search: maintain top-K most promising paths
- Hybrid approaches for targeted deep exploration

**Impact**: Discover longer synthesis routes that current search misses.

---

### Scalability Improvements

#### 5. True Multiprocessing / MPI Support
**Current State**: Python threading (GIL-limited for CPU-bound work)

**Recommendation**:
- Refactor to `multiprocessing` for single-machine parallelism
- Implement MPI-based distributed search for HPC clusters
- Central coordinator pattern with worker nodes

**Complexity**: Moderate-to-high; requires serialization of tree state and careful synchronization.

**Impact**: Linear scaling on HPC systems; enables much larger searches.

#### 6. Memory-Efficient Tree Storage
**Current State**: All nodes kept in memory

**Recommendation**:
- Implement tree pruning for unpromising branches
- Disk-backed node storage for very large trees
- Lazy loading of node details

**Impact**: Enable searches with millions of nodes without memory exhaustion.

#### 7. Batched DORAnet Expansion
**Current State**: Sequential DORAnet calls per node

**Recommendation**: Batch multiple molecules for single DORAnet network generation

**Impact**: Reduced overhead from repeated network initialization.

---

### Chemistry & Domain Enhancements

#### 8. Reaction Feasibility Scoring
**Current State**: All DORAnet reactions treated equally

**Recommendation**:
- Integrate thermodynamic feasibility predictions (deltaG)
- Add kinetic accessibility estimates
- Score by literature precedent / reaction database frequency

**Impact**: Prioritize chemically realistic pathways.

#### 9. Reaction Condition Compatibility
**Current State**: No checking of sequential reaction compatibility

**Recommendation**:
- Track reaction conditions (pH, temperature, solvents)
- Flag incompatible consecutive steps
- Suggest protection/deprotection strategies

**Impact**: More practically implementable synthesis routes.

#### 10. Improved Stereochemistry Handling
**Current State**: SMILES canonicalization may lose stereochemistry

**Recommendation**:
- Use isomeric SMILES throughout
- Track stereocenters explicitly
- Consider stereochemical outcomes of reactions

**Impact**: Correct stereochemistry in final pathway designs.

#### 11. Retrobiosynthesis Rule Curation
**Current State**: Uses DORAnet's built-in enzymatic rules

**Recommendation**:
- Curate rules based on experimentally validated transformations
- Add confidence scores to reaction rules
- Incorporate organism-specific enzyme availability

**Impact**: Higher confidence in enzymatic pathway feasibility.

---

### RetroTide Integration

#### 12. Shared Learning Across Searches
**Current State**: Each RetroTide search is independent

**Recommendation**:
- Cache successful PKS module sequences
- Transfer learned patterns between related targets
- Build a PKS design database from successful searches

**Impact**: Faster convergence for similar targets.

#### 13. Bidirectional Feedback Loop
**Current State**: RetroTide results don't influence DORAnet selection

**Recommendation**:
- Update DORAnet node values based on RetroTide success/failure
- Prioritize fragments with verified PKS routes
- Penalize fragments that consistently fail RetroTide

**Impact**: Smarter fragment selection based on actual synthesizability.

#### 14. Parallel RetroTide Searches
**Current State**: RetroTide searches run sequentially after expansion

**Recommendation**:
- Launch RetroTide searches in parallel thread pool
- Asynchronous result collection
- Early termination when sufficient routes found

**Impact**: Faster overall search when multiple PKS matches found.

---

### Code Quality & User Experience

#### 15. CLI Target Molecule Input
**Current State**: Target SMILES hardcoded in runner script

**Recommendation**:
```bash
python scripts/run_DORAnet_single_agent.py --smiles "CCCCCCCCC(=O)O" --name nonanoic_acid
```

**Impact**: More flexible usage without code modification.

#### 16. Web API / Interface
**Current State**: CLI-only access

**Recommendation**:
- FastAPI/Flask REST API for programmatic access
- Simple web interface for non-programmers
- Real-time search progress visualization

**Impact**: Broader accessibility for chemists and biologists.

#### 17. Refactor Shared MCTS Logic
**Current State**: Some duplication between `DORAnetMCTS` and `AsyncExpansionDORAnetMCTS`

**Recommendation**:
- Extract base `MCTSBase` class with common methods
- Inherit and override only parallel-specific logic
- Cleaner separation of concerns

**Impact**: Easier maintenance and extension.

#### 18. Expanded Test Coverage
**Current State**: 24 unit tests covering core functionality

**Recommendation**:
- Integration tests for full DORAnet → RetroTide pipeline
- Property-based testing for SMILES handling
- Benchmark tests for performance regression detection

**Impact**: Higher confidence in code correctness after changes.

#### 19. Configuration File Support
**Current State**: All parameters via CLI arguments

**Recommendation**:
- YAML/JSON configuration files for complex setups
- Named configuration profiles (e.g., "quick_search", "thorough_search")
- Environment variable overrides

**Impact**: Reproducible experiments; easier parameter management.

---

## Priority Recommendations

### High Priority (High Impact, Moderate Effort)
1. **CLI target molecule input** - Quick win for usability
2. **Bidirectional feedback loop** - Improves search quality significantly
3. **Reaction feasibility scoring** - More realistic pathways

### Medium Priority (High Impact, Higher Effort)
4. **Learned value function** - Requires training data and ML pipeline
5. **MPI support** - Enables HPC scaling
6. **True rollout phase** - Fundamental MCTS improvement

### Lower Priority (Nice to Have)
7. **Web interface** - Broader accessibility
8. **Iterative deepening** - Alternative search strategy
9. **Configuration file support** - Quality of life improvement

---

## References

- Chaslot, G., Winands, M.H.M., & van den Herik, H.J. (2008). "Parallel Monte-Carlo Tree Search."
- Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search."
- Segler, M.H.S., et al. (2018). "Planning chemical syntheses with deep neural networks and symbolic AI."

---

*Document generated: January 2026*
*Repository: RL Agents for Retrosynthesis v0.1.0*
