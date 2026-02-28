"""
Coupling experiment: Can DORAnet synthetic chemistry couple these two fragments
to produce a kavalactone?

Reactant 1: CCC1CC(OC)=CC(=O)O1  (6-ethyl-4-methoxy-5,6-dihydro-2H-pyran-2-one)
Reactant 2: COc1ccc(O)cc1         (4-methoxyphenol)
Target:     COC1=CC(=O)OC(CCc2ccc(OC)cc2)C1  (kavalactone)
"""

import doranet.modules.synthetic as synthetic
from rdkit import Chem
from rdkit.Chem import Descriptors

# ── Define molecules ──────────────────────────────────────────────────────────
REACTANT_1 = "CCC1CC(OC)=CC(=O)O1"
REACTANT_2 = "COc1ccc(O)cc1"
TARGET = "COC1=CC(=O)OC(CCc2ccc(OC)cc2)C1"

# Canonicalize everything for consistent comparison
canon_r1 = Chem.MolToSmiles(Chem.MolFromSmiles(REACTANT_1))
canon_r2 = Chem.MolToSmiles(Chem.MolFromSmiles(REACTANT_2))
canon_target = Chem.MolToSmiles(Chem.MolFromSmiles(TARGET))

print("=" * 70)
print("COUPLING EXPERIMENT: DORAnet Synthetic Chemistry")
print("=" * 70)
print(f"Reactant 1 : {REACTANT_1}  →  canonical: {canon_r1}")
print(f"Reactant 2 : {REACTANT_2}  →  canonical: {canon_r2}")
print(f"Target     : {TARGET}  →  canonical: {canon_target}")
print(f"Target MW  : {Descriptors.MolWt(Chem.MolFromSmiles(TARGET)):.2f}")
print()

# ── Chemistry helpers ─────────────────────────────────────────────────────────
helpers = {
    "O", "O=O", "[H][H]", "O=C=O", "C=O",
    "[C-]#[O+]", "Br", "[Br][Br]", "CO",
    "C=C", "O=S(O)O", "N", "O=S(=O)(O)O",
    "O=NO", "N#N", "O=[N+]([O-])O", "NO",
    "C#N", "S", "O=S=O", "N#CO",
}

# ── Run forward synthesis ─────────────────────────────────────────────────────
print("Running DORAnet synthetic forward generation (gen=1)...")
print(f"  Starters: {{{canon_r1}, {canon_r2}}}")
print(f"  Helpers : {len(helpers)} small molecules")
print()

network = synthetic.generate_network(
    job_name="coupling_experiment",
    starters={REACTANT_1, REACTANT_2},
    helpers=helpers,
    gen=1,
    direction="forward",
)

# ── Parse results ─────────────────────────────────────────────────────────────
mols_list = list(network.mols)
ops_list = list(network.ops)

print(f"Network generated: {len(mols_list)} molecules, {len(list(network.rxns))} reactions")
print()

# Build mol index → SMILES mapping
mol_smiles = {}
for i, mol_obj in enumerate(mols_list):
    uid = getattr(mol_obj, "uid", None)
    if uid:
        mol_smiles[i] = str(uid)

# Rebuild rxns list (iterator may be consumed)
network2 = synthetic.generate_network(
    job_name="coupling_experiment_2",
    starters={REACTANT_1, REACTANT_2},
    helpers=helpers,
    gen=1,
    direction="forward",
)
mols_list2 = list(network2.mols)
rxns_list = list(network2.rxns)

mol_smiles2 = {}
for i, mol_obj in enumerate(mols_list2):
    uid = getattr(mol_obj, "uid", None)
    if uid:
        mol_smiles2[i] = str(uid)

# ── Check for target ──────────────────────────────────────────────────────────
print("=" * 70)
print("CHECKING FOR TARGET PRODUCT")
print("=" * 70)

# Collect all unique product SMILES (excluding starters and helpers)
starter_canon = {canon_r1, canon_r2}
helper_canon = set()
for h in helpers:
    m = Chem.MolFromSmiles(h)
    if m:
        helper_canon.add(Chem.MolToSmiles(m))

products = {}  # canonical SMILES → original SMILES
target_found = False

for i, mol_obj in enumerate(mols_list):
    uid = getattr(mol_obj, "uid", None)
    if not uid:
        continue
    smiles_str = str(uid)
    if "*" in smiles_str:
        continue
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        continue
    canonical = Chem.MolToSmiles(mol)
    if canonical in starter_canon or canonical in helper_canon:
        continue
    products[canonical] = smiles_str
    if canonical == canon_target:
        target_found = True

print(f"\nTotal unique products (excluding starters/helpers): {len(products)}")
print()

if target_found:
    print(">>> TARGET FOUND! <<<")
    print(f"    {canon_target}")
else:
    print("Target NOT found among products.")
    print()

# ── List all products with MW ─────────────────────────────────────────────────
print()
print("=" * 70)
print("ALL PRODUCTS (sorted by MW)")
print("=" * 70)

product_list = []
for canonical, original in products.items():
    mol = Chem.MolFromSmiles(canonical)
    mw = Descriptors.MolWt(mol) if mol else 0
    product_list.append((mw, canonical, original))

product_list.sort()

for mw, canonical, original in product_list:
    marker = " <<<< TARGET" if canonical == canon_target else ""
    print(f"  MW {mw:7.2f}  {canonical}{marker}")

# ── Show reactions involving both reactants ───────────────────────────────────
print()
print("=" * 70)
print("COUPLING REACTIONS (involving both reactants)")
print("=" * 70)

ops_list2 = list(network2.ops)

coupling_rxns = []
for rxn in rxns_list:
    reactant_smiles_set = set()
    for idx in rxn.reactants:
        s = mol_smiles2.get(idx, "")
        m = Chem.MolFromSmiles(s)
        if m:
            reactant_smiles_set.add(Chem.MolToSmiles(m))

    # Check if both reactants (or their canonical forms) appear
    has_r1 = canon_r1 in reactant_smiles_set
    has_r2 = canon_r2 in reactant_smiles_set

    if has_r1 and has_r2:
        product_smiles_list = []
        for idx in rxn.products:
            s = mol_smiles2.get(idx, "")
            m = Chem.MolFromSmiles(s)
            if m:
                product_smiles_list.append(Chem.MolToSmiles(m))
            else:
                product_smiles_list.append(s)

        # Get operator info
        op_idx = rxn.operator
        op = ops_list2[op_idx] if op_idx < len(ops_list2) else None
        op_smarts = getattr(op, "uid", str(op)) if op else "unknown"
        try:
            meta = network2.ops.meta(op_idx)
            rxn_name = meta.get("name", "unnamed")
        except Exception:
            rxn_name = "unnamed"

        coupling_rxns.append({
            "name": rxn_name,
            "smarts": str(op_smarts),
            "reactants": list(reactant_smiles_set),
            "products": product_smiles_list,
        })

if coupling_rxns:
    for i, rxn_info in enumerate(coupling_rxns, 1):
        print(f"\n--- Coupling Reaction {i}: {rxn_info['name']} ---")
        print(f"  Reactants: {' + '.join(rxn_info['reactants'])}")
        print(f"  Products:  {' + '.join(rxn_info['products'])}")
        target_in_products = any(p == canon_target for p in rxn_info["products"])
        if target_in_products:
            print(f"  >>> PRODUCES TARGET! <<<")
else:
    print("\nNo coupling reactions found involving both reactants.")

# ── Also try gen=2 if target not found ────────────────────────────────────────
if not target_found:
    print()
    print("=" * 70)
    print("RETRYING WITH gen=2 (two-step pathways)")
    print("=" * 70)

    network3 = synthetic.generate_network(
        job_name="coupling_experiment_gen2",
        starters={REACTANT_1, REACTANT_2},
        helpers=helpers,
        gen=2,
        direction="forward",
    )

    mols_list3 = list(network3.mols)
    print(f"Network generated: {len(mols_list3)} molecules")

    target_found_gen2 = False
    products_gen2 = set()
    for mol_obj in mols_list3:
        uid = getattr(mol_obj, "uid", None)
        if not uid:
            continue
        smiles_str = str(uid)
        if "*" in smiles_str:
            continue
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in starter_canon or canonical in helper_canon:
            continue
        products_gen2.add(canonical)
        if canonical == canon_target:
            target_found_gen2 = True

    print(f"Total unique products (gen=2): {len(products_gen2)}")
    if target_found_gen2:
        print("\n>>> TARGET FOUND IN gen=2! <<<")
    else:
        print("\nTarget still NOT found in gen=2.")
        # Check for any product with similar MW to target
        target_mw = Descriptors.MolWt(Chem.MolFromSmiles(canon_target))
        close_mw = []
        for canonical in products_gen2:
            mol = Chem.MolFromSmiles(canonical)
            if mol:
                mw = Descriptors.MolWt(mol)
                if abs(mw - target_mw) < 20:
                    close_mw.append((mw, canonical))
        if close_mw:
            close_mw.sort()
            print(f"\nProducts with MW close to target ({target_mw:.1f} ± 20):")
            for mw, s in close_mw:
                print(f"  MW {mw:7.2f}  {s}")
        else:
            print(f"\nNo products with MW close to target ({target_mw:.1f} ± 20)")

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
