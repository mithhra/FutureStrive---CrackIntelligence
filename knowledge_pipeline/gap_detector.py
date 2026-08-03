"""
knowledge_pipeline/gap_detector.py
------------------------------------
Knowledge coverage gap analysis for the Construction Intelligence Platform.

Analyses the FAISS chunk metadata and flags domains that are sparsely covered.
Generates a structured gap report and prints recommended document types.
"""

import json
from pathlib import Path
from collections import Counter

BASE_DIR   = Path(__file__).resolve().parent.parent
META_FILE  = BASE_DIR / "vector_store" / "chunk_metadata.json"
META_INV   = BASE_DIR / "knowledge_pipeline" / "metadata.json"
REPORT_OUT = BASE_DIR / "knowledge_pipeline" / "gap_report.json"

MINIMUM_CHUNKS = 50     # domains below this are flagged as sparse

DOMAIN_RECOMMENDATIONS = {
    "quality": {
        "missing_topics": ["ISO 9001 construction", "third-party inspection", "NCR management"],
        "suggested_sources": ["CPWD QA Manual 2022", "IS 2974", "ISO 9001:2015 overview documents"],
        "suggested_doc_types": ["manual", "checklist", "standard"]
    },
    "concrete": {
        "missing_topics": ["self-compacting concrete", "high-performance concrete", "fly ash concrete"],
        "suggested_sources": ["IS 10262:2019 Mix Design", "ACI 211 (purchased)", "NPTEL advanced concrete lectures"],
        "suggested_doc_types": ["standard", "lecture_notes", "report"]
    },
    "cracks": {
        "missing_topics": ["plastic shrinkage cracks", "thermal cracking", "crack width measurement"],
        "suggested_sources": ["IS 456 Section 35", "CPWD Handbook on RCC Repair", "FHWA shrinkage crack report"],
        "suggested_doc_types": ["standard", "manual", "report"]
    },
    "honeycombing": {
        "missing_topics": ["vibration techniques", "formwork permeability", "surface finishing"],
        "suggested_sources": ["CPWD Specifications Vol.1 Chapter 4", "FHWA concrete defects guide"],
        "suggested_doc_types": ["manual", "specification"]
    },
    "safety": {
        "missing_topics": ["confined space entry", "lifting operations", "scaffolding erection"],
        "suggested_sources": ["IS 3696 (Scaffolding Safety)", "NBC Part 7 Safety", "OSHA 1926 Subpart R"],
        "suggested_doc_types": ["standard", "regulation", "checklist"]
    },
    "boq": {
        "missing_topics": ["NMM (New Method of Measurement)", "BOQ software integration", "variation orders"],
        "suggested_sources": ["CPWD DSR 2023", "IS 1200 (Method of Measurement)", "RICS NRM guidance"],
        "suggested_doc_types": ["standard", "specification", "guidance"]
    },
    "material_management": {
        "missing_topics": ["material traceability", "batch testing", "supplier qualification"],
        "suggested_sources": ["IS 269 (Cement)", "IS 383 (Aggregates)", "CPWD stores manual"],
        "suggested_doc_types": ["standard", "manual"]
    },
    "cost": {
        "missing_topics": ["Earned Value Management (EVM)", "contingency planning", "escalation clauses"],
        "suggested_sources": ["PMI PMBOK (open access summary)", "RICS cost management guide", "CPWD rate analysis"],
        "suggested_doc_types": ["guidance", "report", "manual"]
    },
    "delays": {
        "missing_topics": ["delay damages", "acceleration claims", "time-impact analysis"],
        "suggested_sources": ["SCL Delay Protocol (open version)", "NPTEL CPM/PERT lectures", "FIDIC delay clauses summary"],
        "suggested_doc_types": ["protocol", "lecture_notes", "guidance"]
    },
    "standards": {
        "missing_topics": ["IS 456 full text", "NBC Part 6", "IS 875 wind loads"],
        "suggested_sources": ["BIS online portal (free registration)", "National Building Code 2016"],
        "suggested_doc_types": ["standard", "code"]
    },
    "inspection": {
        "missing_topics": ["NDT methods", "rebound hammer test", "ultrasonic pulse velocity"],
        "suggested_sources": ["IS 13311 (NDT)", "ASTM C805 (rebound hammer)", "CPWD inspection checklists"],
        "suggested_doc_types": ["standard", "checklist", "manual"]
    },
    "reports": {
        "missing_topics": ["case studies", "failure analysis reports", "post-construction review"],
        "suggested_sources": ["IIT technical reports", "FHWA failure analysis documents", "ResearchGate open papers"],
        "suggested_doc_types": ["report", "case_study", "technical_note"]
    }
}


def run():
    if not META_FILE.exists():
        print("ERROR: vector_store/chunk_metadata.json not found.")
        print("Run processor.py and embedder.py first.")
        return

    with open(META_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(META_INV, "r", encoding="utf-8") as f:
        inventory = json.load(f)

    domain_counts = Counter(c["domain"] for c in chunks)
    all_domains   = inventory["domains"]

    print("\nConstruction Intelligence Knowledge Base Gap Analysis")
    print("=" * 60)
    print(f"{'Domain':<22} {'Chunks':>7}  {'Status'}")
    print("-" * 60)

    gap_report = {
        "total_chunks": len(chunks),
        "domain_coverage": {},
        "sparse_domains": [],
        "missing_domains": [],
        "recommendations": {}
    }

    for domain in sorted(all_domains):
        count = domain_counts.get(domain, 0)
        gap_report["domain_coverage"][domain] = count

        if count == 0:
            status = "MISSING"
            gap_report["missing_domains"].append(domain)
        elif count < MINIMUM_CHUNKS:
            status = f"SPARSE (<{MINIMUM_CHUNKS})"
            gap_report["sparse_domains"].append(domain)
        else:
            status = "OK"

        print(f"  {domain:<20} {count:>7}  {status}")

        if count < MINIMUM_CHUNKS:
            gap_report["recommendations"][domain] = DOMAIN_RECOMMENDATIONS.get(domain, {
                "missing_topics": ["No specific recommendations available"],
                "suggested_sources": ["Search NPTEL, CPWD, or BIS portal"],
                "suggested_doc_types": ["pdf", "manual", "standard"]
            })

    print("-" * 60)
    print(f"  {'TOTAL':<20} {len(chunks):>7}")
    print(f"\nSparse domains  : {len(gap_report['sparse_domains'])}")
    print(f"Missing domains : {len(gap_report['missing_domains'])}")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(gap_report, f, indent=2)

    print(f"\nFull gap report saved to: {REPORT_OUT}")

    if gap_report["recommendations"]:
        print("\nTop recommendations:")
        for domain, recs in list(gap_report["recommendations"].items())[:3]:
            print(f"\n  Domain: {domain}")
            print(f"  Missing topics : {', '.join(recs.get('missing_topics', [])[:2])}")
            print(f"  Add sources    : {', '.join(recs.get('suggested_sources', [])[:2])}")


if __name__ == "__main__":
    run()
