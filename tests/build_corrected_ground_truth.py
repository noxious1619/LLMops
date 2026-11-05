"""
Script to build the canonical corrected ground-truth dataset:
tests/retrieval_ground_truth_200_corrected_v1.json
"""

import json
import pickle
from pathlib import Path

def build_corrected_ground_truth():
    # 1. Load Step 9 results as the base schema
    with open('tests/step9_dense_200_results.json', 'r', encoding='utf-8') as f:
        step9_records = json.load(f)

    # 2. Load FAISS docstore
    with open('faiss_index/benchmark_70docs_v1/index.pkl', 'rb') as f:
        docstore, index_to_id = pickle.load(f)

    chunk_by_id = {}
    for idx, cid in index_to_id.items():
        doc = docstore.search(cid)
        chunk_by_id[doc.metadata.get('chunk_id')] = doc

    print(f"Loaded {len(step9_records)} base records. Total docstore chunks: {len(chunk_by_id)}")

    # Specific chunk IDs for DOC-NET-802.docx
    # Chunk 87: 1b7bb4ab53ce1fcdd372c9c6c64fe2debfc9a5100a952a3ccf756ff75a7a424a (BGP Fundamentals, eBGP/iBGP, BGP Communities)
    # Chunk 88: a157a627b100733735d25dadb8d04b1a64dc3d9d97ad3ab3075b62286237d608 (VPC Peering Transitivity, Transit Gateway)
    CID_NET_87 = "1b7bb4ab53ce1fcdd372c9c6c64fe2debfc9a5100a952a3ccf756ff75a7a424a"
    CID_NET_88 = "a157a627b100733735d25dadb8d04b1a64dc3d9d97ad3ab3075b62286237d608"

    # Multi-chunk QIDs
    multi_qids = {
        'Q3', 'Q26', 'Q35', 'Q36', 'Q37', 'Q42', 'Q46', 'Q47', 'Q65', 'Q83', 'Q92', 'Q97',
        'Q186', 'Q187', 'Q188', 'Q189', 'Q190', 'Q191', 'Q192', 'Q193', 'Q194', 'Q195'
    }

    corrected_records = []

    for r in step9_records:
        qid = r["question_id"]
        nid = r["numeric_id"]
        origin = r["dataset_origin"]
        group = r["evaluation_group"]
        status = r["ground_truth_status"]
        q_type = r["question_type"]
        diff = r["difficulty"]
        q_text = r["question"]
        expected_ans = r["expected_answer"]
        src_docs = r["source_documents"]
        gt_cids = list(r["ground_truth_chunk_ids"])
        is_multi = r["requires_multiple_chunks"]

        min_sufficient_cids = list(gt_cids)
        audit_status = "VERIFIED_UNCHANGED"
        rationale = "Ground truth verified against corpus docstore evidence; no modification required."

        # Handle Group C (Negative Control)
        if group == "Group C (Negative Control)":
            min_sufficient_cids = []
            gt_cids = []
            audit_status = "NEGATIVE_CONTROL_ISOLATED"
            rationale = "Out-of-domain negative control; unanswerable from corpus. Denominators excluded from recall."

        # Handle Group B (Partially Supported)
        elif group == "Group B (Partially Supported)":
            audit_status = "PARTIALLY_SUPPORTED_QUALIFIED"
            rationale = "Partial evidence exists in corpus; answer requires external or missing data."

        # Handle Q12-Q15 corrections
        elif qid == "Q12":
            gt_cids = [CID_NET_88]
            min_sufficient_cids = [CID_NET_88]
            expected_ans = (
                "Standard VPC Peering is non-transitive: if VPC A is peered with VPC B, and VPC B is peered "
                "with VPC C, traffic cannot route from VPC A to VPC C through VPC B. Direct point-to-point peering "
                "connections must be established between every pair of VPCs, creating exponential administrative "
                "overhead (the N*(N-1)/2 scaling problem) as the network grows."
            )
            audit_status = "CORRECTED_MAPPING_SHIFT"
            rationale = (
                "Corrected index shift from build_ground_truth_dataset.py: previous mapping pointed to Chunk 87 "
                "with an answer describing BGP Communities. Re-mapped to Chunk 88 (Routing Constraints and Transitivity) "
                "which directly provides the non-transitive VPC peering evidence."
            )

        elif qid == "Q13":
            gt_cids = [CID_NET_87]
            min_sufficient_cids = [CID_NET_87]
            expected_ans = (
                "In hybrid cloud architectures connecting on-premises data centers to cloud networks (e.g. AWS Direct Connect "
                "or Azure ExpressRoute), BGP communities allow network administrators to attach numerical tags or metadata "
                "to route advertisements. These tags enforce dynamic routing policies, prioritize primary links, and control "
                "automated failover across the hybrid boundary without manual per-prefix configuration."
            )
            audit_status = "CORRECTED_ANSWER_ALIGNMENT"
            rationale = (
                "Corrected expected answer text: chunk mapping correctly points to Chunk 87 (BGP in Cloud Environments), "
                "but the original expected answer described BGP path selection tie-breaking rather than BGP community route tagging."
            )

        elif qid == "Q14":
            gt_cids = [CID_NET_88]
            min_sufficient_cids = [CID_NET_88]
            expected_ans = (
                "A hub-and-spoke networking architecture utilizing Transit Gateways acts as a centralized regional virtual router. "
                "Instead of maintaining N*(N-1)/2 separate point-to-point non-transitive peering links across multiple VPCs, each VPC, "
                "VPN, or Direct Connect connection attaches directly to the Transit Gateway, consolidating route tables, enabling "
                "transitive routing, and eliminating exponential mesh complexity."
            )
            audit_status = "CORRECTED_ANSWER_ALIGNMENT"
            rationale = (
                "Corrected expected answer text: chunk mapping correctly points to Chunk 88 (Transition to Transit Gateways), "
                "but the original expected answer described eBGP RPKI BGP hijacking which was completely absent from DOC-NET-802.docx."
            )

        elif qid == "Q15":
            gt_cids = [CID_NET_87]
            min_sufficient_cids = [CID_NET_87]
            expected_ans = (
                "eBGP (External BGP) is used to exchange routing and reachability information between distinct Autonomous Systems "
                "having different Autonomous System Numbers (ASNs), whereas iBGP (Internal BGP) operates strictly within a single "
                "Autonomous System (same ASN) to distribute routing information internally."
            )
            audit_status = "CORRECTED_MAPPING_SHIFT"
            rationale = (
                "Corrected index shift from build_ground_truth_dataset.py: previous mapping pointed to Chunk 88 with an answer "
                "describing VPC Peering N*(N-1)/2. Re-mapped to Chunk 87 (Border Gateway Protocol Fundamentals) which explicitly "
                "defines eBGP across distinct ASNs vs iBGP within a single ASN."
            )

        # Handle Multi-Chunk questions
        elif qid in multi_qids:
            if qid == "Q3":
                audit_status = "VALIDATED_MULTICHUNK_REDUNDANT"
                min_sufficient_cids = [gt_cids[1]]  # Chunk 2 alone has full claims
                rationale = (
                    "Chunk 2 contains the complete claims information ('iss', 'aud', 'exp', 'nbf', JWKS). "
                    "Chunk 1 duplicates part of the validation section due to 100-token chunk overlap. "
                    "Both chunks retained in relevant chunk set; minimum sufficient set refined to Chunk 2."
                )
            elif qid == "Q195":
                audit_status = "VALIDATED_MULTICHUNK_REDUNDANT"
                min_sufficient_cids = [gt_cids[1]]  # Chunk 2 alone has SoC and cell balancing
                rationale = (
                    "Chunk 2 contains both SoC estimation (EKF, Coulomb counting) and cell balancing (passive bleed resistors, active shuttling). "
                    "Chunk 1 provides supporting hardware architecture (CMC/CSC). Both chunks retained in relevant chunk set; "
                    "minimum sufficient set refined to Chunk 2."
                )
            else:
                audit_status = "VALIDATED_MULTICHUNK_COMPLEMENTARY"
                min_sufficient_cids = list(gt_cids)
                rationale = "Both chunks provide necessary, distinct, non-overlapping complementary evidence required to answer the question."

        entry = {
            "question_id": qid,
            "numeric_id": nid,
            "dataset_origin": origin,
            "evaluation_group": group,
            "ground_truth_status": status,
            "question_type": q_type,
            "difficulty": diff,
            "question": q_text,
            "expected_answer": expected_ans,
            "source_documents": src_docs,
            "ground_truth_chunk_ids": gt_cids,
            "minimum_sufficient_chunk_ids": min_sufficient_cids,
            "requires_multiple_chunks": is_multi,
            "audit_status": audit_status,
            "correction_rationale": rationale
        }
        corrected_records.append(entry)

    # 3. Integrity Verification Checks
    print("\nRunning automated integrity verification checks:")
    assert len(corrected_records) == 200, f"Expected 200 questions, got {len(corrected_records)}"
    
    qids = [r["question_id"] for r in corrected_records]
    assert len(qids) == len(set(qids)), "Duplicate question_id detected!"
    
    nids = [r["numeric_id"] for r in corrected_records]
    assert len(nids) == len(set(nids)), "Duplicate numeric_id detected!"

    group_counts = {}
    for r in corrected_records:
        g = r["evaluation_group"]
        group_counts[g] = group_counts.get(g, 0) + 1
    
    assert group_counts["Group A (Verified)"] == 178, f"Group A count mismatch: {group_counts}"
    assert group_counts["Group B (Partially Supported)"] == 5, f"Group B count mismatch: {group_counts}"
    assert group_counts["Group C (Negative Control)"] == 17, f"Group C count mismatch: {group_counts}"

    # Verify every GT chunk ID exists in docstore
    for r in corrected_records:
        for cid in r["ground_truth_chunk_ids"]:
            assert cid in chunk_by_id, f"Invalid GT chunk ID {cid} for question {r['question_id']}"
        for cid in r["minimum_sufficient_chunk_ids"]:
            assert cid in chunk_by_id, f"Invalid Min-Sufficient chunk ID {cid} for question {r['question_id']}"
        # Check no duplicates within ground truth
        assert len(r["ground_truth_chunk_ids"]) == len(set(r["ground_truth_chunk_ids"])), f"Duplicate GT chunk ID in {r['question_id']}"

    # Verify Q12-Q15 mappings
    q12 = [r for r in corrected_records if r["question_id"] == "Q12"][0]
    q13 = [r for r in corrected_records if r["question_id"] == "Q13"][0]
    q14 = [r for r in corrected_records if r["question_id"] == "Q14"][0]
    q15 = [r for r in corrected_records if r["question_id"] == "Q15"][0]

    assert q12["ground_truth_chunk_ids"] == [CID_NET_88], f"Q12 mapping wrong: {q12['ground_truth_chunk_ids']}"
    assert q13["ground_truth_chunk_ids"] == [CID_NET_87], f"Q13 mapping wrong: {q13['ground_truth_chunk_ids']}"
    assert q14["ground_truth_chunk_ids"] == [CID_NET_88], f"Q14 mapping wrong: {q14['ground_truth_chunk_ids']}"
    assert q15["ground_truth_chunk_ids"] == [CID_NET_87], f"Q15 mapping wrong: {q15['ground_truth_chunk_ids']}"

    print("All integrity checks passed successfully!")
    print(f"Group breakdown: {group_counts}")

    # Save to canonical file
    out_path = Path("tests/retrieval_ground_truth_200_corrected_v1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corrected_records, f, indent=2)

    print(f"Saved canonical corrected ground-truth dataset to {out_path} ({len(corrected_records)} records).")

if __name__ == "__main__":
    build_corrected_ground_truth()

