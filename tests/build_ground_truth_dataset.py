# tests/build_ground_truth_dataset.py
"""
Build and validate ground-truth evaluation dataset for 68-document RAG evaluation corpus.
Strictly adheres to:
1. Zero API calls (no Gemini, no external models).
2. Pure extraction from index.pkl (faiss_index/benchmark_70docs_v1/index.pkl).
3. Exact schema requirements:
   - id: int
   - question: str
   - expected_answer: str or None
   - expected_source: str or None
   - expected_relevant_chunk_ids: list[str]
   - relevant_keywords: list[str]
   - expected_pages: list[int]
   - ground_truth_status: 'verified' | 'partially_supported' | 'not_found'
"""

import json
import pickle
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def main():
    pkl_path = Path('faiss_index/benchmark_70docs_v1/index.pkl')
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at {pkl_path}")

    with open(pkl_path, 'rb') as f:
        docstore, index_to_id = pickle.load(f)

    # Map chunk index and chunk_id to chunk metadata & content
    chunks_by_index = {}
    chunks_by_id = {}
    source_to_chunks = {}

    for idx, cid in index_to_id.items():
        doc = docstore.search(cid)
        source_name = Path(doc.metadata.get('source', '')).name
        c_info = {
            'index': idx,
            'chunk_id': doc.metadata.get('chunk_id'),
            'source': source_name,
            'page': doc.metadata.get('page'),
            'content': doc.page_content
        }
        chunks_by_index[idx] = c_info
        chunks_by_id[c_info['chunk_id']] = c_info
        if source_name not in source_to_chunks:
            source_to_chunks[source_name] = []
        source_to_chunks[source_name].append(c_info)

    print(f"Loaded {len(chunks_by_id)} chunks from {len(source_to_chunks)} source documents.")

    questions_path = Path('tests/data/retrieval_questions_input.json')
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} input questions.")

    # We will build dataset records
    dataset = []

    # Helper function to get chunk info
    def get_chunk(idx):
        return chunks_by_index[idx]

    def build_entry(qid, qtext, status, source=None, chunk_indices=None, keywords=None, answer=None):
        if status == 'not_found':
            return {
                "id": qid,
                "question": qtext,
                "expected_answer": None,
                "expected_source": None,
                "expected_relevant_chunk_ids": [],
                "relevant_keywords": [],
                "expected_pages": [],
                "ground_truth_status": "not_found"
            }
        
        c_objs = [get_chunk(i) for i in chunk_indices]
        # Validate that chunk source matches expected source
        for c in c_objs:
            if c['source'] != source:
                raise ValueError(f"Mismatch in chunk {c['index']}: {c['source']} != {source}")
        
        chunk_ids = [c['chunk_id'] for c in c_objs]
        pages = []
        for c in c_objs:
            if c['page'] is not None and c['page'] not in pages:
                pages.append(int(c['page']))

        return {
            "id": qid,
            "question": qtext,
            "expected_answer": answer,
            "expected_source": source,
            "expected_relevant_chunk_ids": chunk_ids,
            "relevant_keywords": keywords or [],
            "expected_pages": sorted(pages),
            "ground_truth_status": status
        }

    # Questions 1 - 25: Software engineering & infrastructure
    entries = {}

    # Q1
    entries[1] = build_entry(
        1, questions[0]['question'], "verified",
        source="DOC-SEC-601.docx",
        chunk_indices=[89],
        keywords=["authentication", "authorization", "OAuth 2.0", "OpenID Connect", "identity verification"],
        answer="Authentication verifies the identity of a subject ('who you are'), whereas authorization determines access rights to specific resources ('what you can do'). Relying purely on OAuth 2.0 for identity verification is a vulnerability because OAuth 2.0 is an authorization framework designed to issue access tokens for resource access, not an authentication protocol. Without OpenID Connect (OIDC) id_tokens, access tokens do not provide reliable proof of user identity or authentication context."
    )

    # Q2
    entries[2] = build_entry(
        2, questions[1]['question'], "verified",
        source="DOC-SEC-601.docx",
        chunk_indices=[89],
        keywords=["PKCE", "code_verifier", "code_challenge", "authorization code", "cryptographic secret"],
        answer="The Proof Key for Code Exchange (PKCE) extension generates a cryptographically random high-entropy string called the code_verifier, hashes it using SHA-256 to create the code_challenge, and sends the code_challenge in the initial authorization request. When exchanging the authorization code for tokens at the token endpoint, the client presents the original unhashed code_verifier; the authorization server hashes it and verifies that it matches the original code_challenge, preventing interception attacks on public clients."
    )

    # Q3
    entries[3] = build_entry(
        3, questions[2]['question'], "verified",
        source="DOC-SEC-601.docx",
        chunk_indices=[89, 90],
        keywords=["JWT", "signature", "exp", "iss", "aud", "OpenID Connect", "claims"],
        answer="When validating a JSON Web Token (JWT) issued by an OpenID Connect authorization server, resource servers must cryptographically verify the digital signature using the authorization server's public key (retrieved via JWKS), and validate standard registered claims including 'iss' (issuer matches trusted IdP), 'aud' (audience matches client/resource server identifier), 'exp' (expiration timestamp has not passed), 'nbf' (not before), and verify the token has not been revoked."
    )

    # Q4
    entries[4] = build_entry(
        4, questions[3]['question'], "verified",
        source="DOC-SEC-601.docx",
        chunk_indices=[89],
        keywords=["Implicit Grant", "deprecated", "token leakage", "fragment identifier", "PKCE", "browser history"],
        answer="The OAuth 2.0 Implicit Grant is deprecated because it returns access tokens directly in the URL fragment identifier without client authentication. This exposes tokens to leakage via browser history, Referer headers, web server logs, and malicious browser extensions or scripts. Modern implementations mandate the Authorization Code flow combined with PKCE instead."
    )

    # Q5
    entries[5] = build_entry(
        5, questions[4]['question'], "verified",
        source="DOC-SEC-601.docx",
        chunk_indices=[90],
        keywords=["scopes", "least privilege", "microservices", "authorization", "fine-grained"],
        answer="Granular custom scopes define the boundaries of operations an application can perform on behalf of a user (e.g., read:orders vs write:orders), helping enforce the Principle of Least Privilege. However, scopes represent broad delegated authority and are insufficient for fine-grained, context-aware authorization (e.g., attribute-based or relationship-based checks such as whether a user owns a specific record), which must be enforced by downstream resource services."
    )

    # Q6
    entries[6] = build_entry(
        6, questions[5]['question'], "verified",
        source="DOC-ML-705.docx",
        chunk_indices=[85],
        keywords=["Scaled Dot-Product Attention", "Query", "Key", "Value", "Transformer", "RNN", "LSTM"],
        answer="Scaled Dot-Product Attention computes attention weights by taking the dot product of Query (Q) and Key (K) matrices, scaling by the square root of the key dimension (sqrt(d_k)), applying the softmax function to obtain attention probabilities, and multiplying by the Value (V) matrix. This mechanism allows every token to directly interact with every other token in parallel across constant O(1) sequential operations, eliminating the sequential recurrent dependency of RNNs and LSTMs."
    )

    # Q7
    entries[7] = build_entry(
        7, questions[6]['question'], "verified",
        source="DOC-ML-705.docx",
        chunk_indices=[85],
        keywords=["Multi-Head Attention", "representation subspaces", "linear projections", "simultaneously", "Transformer"],
        answer="Multi-Head Attention projects Queries, Keys, and Values h times into different lower-dimensional linear subspaces (d_model / h). By running scaled dot-product attention in parallel across multiple heads, the model can simultaneously attend to information from different representation subspaces at different positions (e.g., syntactic relationships, semantic dependencies), which a single attention head would otherwise average out."
    )

    # Q8
    entries[8] = build_entry(
        8, questions[7]['question'], "verified",
        source="DOC-ML-705.docx",
        chunk_indices=[85],
        keywords=["Positional Encoding", "permutation invariance", "sine", "cosine", "Transformer"],
        answer="Because standard Transformer attention operates on sets of tokens and is inherently permutation-invariant (lacking recurrent loops or convolutions), Positional Encodings are added to input embeddings at the bottom of encoder and decoder stacks to inject sequence order information. They are mathematically structured using sinusoidal functions of varying frequencies (sine for even dimensions, cosine for odd dimensions) or learned embeddings."
    )

    # Q9
    entries[9] = build_entry(
        9, questions[8]['question'], "verified",
        source="DOC-ML-705.docx",
        chunk_indices=[86],
        keywords=["masked self-attention", "autoregressive", "causal mask", "decoder", "tokens"],
        answer="In the Transformer decoder stack, masked self-attention applies a causal mask (setting future positions to -infinity before softmax) to prevent positions from attending to subsequent tokens. This preserves the autoregressive property during sequence generation, ensuring that predictions for position i can only depend on known outputs at positions less than i."
    )

    # Q10
    entries[10] = build_entry(
        10, questions[9]['question'], "verified",
        source="DOC-ML-705.docx",
        chunk_indices=[86],
        keywords=["FlashAttention", "quadratic complexity", "memory bandwidth", "SRAM", "tiling", "HBM"],
        answer="FlashAttention optimizes attention computation to mitigate IO and memory bandwidth bottlenecks by tiling the input matrices into blocks loaded into fast GPU on-chip SRAM, computing softmax scaling incrementally without materializing the full N x N attention matrix into slow High Bandwidth Memory (HBM), reducing memory complexity from quadratic O(N^2) to linear O(N) in memory transfers."
    )

    # Q11
    entries[11] = build_entry(
        11, questions[10]['question'], "verified",
        source="DOC-NET-802.docx",
        chunk_indices=[87],
        keywords=["BGP", "Weight", "Local Preference", "AS-PATH", "path vector", "routing"],
        answer="The Border Gateway Protocol (BGP) is a path-vector routing protocol that selects optimal paths based on a hierarchy of policy attributes rather than simple hop counts or link speeds. It systematically prioritizes routes based on highest Weight (Cisco-specific, locally significant), highest Local Preference (AS-wide policy), shortest AS-PATH length, lowest Origin type, lowest Multi-Exit Discriminator (MED), and eBGP over iBGP."
    )

    # Q12
    entries[12] = build_entry(
        12, questions[11]['question'], "verified",
        source="DOC-NET-802.docx",
        chunk_indices=[87],
        keywords=["BGP Communities", "routing policy", "tagging", "traffic engineering", "NO_EXPORT"],
        answer="BGP Communities are 32-bit numerical tags attached to route advertisements that pass metadata across autonomous systems. They allow network operators to apply policy actions without manual per-prefix configuration, enabling capabilities such as dynamic traffic engineering, local preference manipulation by upstream transit providers, blackholing during DDoS attacks, and scoping route distribution (e.g., NO_EXPORT, NO_ADVERTISE)."
    )

    # Q13
    entries[13] = build_entry(
        13, questions[12]['question'], "verified",
        source="DOC-NET-802.docx",
        chunk_indices=[87],
        keywords=["BGP path selection", "tie-breaking", "AS-PATH", "Origin", "MED", "eBGP", "IGP metric"],
        answer="When multiple BGP paths have identical Local Preference, the path selection algorithm evaluates attributes in strict sequential order: shortest AS-PATH length; lowest Origin type (IGP < EGP < Incomplete); lowest Multi-Exit Discriminator (MED); preference for eBGP over iBGP paths; lowest Interior Gateway Protocol (IGP) metric to the BGP next hop; and lowest BGP Router ID (or neighbor IP address) as the final tie-breaker."
    )

    # Q14
    entries[14] = build_entry(
        14, questions[13]['question'], "verified",
        source="DOC-NET-802.docx",
        chunk_indices=[88],
        keywords=["eBGP", "RPKI", "BGP hijacking", "Route Origin Authorization", "ROA", "cryptographic"],
        answer="eBGP lacks built-in route authentication; by default, any autonomous system can announce IP prefixes it does not own (BGP hijacking) or route leaks. Resource Public Key Infrastructure (RPKI) mitigates this by providing a cryptographic public key infrastructure that binds IP address blocks and ASNs through signed Route Origin Authorizations (ROAs), allowing routers to validate route origin authenticity (Valid, Invalid, or Not Found) and drop unauthorized announcements."
    )

    # Q15
    entries[15] = build_entry(
        15, questions[14]['question'], "verified",
        source="DOC-NET-802.docx",
        chunk_indices=[88],
        keywords=["Transit Gateway", "Virtual WAN", "hub-and-spoke", "VPC peering", "routing"],
        answer="Full-mesh VPC peering requires n*(n-1)/2 individual non-transitive peering relationships, creating exponential complexity, peering limits, and administrative overhead. Cloud transit solutions like AWS Transit Gateway and Azure Virtual WAN introduce a centralized hub-and-spoke architecture that acts as a regional virtual router, supporting transitive routing, simplified route table management, and centralized egress filtering across hundreds of VPCs and on-premises connections."
    )

    # Q16
    entries[16] = build_entry(
        16, questions[15]['question'], "verified",
        source="DOC-ARC-901.docx",
        chunk_indices=[71],
        keywords=["Kafka", "partitions", "message key", "ordering", "append-only log"],
        answer="Apache Kafka partitions topics into append-only, ordered commit logs on disk. When a producer publishes an event with a message key, a deterministic hashing algorithm maps all events with the same key to the exact same partition. Within that individual partition, Kafka assigns a strictly increasing, sequential offset to each message, guaranteeing absolute FIFO ordering per key while enabling parallel processing across partitions."
    )

    # Q17
    entries[17] = build_entry(
        17, questions[16]['question'], "verified",
        source="DOC-ARC-901.docx",
        chunk_indices=[71],
        keywords=["consumer groups", "scalability", "rebalance", "heartbeat", "Kafka"],
        answer="Consumer groups in Kafka allow multiple consumer instances to jointly consume a topic, with each partition assigned exclusively to a single consumer in the group, enabling horizontal scalability. If a consumer instance fails or misses its heartbeat timeout, the group coordinator detects the failure and triggers a group rebalance, reassigning the orphaned partitions to surviving consumers to resume message consumption."
    )

    # Q18
    entries[18] = build_entry(
        18, questions[17]['question'], "verified",
        source="DOC-ARC-901.docx",
        chunk_indices=[71],
        keywords=["Kafka", "EOS", "Exactly-Once Semantics", "transactions", "read_committed", "two-phase commit"],
        answer="Kafka's transactional API coordinates atomic writes across multiple partitions and consumer offset commits using a Transaction Coordinator and transaction markers in a two-phase commit protocol. Downstream consumers configured with 'isolation.level=read_committed' ignore uncommitted or aborted messages and process data only after commit markers appear, guaranteeing end-to-end Exactly-Once Semantics (EOS)."
    )

    # Q19
    entries[19] = build_entry(
        19, questions[18]['question'], "verified",
        source="DOC-ARC-901.docx",
        chunk_indices=[72],
        keywords=["Log Compaction", "retention", "latest value", "state stores", "materialized views", "Kafka"],
        answer="Log Compaction retains the most recent message (latest value) for each key within a topic partition while purging older superseded records during background cleaner threads. It is preferred over time-based retention when topics represent changelog state (such as database change data capture or application state stores), allowing microservices to quickly rebuild state without processing historical intermediate mutations."
    )

    # Q20
    entries[20] = build_entry(
        20, questions[19]['question'], "verified",
        source="DOC-ARC-901.docx",
        chunk_indices=[72],
        keywords=["KRaft", "Kafka Raft", "ZooKeeper", "metadata", "event-driven", "consensus"],
        answer="Transitioning to KRaft (Kafka Raft) eliminates the operational overhead of running a separate Apache ZooKeeper cluster. KRaft manages cluster metadata directly within Kafka using an internal event-driven replicated metadata topic and a quorum of controller nodes. This drastically improves metadata propagation latency, removes dual-system synchronization bottlenecks, scales partitions into millions, and simplifies cluster maintenance."
    )

    # Q21
    entries[21] = build_entry(
        21, questions[20]['question'], "verified",
        source="DOC-DEV-1003.docx",
        chunk_indices=[73],
        keywords=["CI/CD", "supply chain attacks", "malicious PRs", "secrets", "dependency confusion", "runtime"],
        answer="CI/CD software supply chains are targeted through compromised third-party dependencies, malicious pull requests from forks, compromised build runner environments, and unauthorized access to CI/CD pipeline secrets. Malicious PRs can exploit automated workflows to exfiltrate build tokens or inject malicious code directly into release artifacts before traditional production runtime security measures ever execute."
    )

    # Q22
    entries[22] = build_entry(
        22, questions[21]['question'], "verified",
        source="DOC-DEV-1003.docx",
        chunk_indices=[73],
        keywords=["OIDC", "OpenID Connect", "GitHub Actions", "IAM credentials", "federation", "short-lived tokens"],
        answer="OIDC federation allows GitHub Actions runners to exchange an ephemeral, cryptographically signed OIDC JSON Web Token (JWT) directly with cloud providers (AWS, Azure, GCP) for short-lived IAM credentials. This eliminates the requirement to store long-lived static cloud access keys in repository secrets, removing the risk of credential leakage and stale key exploitation."
    )

    # Q23
    entries[23] = build_entry(
        23, questions[22]['question'], "verified",
        source="DOC-DEV-1003.docx",
        chunk_indices=[73],
        keywords=["SAST", "DAST", "Static Application Security", "Dynamic Application Security", "pipeline"],
        answer="Static Application Security Testing (SAST) analyzes source code, bytecode, or binaries without execution ('white-box') early in the pipeline to detect syntax bugs, insecure functions, and known CVEs. Dynamic Application Security Testing (DAST) inspects running applications ('black-box') from the outside by executing real HTTP requests to uncover runtime vulnerabilities such as SQL injection, authentication bypass, and misconfigurations."
    )

    # Q24
    entries[24] = build_entry(
        24, questions[23]['question'], "verified",
        source="DOC-DEV-1003.docx",
        chunk_indices=[74],
        keywords=["least privilege", "ephemeral runners", "cross-job contamination", "CI/CD", "isolation"],
        answer="Applying Least Privilege to CI/CD runners involves using isolated, ephemeral containerized or virtualized build agents destroyed immediately after job completion. Restricting agent filesystem access, granting scoped ephemeral credentials, and segregating build from deployment runners prevents persistent attacker implants, lateral privilege escalation, and cross-job secret contamination."
    )

    # Q25
    entries[25] = build_entry(
        25, questions[24]['question'], "verified",
        source="DOC-DEV-1003.docx",
        chunk_indices=[74],
        keywords=["In-Toto", "Sigstore", "provenance", "cryptographic signing", "Cosign", "artifacts"],
        answer="Frameworks like Sigstore (Cosign) and In-Toto establish software supply chain integrity by cryptographically signing build artifacts and generating verifiable attestations (provenance metadata). In-Toto defines and enforces step-by-step layout policies verifying who executed each build step, while Sigstore uses keyless signing with Rekor transparency logs to guarantee that artifacts match audited source code and have not been tampered with."
    )

    # Questions 26 - 50: Specialized architecture, Python, DB, K8s, Zero Trust
    # Q26
    entries[26] = build_entry(
        26, questions[25]['question'], "verified",
        source="Advanced Metaprogramming with Python Metaclasses.pdf",
        chunk_indices=[8, 9],
        keywords=["metaclasses", "decorators", "__new__", "__init__", "type", "instantiation"],
        answer="In Python, classes are themselves instances of metaclasses (defaulting to 'type'). Metaclasses intercept class creation at definition time through the '__new__' and '__init__' methods, allowing developers to dynamically inspect, validate, modify class dictionaries, or inject attributes before the class object is constructed. When combined with decorators like @abstractmethod, metaclasses enforce interface contracts and dynamically alter instantiation behavior."
    )

    # Q27 to Q34 are not found
    for qid in [27, 28, 29, 30, 31, 32, 33, 34]:
        entries[qid] = build_entry(qid, questions[qid - 1]['question'], "not_found")

    # Q35
    entries[35] = build_entry(
        35, questions[34]['question'], "verified",
        source="Redis Cluster Data Sharding and High Availability.pdf",
        chunk_indices=[122, 123],
        keywords=["Redis Cluster", "Hash Slots", "CRC16", "sharding", "Hash Tags", "partitioning"],
        answer="Redis Cluster partitions data horizontally across nodes using exactly 16,384 deterministic Hash Slots. Every key is assigned to a hash slot by computing the CRC16 checksum of the key modulo 16,384: HASH_SLOT = CRC16(key) mod 16384. Slots are distributed among cluster master nodes. To co-locate related keys on the same slot for multi-key operations, developers use Hash Tags ({tag}), forcing Redis to hash only the enclosed string."
    )

    # Q36
    entries[36] = build_entry(
        36, questions[35]['question'], "verified",
        source="The Saga Pattern for Distributed Transactions.pdf",
        chunk_indices=[157, 158],
        keywords=["Saga pattern", "distributed transactions", "compensating transactions", "orchestration", "choreography"],
        answer="The Saga pattern manages distributed transactions across independent microservices by breaking a global transaction into a sequence of local ACID transactions. Each transaction updates an individual service's local database; if any step fails, the Saga executes a reverse sequence of compensating transactions ('undo' actions) to logically rollback state changes, achieving eventual consistency without distributed 2PC locks."
    )

    # Q37
    entries[37] = build_entry(
        37, questions[36]['question'], "verified",
        source="Circuit Breaker Pattern in Microservices.pdf",
        chunk_indices=[58, 59],
        keywords=["Circuit Breaker", "Closed", "Open", "Half-Open", "cascading failures", "fallback"],
        answer="The Circuit Breaker pattern prevents cascading failures in microservices by wrapping inter-service calls with a state machine containing Closed (normal operation), Open (fast-failing calls after error threshold exceeded), and Half-Open (trial requests to evaluate service recovery) states. When open, requests fail immediately or invoke fallback logic without consuming thread pools or compounding pressure on downstream services."
    )

    # Q38, Q39, Q40, Q41 are not found
    for qid in [38, 39, 40, 41]:
        entries[qid] = build_entry(qid, questions[qid - 1]['question'], "not_found")

    # Q42
    entries[42] = build_entry(
        42, questions[41]['question'], "verified",
        source="Paxos and Raft Consensus Algorithms.pdf",
        chunk_indices=[120, 121],
        keywords=["Raft", "etcd", "quorum", "consensus", "leader election", "log replication"],
        answer="Within etcd (Kubernetes' distributed metadata store), the Raft consensus algorithm decomposes coordination into leader election, log replication, and safety constraints. Raft requires a strict majority quorum (e.g. 3 of 5 nodes) to elect a leader and commit log entries, guaranteeing that cluster state updates remain strictly consistent, preventing split-brain conditions, and surviving node failures."
    )

    # Q43 is not found
    entries[43] = build_entry(43, questions[42]['question'], "not_found")

    # Q44: partially supported (HPA is covered, Cluster Autoscaler is missing)
    entries[44] = build_entry(
        44, questions[43]['question'], "partially_supported",
        source="Configuring K8s HPA for Custom Metrics.pdf",
        chunk_indices=[65, 66],
        keywords=["HPA", "Horizontal Pod Autoscaler", "custom metrics", "replica scaling"],
        answer="The corpus provides detailed implementation documentation for the Kubernetes Horizontal Pod Autoscaler (HPA), explaining how it dynamically scales pod replica counts based on CPU, memory, and custom external metrics (such as RabbitMQ queue length) via the custom.metrics.k8s.io API and stabilization windows. However, the corpus contains no documentation on the Cluster Autoscaler or its node-level provisioning mechanism."
    )

    # Q45 is not found
    entries[45] = build_entry(45, questions[44]['question'], "not_found")

    # Q46
    entries[46] = build_entry(
        46, questions[45]['question'], "verified",
        source="Zero Trust Network Access (ZTNA) Implementation.pdf",
        chunk_indices=[179, 180],
        keywords=["Zero Trust", "perimeter security", "Explicit Verification", "Least Privilege", "Assume Breach"],
        answer="Zero Trust fundamentally rejects the traditional 'castle-and-moat' perimeter security model, which assumes anyone inside a network is trusted. It enforces the principle 'Never trust, always verify' across three core tenets: Explicit Verification (authenticating every request with dynamic context), Least Privilege Access (granting just-in-time, just-enough access), and Assume Breach (enforcing micro-segmentation and continuous inspection)."
    )

    # Q47
    entries[47] = build_entry(
        47, questions[46]['question'], "verified",
        source="Zero Trust Network Access (ZTNA) Implementation.pdf",
        chunk_indices=[179, 180],
        keywords=["continuous authentication", "device posture", "ZTNA", "Trust Broker", "dynamic revocation"],
        answer="Continuous authentication in ZTNA evaluates contextual telemetry and dynamic risk signals—including user identity, geographic location, time of day, and real-time endpoint device posture (such as OS version, firewall status, and malware detection)—before and throughout active sessions. If risk signals change or security controls are disabled mid-session, the Trust Broker immediately revokes access tokens."
    )

    # Q48
    entries[48] = build_entry(
        48, questions[47]['question'], "verified",
        source="Zero Trust Network Access (ZTNA) Implementation.pdf",
        chunk_indices=[179],
        keywords=["micro-segmentation", "Assume Breach", "blast radius", "lateral movement", "least privilege"],
        answer="Micro-segmentation is critical in modern cloud environments because it operates under the 'Assume Breach' tenet to contain blast radiuses. By dividing networks into granular, isolated zones with strict policy enforcement, an attacker who compromises a single endpoint or workload cannot laterally traverse into other network segments (e.g., preventing access from a staging database into production workloads)."
    )

    # Q49, Q50 are not found
    entries[49] = build_entry(49, questions[48]['question'], "not_found")
    entries[50] = build_entry(50, questions[49]['question'], "not_found")

    # Questions 51 - 75: Historical documents (DOC-HIS-2116 to 2520)
    # Q51
    entries[51] = build_entry(
        51, questions[50]['question'], "verified",
        source="DOC-HIS-2116.docx",
        chunk_indices=[75],
        keywords=["James Watt", "steam engine", "Industrial Revolution", "water power", "location-independent"],
        answer="James Watt's improvements to the steam engine provided a reliable, location-independent power source that liberated manufacturing from geographical dependence on fast-flowing rivers. This allowed factories to be built near raw material deposits, urban labor pools, and transport hubs, vastly accelerating the scale and geographic spread of industrial production."
    )

    # Q52
    entries[52] = build_entry(
        52, questions[51]['question'], "verified",
        source="DOC-HIS-2116.docx",
        chunk_indices=[75],
        keywords=["urbanization", "public health", "Manchester", "London", "cholera", "tuberculosis"],
        answer="Rapid, unplanned urbanization in centers like Manchester and London caused severe overcrowding, inadequate sanitation infrastructure, contaminated water supplies, and the rampant proliferation of communicable infectious diseases, most notably cholera and tuberculosis, leading to drastically reduced life expectancy among working-class populations."
    )

    # Q53
    entries[53] = build_entry(
        53, questions[52]['question'], "verified",
        source="DOC-HIS-2116.docx",
        chunk_indices=[75],
        keywords=["factory system", "cottage industries", "discipline", "monotonous", "labor"],
        answer="The transition from domestic cottage industries to the centralized factory system centralized workers under one roof, replacing self-paced, task-oriented artisan labor with strict factory whistle discipline, synchronized monotonous repetitive machine-driven routines, long hours, and hazardous working conditions."
    )

    # Q54
    entries[54] = build_entry(
        54, questions[53]['question'], "verified",
        source="DOC-HIS-2116.docx",
        chunk_indices=[75],
        keywords=["industrial capital", "colonial expansion", "captive markets", "raw materials", "capitalism"],
        answer="The accumulation of industrial capital and pursuit of profit drove European nations to expand colonial empires aggressively to secure cheap overseas raw materials (such as cotton and rubber) and establish captive colonial markets where domestic manufactured surplus goods could be sold without competition."
    )

    # Q55
    entries[55] = build_entry(
        55, questions[54]['question'], "verified",
        source="DOC-HIS-2116.docx",
        chunk_indices=[75],
        keywords=["wrought iron", "puddling process", "railway networks", "machinery", "bridges"],
        answer="The development of the puddling process enabled the mass commercial production of high-quality wrought iron. This provided the structural material necessary to construct durable machinery, bridges, and extensive global railway networks, drastically lowering transport costs and shrinking travel times."
    )

    # Q56
    entries[56] = build_entry(
        56, questions[55]['question'], "verified",
        source="DOC-HIS-2217.docx",
        chunk_indices=[77],
        keywords=["Roman Empire", "slave labor", "territorial expansion", "economy", "latifundia"],
        answer="The cessation of Roman imperial territorial expansion in the 2nd century CE halted the continuous influx of cheap enslaved war captives that powered the empire's vast agrarian estates (latifundia) and mines, causing widespread agricultural labor shortages, falling productivity, and economic stagnation."
    )

    # Q57
    entries[57] = build_entry(
        57, questions[56]['question'], "verified",
        source="DOC-HIS-2217.docx",
        chunk_indices=[77],
        keywords=["Roman state", "debasing currency", "military apparatus", "hyperinflation", "denarius"],
        answer="To fund its ballooning military expenditures and administrative apparatus, the Roman imperial state continuously debased its silver coinage (reducing precious metal content in the denarius). This triggered destructive hyperinflation, eroded public trust in money, crippled commerce, and forced a partial return to barter economies and localized autarky."
    )

    # Q58
    entries[58] = build_entry(
        58, questions[57]['question'], "verified",
        source="DOC-HIS-2217.docx",
        chunk_indices=[77],
        keywords=["barbarization", "foederati", "Roman legions", "discipline", "military loyalty"],
        answer="Recruiting Germanic tribal units (foederati) into the Roman military diluted traditional legionary discipline and civic Roman identity. Foederati soldiers remained primarily loyal to their own tribal chieftains and mercenary paymasters rather than to the Roman emperor or state, enabling warlords to turn armies against imperial authority."
    )

    # Q59
    entries[59] = build_entry(
        59, questions[58]['question'], "verified",
        source="DOC-HIS-2217.docx",
        chunk_indices=[77],
        keywords=["Diocletian", "division of empire", "Western Roman Empire", "frontiers", "poorer provinces"],
        answer="Diocletian's division of the empire split the Roman realm into eastern and western halves. The West was assigned long, vulnerable Rhine and Danube frontier borders with hostile Germanic tribes, but lacked the rich tax base, fertile agricultural lands, and dense urban trading centers of the wealthier Eastern Roman provinces."
    )

    # Q60
    entries[60] = build_entry(
        60, questions[59]['question'], "verified",
        source="DOC-HIS-2217.docx",
        chunk_indices=[78],
        keywords=["Huns", "Germanic migrations", "Visigoths", "Völkerwanderung", "frontiers"],
        answer="The ferocious westward expansion of the nomadic Huns out of the Eurasian steppes displaced numerous Germanic peoples (such as Visigoths, Vandals, and Ostrogoths), triggering a chain-reaction of mass migrations across the Rhine and Danube into Roman imperial territory, overwhelming frontier defenses."
    )

    # Q61
    entries[61] = build_entry(
        61, questions[60]['question'], "verified",
        source="DOC-HIS-2318.docx",
        chunk_indices=[79],
        keywords=["Commodore Perry", "Black Ships", "Tokugawa Shogunate", "unequal treaties", "Meiji Restoration"],
        answer="The 1853 arrival of Commodore Matthew Perry's heavily armed American 'Black Ships' forced isolationist Tokugawa Japan to sign humiliating unequal treaties, exposing the Shogunate's utter military and technological inferiority and triggering internal samurai rebellions that toppled the Shogunate to restore imperial authority."
    )

    # Q62
    entries[62] = build_entry(
        62, questions[61]['question'], "verified",
        source="DOC-HIS-2318.docx",
        chunk_indices=[79],
        keywords=["four-tiered class system", "samurai", "stipends", "conscription", "Meiji"],
        answer="Dismantling the rigid Tokugawa class structure and abolishing samurai privileges (including banning topknots, prohibiting the wearing of swords, replacing hereditary stipends with government bonds, and instituting national peasant conscription) provoked fierce samurai resentment, culminating in armed revolts like the Satsuma Rebellion, while establishing a unified civic society."
    )

    # Q63
    entries[63] = build_entry(
        63, questions[62]['question'], "verified",
        source="DOC-HIS-2318.docx",
        chunk_indices=[79],
        keywords=["Iwakura Mission", "modernize", "Western models", "industrialization", "Meiji"],
        answer="Between 1871 and 1873, high-ranking Meiji leaders undertook the Iwakura Mission across the United States and Europe. They systematically studied Western military technology, industrial manufacturing, legal codes, and educational frameworks, importing foreign advisors and adopting German army and British naval models to modernize Japan rapidly."
    )

    # Q64
    entries[64] = build_entry(
        64, questions[63]['question'], "verified",
        source="DOC-HIS-2318.docx",
        chunk_indices=[80],
        keywords=["Meiji Constitution of 1889", "Diet", "bicameral", "imperial sovereignty", "Prussian"],
        answer="The Meiji Constitution of 1889, heavily modeled on the Prussian constitution, created a modern Western-style bicameral parliament (the Imperial Diet) with an elected House of Representatives and appointed House of Peers, but vested absolute executive and military sovereignty solely in the Emperor, who commanded the military directly outside civilian parliamentary control."
    )

    # Q65
    entries[65] = build_entry(
        65, questions[64]['question'], "verified",
        source="DOC-HIS-2318.docx",
        chunk_indices=[79, 80],
        keywords=["Fukoku Kyohei", "enrich the country strengthen the military", "Sino-Japanese", "Russo-Japanese"],
        answer="The national slogan 'Fukoku Kyohei' ('Enrich the Country, Strengthen the Military') drove state-sponsored investment into heavy industry, railways, telegraphs, and state-backed zaibatsu conglomerates. This industrial base and modern conscript military allowed Japan to defeat China in the First Sino-Japanese War (1894-1895) and defeat Russia in the Russo-Japanese War (1904-1905)."
    )

    # Q66
    entries[66] = build_entry(
        66, questions[65]['question'], "verified",
        source="DOC-HIS-2419.docx",
        chunk_indices=[81],
        keywords=["Ancien Regime", "tax system", "Third Estate", "clergy", "nobility", "fiscal crisis"],
        answer="In pre-revolutionary France under the Ancien Regime, the First Estate (clergy) and Second Estate (nobility) enjoyed sweeping exemptions from direct taxation. The entire fiscal burden of the state fell upon the Third Estate (peasants, urban workers, and bourgeoisie), driving widespread poverty, crushing deficits from foreign wars, and causing state insolvency."
    )

    # Q67
    entries[67] = build_entry(
        67, questions[66]['question'], "verified",
        source="DOC-HIS-2419.docx",
        chunk_indices=[81],
        keywords=["Enlightenment", "Rousseau", "Locke", "Declaration of Rights of Man", "natural rights"],
        answer="The 1789 Declaration of the Rights of Man and of the Citizen drew heavily on John Locke's philosophy of natural and inalienable rights (liberty, property, security, and resistance to oppression) and Jean-Jacques Rousseau's concept of popular sovereignty and the 'general will,' asserting that sovereignty resides in the nation rather than a monarch."
    )

    # Q68
    entries[68] = build_entry(
        68, questions[67]['question'], "verified",
        source="DOC-HIS-2419.docx",
        chunk_indices=[81],
        keywords=["European monarchies", "radicalization", "Jacobins", "French Revolution", "war"],
        answer="The outbreak of war in 1792 against European monarchies determined to crush the revolution heightened national paranoia, internal subversion fears, and hyperinflation. This emergency atmosphere discredited moderate constitutional monarchists and fueled the political rise of radical Jacobins and sans-culottes who demanded extreme revolutionary defense."
    )

    # Q69
    entries[69] = build_entry(
        69, questions[68]['question'], "verified",
        source="DOC-HIS-2419.docx",
        chunk_indices=[81],
        keywords=["Robespierre", "Reign of Terror", "virtue", "Committee of Public Safety", "counter-revolution"],
        answer="Maximilien Robespierre and the Committee of Public Safety justified the Reign of Terror (1793-1794) as a necessary measure to defend the Republic against external invaders and internal counter-revolutionary treason, arguing that 'terror is nothing other than justice, prompt, severe, inflexible; it is therefore an emanation of virtue.'"
    )

    # Q70
    entries[70] = build_entry(
        70, questions[69]['question'], "verified",
        source="DOC-HIS-2419.docx",
        chunk_indices=[82],
        keywords=["Napoleon Bonaparte", "Napoleonic Code", "legal equality", "authoritarian", "empire"],
        answer="Napoleon consolidated key revolutionary gains by instituting the Napoleonic Civil Code, which guaranteed legal equality for men, standardized property rights, and abolished feudal privileges. Simultaneously, he suppressed free press, re-centralized executive power, crowned himself Emperor in 1804, and established a militarized authoritarian police state."
    )

    # Q71
    entries[71] = build_entry(
        71, questions[70]['question'], "verified",
        source="DOC-HIS-2520.docx",
        chunk_indices=[83],
        keywords=["Silk Road", "caravanserai", "relay trade", "oasis cities", "merchants"],
        answer="Trade along the terrestrial Silk Road operated as a relay trade system rather than individual merchants journeying end-to-end. Merchants moved goods between fortified oasis cities (such as Samarkand and Dunhuang) and roadside inns called caravanserai, where goods were exchanged, horses and camels rested, and local intermediaries took custody of shipments."
    )

    # Q72
    entries[72] = build_entry(
        72, questions[71]['question'], "verified",
        source="DOC-HIS-2520.docx",
        chunk_indices=[83],
        keywords=["Silk Road", "Buddhism", "India", "China", "monasteries"],
        answer="The Silk Road served as the primary conduit for the spread of Buddhism from northern India into China, Korea, and Japan. Monks and pilgrims traveled alongside merchant caravans; Buddhist monasteries established along trading oasis hubs provided lodging and banking services, fostering monastic patronage and translation of Sanskrit scriptures into Chinese."
    )

    # Q73
    entries[73] = build_entry(
        73, questions[72]['question'], "verified",
        source="DOC-HIS-2520.docx",
        chunk_indices=[84],
        keywords=["papermaking", "Islamic Golden Age", "transmission", "administration", "Silk Road"],
        answer="Following the Battle of Talas (751 CE), Chinese papermaking techniques were transmitted across Central Asia to Baghdad. Inexpensive, durable paper replaced scarce parchment and fragile papyrus, fueling administrative record-keeping, widespread book production, and scholarly translation movements across the Islamic Golden Age and later into Europe."
    )

    # Q74
    entries[74] = build_entry(
        74, questions[73]['question'], "verified",
        source="DOC-HIS-2520.docx",
        chunk_indices=[84],
        keywords=["pandemics", "Silk Road", "Plague of Justinian", "Black Death", "caravans"],
        answer="Merchant caravans and trade networks along the Silk Road served as epidemiological vectors for lethal pandemics. Rodents and fleas harboring Yersinia pestis traveled along caravan routes, facilitating catastrophic outbreaks such as the 6th-century Plague of Justinian and the 14th-century Black Death across Eurasia."
    )

    # Q75
    entries[75] = build_entry(
        75, questions[74]['question'], "verified",
        source="DOC-HIS-2520.docx",
        chunk_indices=[84],
        keywords=["Pax Mongolica", "decline", "terrestrial Silk Road", "maritime trade", "15th century"],
        answer="The disintegration of the Mongol Empire fragmented the Pax Mongolica, introducing regional conflicts, banditry, and exorbitant tolls along overland corridors. Concurrently, the rise of European and Asian maritime trade routes offered cheaper, safer, and higher-capacity ocean transit, leading to the gradual abandonment of terrestrial Silk Road routes in the 15th century."
    )

    # Questions 76 - 100: History & Specialized text files
    # Q76
    entries[76] = build_entry(
        76, questions[75]['question'], "verified",
        source="byzantine_empire_and_justinian_code.txt",
        chunk_indices=[56],
        keywords=["Corpus Juris Civilis", "Justinian Code", "Tribonian", "Roman law", "civil law"],
        answer="Emperor Justinian commissioned a legal committee headed by the jurist Tribonian to compile, edit, and systematize centuries of conflicting Roman imperial decrees, edicts, and juristic writings into the Corpus Juris Civilis (comprising the Code, Digest, and Institutes), which eliminated contradictions and formed the foundational basis of modern European civil law."
    )

    # Q77
    entries[77] = build_entry(
        77, questions[76]['question'], "verified",
        source="byzantine_empire_and_justinian_code.txt",
        chunk_indices=[56],
        keywords=["Constantinople", "Bosporus Strait", "Theodosian Walls", "bureaucracy", "Byzantine"],
        answer="Constantinople possessed formidable strategic advantages: located on the Bosporus Strait controlling lucrative trade between Europe and Asia, protected by the massive Theodosian Walls, and supported by a centralized bureaucracy and professional standing military capable of resisting Persian, Arab, and Slavic invasions for a thousand years."
    )

    # Q78: partially supported
    entries[78] = build_entry(
        78, questions[77]['question'], "partially_supported",
        source="byzantine_empire_and_justinian_code.txt",
        chunk_indices=[57],
        keywords=["theological disputes", "Great Schism 1054", "Eastern Orthodox", "Roman Catholic"],
        answer="The corpus documents long-standing theological, linguistic, and political disputes between the Greek Patriarchate in Constantinople and the Latin Papacy in Rome that culminated in the Great Schism of 1054, permanently dividing Christianity into Eastern Orthodox and Roman Catholic churches. However, the specific history and societal impact of the Iconoclastic Controversy are not covered in the text."
    )

    # Q79: partially supported
    entries[79] = build_entry(
        79, questions[78]['question'], "partially_supported",
        source="byzantine_empire_and_justinian_code.txt",
        chunk_indices=[56],
        keywords=["Byzantine military", "Arab invasions", "Slavic invasions", "defense"],
        answer="The corpus records that the Byzantine Empire maintained a complex, centralized administration and a sophisticated professional military that successfully repelled persistent incursions by Persians, Arabs, and Slavs. However, the specific military 'theme system' of administrative provinces and Byzantine diplomatic apparatus are not detailed in the text."
    )

    # Q80: not found
    entries[80] = build_entry(80, questions[79]['question'], "not_found")

    # Q81
    entries[81] = build_entry(
        81, questions[80]['question'], "verified",
        source="thirty_years_war_and_westphalia.txt",
        chunk_indices=[166],
        keywords=["Defenestration of Prague", "Thirty Years War", "Bohemian Protestants", "Habsburg", "1618"],
        answer="In May 1618, Protestant Bohemian nobles threw two Catholic imperial Habsburg regents out of a high window in Prague Castle (the Defenestration of Prague) in defiance of King Ferdinand II's revocation of religious freedoms, igniting the Bohemian Revolt that erupted into the pan-European Thirty Years' War."
    )

    # Q82
    entries[82] = build_entry(
        82, questions[81]['question'], "verified",
        source="thirty_years_war_and_westphalia.txt",
        chunk_indices=[166],
        keywords=["Cardinal Richelieu", "France", "Habsburg encirclement", "Protestant", "Raison d'etat"],
        answer="Although Catholic, France under First Minister Cardinal Richelieu pursued a foreign policy governed by 'raison d'état' (national interest) rather than religious solidarity. Richelieu intervened on the Protestant side to break the encirclement of France by Habsburg territories in Spain, the Low Countries, and the Holy Roman Empire."
    )

    # Q83
    entries[83] = build_entry(
        83, questions[82]['question'], "verified",
        source="thirty_years_war_and_westphalia.txt",
        chunk_indices=[166, 167],
        keywords=["mercenary armies", "Wallenstein", "Thirty Years War", "demographic devastation", "plundering"],
        answer="The Thirty Years' War relied heavily on undisciplined mercenary armies (such as those led by Albrecht von Wallenstein) that operated on the doctrine of 'war feeds war.' These armies sustained themselves through pillaging, requisitioning, and terrorizing civilian populations, causing massive famine, disease epidemics, and the loss of up to a third of the German population."
    )

    # Q84
    entries[84] = build_entry(
        84, questions[83]['question'], "verified",
        source="thirty_years_war_and_westphalia.txt",
        chunk_indices=[167],
        keywords=["Peace of Westphalia", "1648", "state sovereignty", "cuius regio eius religio", "Westphalian"],
        answer="The 1648 Peace of Westphalia recognized the territorial sovereignty of independent states, affirmed the principle of cuius regio, eius religio (allowing rulers to determine their territory's religion, extending to Calvinism), and established the foundational framework of modern international law based on non-interference in sovereign domestic affairs."
    )

    # Q85
    entries[85] = build_entry(
        85, questions[84]['question'], "verified",
        source="thirty_years_war_and_westphalia.txt",
        chunk_indices=[167],
        keywords=["Holy Roman Emperor", "German principalities", "sovereignty", "Peace of Westphalia", "decentralized"],
        answer="The Peace of Westphalia granted over 300 German princes the sovereign right to enact their own foreign treaties, alliances, and domestic policies without imperial veto, permanently shattering the centralized political and religious authority of the Holy Roman Emperor and rendering the Empire a fragmented confederation."
    )

    # Q86
    entries[86] = build_entry(
        86, questions[85]['question'], "verified",
        source="aztec_empire_spanish_conquest.txt",
        chunk_indices=[52],
        keywords=["chinampa", "Tenochtitlan", "agriculture", "Lake Texcoco", "Aztec"],
        answer="The Aztecs sustained Tenochtitlan's massive population of over 200,000 using 'chinampas'—highly productive artificial agricultural islands constructed on the shallow bed of Lake Texcoco by layering mud, vegetation, and reeds, enabling multi-crop harvesting throughout the year."
    )

    # Q87
    entries[87] = build_entry(
        87, questions[86]['question'], "verified",
        source="aztec_empire_spanish_conquest.txt",
        chunk_indices=[52],
        keywords=["tribute", "Flower Wars", "human sacrifice", "Tlaxcalans", "indigenous allies"],
        answer="The Aztec Triple Alliance exacted heavy tributary burdens in goods and seized human captives for ritual sacrifice through state-sponsored 'Flower Wars.' This fueled deep resentment among subjugated peoples, enabling Hernan Cortes to recruit tens of thousands of indigenous allies, notably the fiercely anti-Aztec Tlaxcalans."
    )

    # Q88
    entries[88] = build_entry(
        88, questions[87]['question'], "verified",
        source="aztec_empire_spanish_conquest.txt",
        chunk_indices=[53],
        keywords=["smallpox", "pathogens", "Tenochtitlan", "Cuitlahuac", "epidemiology"],
        answer="The smallpox virus, accidentally introduced by European forces, swept through Tenochtitlan during the Spanish retreat. Having no acquired immunological defense, an estimated 40% of the population perished, including Moctezuma's successor Emperor Cuitlahuac, shattering the Aztec military command and defense before Cortes laid final siege."
    )

    # Q89
    entries[89] = build_entry(
        89, questions[88]['question'], "verified",
        source="aztec_empire_spanish_conquest.txt",
        chunk_indices=[52],
        keywords=["Spanish metallurgy", "steel swords", "firearms", "horses", "cavalry"],
        answer="Spanish conquistadors possessed devastating technological advantages: forged steel swords and armor impervious to obsidian macuahuitl blades, harquebuses, cannons, and warhorses that provided psychological shock value and rapid tactical battlefield maneuverability."
    )

    # Q90: partially supported
    entries[90] = build_entry(
        90, questions[89]['question'], "partially_supported",
        source="aztec_empire_spanish_conquest.txt",
        chunk_indices=[52, 53],
        keywords=["Tenochtitlan", "Mexico City", "Spanish colonial rule", "fall of Aztec Empire"],
        answer="The corpus records the 93-day siege of Tenochtitlan, the surrender of Cuauhtemoc in 1521, the end of the Aztec Empire, and the inception of 300 years of Spanish colonial rule with Tenochtitlan becoming modern Mexico City. However, the specific architectural demolition of the Templo Mayor and immediate rebuilding details are not documented in the text."
    )

    # Q91
    entries[91] = build_entry(
        91, questions[90]['question'], "verified",
        source="the_haitian_revolution.txt",
        chunk_indices=[162],
        keywords=["Saint-Domingue", "sugar colony", "plantation economy", "enslaved Africans", "mortality"],
        answer="Saint-Domingue was France's most lucrative Caribbean colony, producing 40% of Europe's sugar and 60% of its coffee through a brutal plantation regime where 500,000 enslaved Africans endured extreme labor exploitation, horrific abuse, and high mortality rates requiring relentless importation of slave ships."
    )

    # Q92
    entries[92] = build_entry(
        92, questions[91]['question'], "verified",
        source="the_haitian_revolution.txt",
        chunk_indices=[162, 163],
        keywords=["Toussaint Louverture", "military tactics", "guerrilla", "yellow fever", "Spanish alliance"],
        answer="Toussaint Louverture organized rebel slaves into a disciplined army, skillfully shifting diplomatic alliances between Spanish, French, and British factions, utilizing guerrilla tactics, fortifying interior mountain positions, and letting yellow fever decimate invading European armies before mounting counter-offensives."
    )

    # Q93
    entries[93] = build_entry(
        93, questions[92]['question'], "verified",
        source="the_haitian_revolution.txt",
        chunk_indices=[162],
        keywords=["Declaration of Rights of Man", "French Revolution", "rebellion", "liberty", "equality"],
        answer="The French revolutionary rhetoric and the 'Declaration of the Rights of Man and of the Citizen' destabilized colonial society by promoting ideals of universal liberty and equality. Enslaved Africans seized upon this ideological opening and the division among white planters and free people of color to launch a massive coordinated revolt in August 1791."
    )

    # Q94
    entries[94] = build_entry(
        94, questions[93]['question'], "verified",
        source="the_haitian_revolution.txt",
        chunk_indices=[163],
        keywords=["Napoleon", "Leclerc expedition", "reinstate slavery", "yellow fever", "Vertieres"],
        answer="In 1802, Napoleon sent 30,000 veteran troops under General Charles Leclerc to crush Toussaint's government and secretly reinstate slavery. The expedition collapsed due to fierce resistance led by Jean-Jacques Dessalines and a catastrophic outbreak of yellow fever that decimated the French military, culminating in their defeat at the Battle of Vertieres."
    )

    # Q95
    entries[95] = build_entry(
        95, questions[94]['question'], "verified",
        source="the_haitian_revolution.txt",
        chunk_indices=[163],
        keywords=["diplomatic isolation", "French indemnity", "1825", "economic burden", "Haiti"],
        answer="Following independence in 1804, European powers and the United States enforced diplomatic and commercial isolation on Haiti for decades fearing domestic slave uprisings. In 1825, France forced Haiti to pay a crushing 150-million-franc indemnity backed by naval warships to secure recognition, burdening Haiti with ruinous foreign debt for over a century."
    )

    # Q96
    entries[96] = build_entry(
        96, questions[95]['question'], "verified",
        source="soviet_command_economy_and_five_year_plans.txt",
        chunk_indices=[133],
        keywords=["Stalin", "First Five-Year Plan", "heavy industry", "Gosplan", "Magnitogorsk"],
        answer="Stalin's First Five-Year Plan (1928-1932) terminated the New Economic Policy and mobilized the USSR into a centralized command economy directed by Gosplan. It prioritized heavy industry—coal, iron, steel, electricity, and machinery—building massive industrial complexes like Magnitogorsk using state propaganda, target quotas, and forced labor."
    )

    # Q97
    entries[97] = build_entry(
        97, questions[96]['question'], "verified",
        source="soviet_command_economy_and_five_year_plans.txt",
        chunk_indices=[133, 134],
        keywords=["collectivization", "dekulakization", "Holodomor", "famine", "Kolkhozes"],
        answer="Forced agricultural collectivization abolished private landholdings, forcing peasants onto state-run collective farms (Kolkhozes). Resisting peasants and branded 'kulaks' were executed or deported to Siberian Gulags; punitive grain requisitioning quotas led to the 1932-1933 Holodomor famine, killing an estimated 3.5 to 5 million people in Ukraine and major grain regions."
    )

    # Q98: partially supported
    entries[98] = build_entry(
        98, questions[97]['question'], "partially_supported",
        source="soviet_command_economy_and_five_year_plans.txt",
        chunk_indices=[133, 134],
        keywords=["Gosplan", "command economy", "production targets", "resource allocation"],
        answer="The corpus details that Gosplan (the State Planning Committee) directed the rigid Soviet command economy by establishing centralized production quotas and dictating resource allocation across industries and agriculture without market pricing. However, the specific mathematical method of calculating 'material balances' is not documented in the text."
    )

    # Q99
    entries[99] = build_entry(
        99, questions[98]['question'], "verified",
        source="soviet_command_economy_and_five_year_plans.txt",
        chunk_indices=[134],
        keywords=["command economy", "innovation", "consumer goods", "coercion", "stagnation"],
        answer="The Soviet command economy struggled with non-military innovation and quality control due to the absence of market price signals, perverse incentives focusing strictly on gross output volume rather than usability, chronic neglect of the consumer sector, and reliance on bureaucratic coercion rather than technological improvement."
    )

    # Q100
    entries[100] = build_entry(
        100, questions[99]['question'], "verified",
        source="soviet_command_economy_and_five_year_plans.txt",
        chunk_indices=[134],
        keywords=["heavy industry", "military production", "economic stagnation", "consumer goods", "collapse"],
        answer="By funneling national wealth, raw materials, and top engineering talent into heavy industry and the military apparatus, the Soviet system created chronic shortages of basic consumer necessities, agricultural deficits, and misallocation of resources, embedding structural stagnation that contributed directly to the USSR's economic collapse in the late 20th century."
    )

    # Questions 101 - 150: Automotive engineering documents
    # Q101 to Q105: CAN Bus
    entries[101] = build_entry(
        101, questions[100]['question'], "verified",
        source="Automotive_CAN_Bus_Architecture_and_Diagnostics.docx",
        chunk_indices=[20],
        keywords=["CAN bus", "arbitration", "dominant bit", "recessive bit", "CSMA/CD + AMP", "identifier"],
        answer="The CAN bus employs non-destructive bitwise arbitration using Carrier Sense Multiple Access with Collision Detection and Arbitration on Message Priority (CSMA/CD + AMP). During the arbitration field, transmitting nodes monitor the bus. A dominant bit (logic 0) overwrites a recessive bit (logic 1). A node transmitting a recessive bit that senses a dominant bit detects arbitration loss and immediately halts transmission without corrupting the higher-priority message."
    )

    entries[102] = build_entry(
        102, questions[101]['question'], "verified",
        source="Automotive_CAN_Bus_Architecture_and_Diagnostics.docx",
        chunk_indices=[21],
        keywords=["CAN FD", "Flexible Data-Rate", "payload", "64 bytes", "baud rate", "bit rate switching"],
        answer="CAN FD (Flexible Data-Rate) was introduced to overcome Classical CAN's 8-byte payload and 1 Mbps bandwidth limits. CAN FD expands data payloads up to 64 bytes and uses dual bit-rate switching—maintaining standard arbitration speed (500 kbps) while accelerating the data phase up to 5 Mbps, reducing bus load and frame overhead for ADAS and powertrain data."
    )

    entries[103] = build_entry(
        103, questions[102]['question'], "verified",
        source="Automotive_CAN_Bus_Architecture_and_Diagnostics.docx",
        chunk_indices=[21],
        keywords=["Zonal Architectures", "wiring harness", "Zone Controllers", "High-Speed Ethernet", "ECUs"],
        answer="Zonal Architectures simplify physical wiring harnesses by replacing vehicle-wide point-to-point connections with localized Zone Controllers (Front, Rear, Left, Right). Sensors and actuators connect locally to their nearest Zone Controller over short LIN or CAN links, and the Zone Controllers aggregate data over a high-bandwidth Ethernet backbone, reducing harness weight by up to 30%."
    )

    entries[104] = build_entry(
        104, questions[103]['question'], "verified",
        source="Automotive_CAN_Bus_Architecture_and_Diagnostics.docx",
        chunk_indices=[22],
        keywords=["SecOC", "Secure Onboard Communication", "MAC", "Freshness Value", "AUTOSAR", "CAN"],
        answer="Secure Onboard Communication (SecOC) authenticates CAN messages by generating a cryptographic Message Authentication Code (MAC, via CMAC or HMAC) appended to the payload. It incorporates a monotonically increasing Freshness Value (counter or timestamp) into the MAC computation, preventing replay attacks and unauthorized message injection."
    )

    entries[105] = build_entry(
        105, questions[104]['question'], "verified",
        source="Automotive_CAN_Bus_Architecture_and_Diagnostics.docx",
        chunk_indices=[21],
        keywords=["LIN", "Local Interconnect Network", "master-slave", "low cost", "single-wire", "sensors"],
        answer="The Local Interconnect Network (LIN) operates on a strict single-master multiple-slave polling architecture using an inexpensive single-wire transceiver and clock synchronization, eliminating message collisions, arbitration logic, and expensive quartz crystals in slave nodes, making it ideal for low-speed body electronics (mirrors, seats, window controls)."
    )

    # Q106 to Q110: EV BMS
    entries[106] = build_entry(
        106, questions[105]['question'], "verified",
        source="Electric_Vehicle_Battery_Management_Systems_BMS.docx",
        chunk_indices=[96],
        keywords=["BMS", "Extended Kalman Filter", "EKF", "SoC", "State of Charge", "Coulomb counting"],
        answer="Battery Management Systems (BMS) combine Coulomb Counting (current integration) with Extended Kalman Filtering (EKF) to estimate State of Charge (SoC). The EKF uses an equivalent circuit model of the battery and continuously updates its internal state estimates by comparing model predictions with real-time terminal voltage and current measurements, correcting for sensor drift and flat OCV plateaus."
    )

    entries[107] = build_entry(
        107, questions[106]['question'], "verified",
        source="Electric_Vehicle_Battery_Management_Systems_BMS.docx",
        chunk_indices=[96],
        keywords=["cell balancing", "passive balancing", "active balancing", "resistors", "capacitors", "inductors"],
        answer="Passive cell balancing bleeds off excess energy from the highest-voltage cells as heat through switched bypass resistors until all cells match. Active balancing uses inductive or capacitive switched-mode DC-DC converter circuits to transfer electrical charge dynamically from higher-capacity cells to lower-capacity cells with minimal energy loss."
    )

    entries[108] = build_entry(
        108, questions[107]['question'], "verified",
        source="Electric_Vehicle_Battery_Management_Systems_BMS.docx",
        chunk_indices=[97],
        keywords=["thermal runaway", "BMS", "contactor opening", "pyro-fuse", "cooling", "degassing"],
        answer="The BMS detects thermal runaway onset by monitoring rapid temperature spikes, voltage collapses, and off-gassing pressure changes. When detected, it activates max coolant flow, alerts occupants, opens high-voltage contactors, and triggers pyrotechnic disconnect fuses (pyro-fuses) to isolate the high-voltage pack within milliseconds."
    )

    entries[109] = build_entry(
        109, questions[108]['question'], "verified",
        source="Electric_Vehicle_Battery_Management_Systems_BMS.docx",
        chunk_indices=[97],
        keywords=["pre-charge circuits", "inrush current", "DC-link capacitors", "contactors", "welding"],
        answer="Before closing main high-voltage contactors, the BMS energizes a pre-charge relay with a series resistor to slowly charge downstream DC-link inverter capacitors up to ~95% of battery voltage. This limits catastrophic inrush currents that would otherwise weld the main contactor contacts together."
    )

    entries[110] = build_entry(
        110, questions[109]['question'], "verified",
        source="Electric_Vehicle_Battery_Management_Systems_BMS.docx",
        chunk_indices=[97],
        keywords=["isolation resistance", "high voltage", "chassis ground", "insulation fault", "electric shock"],
        answer="High-voltage isolation monitoring continuously measures the electrical resistance between positive/negative high-voltage buses and the low-voltage vehicle chassis ground. If moisture or insulation breakdown drops resistance below standard thresholds (e.g. 100-500 ohms/volt), the BMS warns the driver or isolates the contactors to prevent electric shock."
    )

    # Q111 to Q115: Autonomous Driving Sensor Fusion
    entries[111] = build_entry(
        111, questions[110]['question'], "verified",
        source="Autonomous_Driving_Sensor_Fusion_and_Perception.docx",
        chunk_indices=[48],
        keywords=["cameras", "radar", "lidar", "environmental limitations", "weather", "point cloud"],
        answer="Cameras provide rich color and texture but degrade severely in fog, glare, darkness, and rain; Radar operates reliably across all weather and measures Doppler velocity but provides low spatial resolution; LiDAR generates precise 3D geometric point clouds but is expensive and suffers backscatter attenuation in dense fog, snow, and rain."
    )

    entries[112] = build_entry(
        112, questions[111]['question'], "verified",
        source="Autonomous_Driving_Sensor_Fusion_and_Perception.docx",
        chunk_indices=[49],
        keywords=["sensor fusion", "early fusion", "late fusion", "deep neural networks", "raw data"],
        answer="Early sensor fusion feeds synchronized raw or low-level feature representations (camera pixels and LiDAR 3D point clouds) directly into a unified multimodal deep neural network backbone. This preserves low-level cross-sensor spatial correlations that late fusion (merging bounding boxes after independent detection) discards."
    )

    entries[113] = build_entry(
        113, questions[112]['question'], "verified",
        source="Autonomous_Driving_Sensor_Fusion_and_Perception.docx",
        chunk_indices=[49],
        keywords=["GPS", "RTK-GPS", "IMU", "HD Maps", "SLAM", "centimeter accuracy"],
        answer="Standard consumer GPS has multi-meter errors and signal dropouts under bridges or urban canyons. Autonomous lane centering requires Real-Time Kinematic (RTK) GPS, high-grade Inertial Measurement Units (IMU), wheel odometry, and LiDAR-based SLAM matching against high-definition (HD) maps to maintain sub-10 cm positioning accuracy."
    )

    entries[114] = build_entry(
        114, questions[113]['question'], "verified",
        source="Autonomous_Driving_Sensor_Fusion_and_Perception.docx",
        chunk_indices=[49],
        keywords=["motion planning", "trajectory generation", "jerk", "collision avoidance", "cost function"],
        answer="Autonomous motion planning formulates trajectory generation as an optimization problem minimizing a composite cost function that penalizes collision risk, road boundary violations, path curvature, acceleration, and jerk (rate of change of acceleration), thereby ensuring obstacle avoidance while maintaining passenger comfort."
    )

    entries[115] = build_entry(
        115, questions[114]['question'], "verified",
        source="Autonomous_Driving_Sensor_Fusion_and_Perception.docx",
        chunk_indices=[49],
        keywords=["MPC", "Model Predictive Control", "prediction horizon", "actuator constraints", "steering"],
        answer="Model Predictive Control (MPC) continuously solves a constrained optimization problem over a moving future time horizon using a dynamic vehicle model. It predicts vehicle lateral/longitudinal behavior, calculates optimal steering and throttle commands that respect physical tire and actuator limits, executes the first time step, and repeats."
    )

    # Q116 to Q120: ICE Thermodynamics
    entries[116] = build_entry(
        116, questions[115]['question'], "verified",
        source="Internal_Combustion_Engine_Thermodynamics.docx",
        chunk_indices=[114],
        keywords=["Miller cycle", "Atkinson cycle", "expansion ratio", "compression ratio", "valve timing"],
        answer="The Miller and Atkinson cycles use late or early Intake Valve Closing (IVC) to make the effective expansion ratio greater than the compression ratio. This over-expansion extracts more thermodynamic work from expanding combustion gases before exhaust valve opening, increasing thermal efficiency and reducing pumping losses."
    )

    entries[117] = build_entry(
        117, questions[116]['question'], "verified",
        source="Internal_Combustion_Engine_Thermodynamics.docx",
        chunk_indices=[115],
        keywords=["GDI", "Gasoline Direct Injection", "charge cooling", "vaporization", "knock"],
        answer="Gasoline Direct Injection (GDI) injects fuel directly into the combustion chamber during intake or compression. The latent heat of vaporization of the evaporating fuel cools the in-cylinder air-fuel charge, increasing charge density, reducing engine knock tendency, and allowing higher compression ratios."
    )

    entries[118] = build_entry(
        118, questions[117]['question'], "verified",
        source="Internal_Combustion_Engine_Thermodynamics.docx",
        chunk_indices=[114],
        keywords=["VGT", "Variable Geometry Turbine", "turbo lag", "vanes", "exhaust gas"],
        answer="Variable Geometry Turbines (VGT) use adjustable internal guide vanes around the turbine wheel. At low engine speeds, vanes narrow the exhaust passages to accelerate gas velocity and spool the turbocharger rapidly to eliminate turbo lag; at high speeds, vanes open to maximize exhaust flow without excessive backpressure."
    )

    entries[119] = build_entry(
        119, questions[118]['question'], "verified",
        source="Internal_Combustion_Engine_Thermodynamics.docx",
        chunk_indices=[115],
        keywords=["Three-Way Catalytic Converter", "TWC", "stoichiometric", "lambda 1.0", "emissions"],
        answer="A Three-Way Catalytic Converter (TWC) simultaneously oxidizes hydrocarbons (HC) and carbon monoxide (CO) into CO2 and H2O while reducing nitrogen oxides (NOx) into N2. This chemical redox equilibrium requires the air-fuel ratio to remain within a razor-thin window around stoichiometry (lambda = 1.0)."
    )

    entries[120] = build_entry(
        120, questions[119]['question'], "verified",
        source="Internal_Combustion_Engine_Thermodynamics.docx",
        chunk_indices=[116],
        keywords=["SCR", "Selective Catalytic Reduction", "DEF", "urea", "ammonia", "NOx"],
        answer="Selective Catalytic Reduction (SCR) injects Diesel Exhaust Fluid (DEF / 32.5% aqueous urea) into the hot exhaust stream. The urea decomposes into ammonia (NH3), which reacts with NOx across a catalytic washcoat to reduce toxic nitrogen oxides into harmless diatomic nitrogen (N2) and water vapor (H2O)."
    )

    # Q121 to Q125: Vehicle Aerodynamics & CFD
    entries[121] = build_entry(
        121, questions[120]['question'], "verified",
        source="Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx",
        chunk_indices=[172],
        keywords=["flow separation", "wake", "form drag", "pressure differential", "rear"],
        answer="Flow separation at the sharp edges or rear of a vehicle causes turbulent shear layers and a chaotic, unsteady low-pressure recirculating wake. The net pressure difference between high stagnation pressure at the vehicle front and low suction pressure behind creates substantial pressure drag (form drag)."
    )

    entries[122] = build_entry(
        122, questions[121]['question'], "verified",
        source="Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx",
        chunk_indices=[172],
        keywords=["active aerodynamics", "active grille shutters", "spoilers", "drag", "downforce"],
        answer="Active aerodynamic systems modulate airflow dynamically: active grille shutters close at high cruising speeds to prevent drag from underhood cooling airflow, and deployable rear spoilers adjust tilt angle to trade off between reduced drag during high-speed highway cruising and increased downforce during hard braking or high-speed cornering."
    )

    entries[123] = build_entry(
        123, questions[122]['question'], "verified",
        source="Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx",
        chunk_indices=[172],
        keywords=["underbody airflow", "diffuser", "downforce", "Bernoulli principle", "ground effect"],
        answer="Smooth underbody panels accelerate airflow underneath the vehicle relative to ambient air. In accordance with Bernoulli's principle, increased fluid velocity creates a low static pressure area under the floorpan. A rear diffuser gradually expands the exit area to slow down airflow, sucking the chassis to the tarmac (ground effect downforce) with minimal drag penalty."
    )

    entries[124] = build_entry(
        124, questions[123]['question'], "verified",
        source="Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx",
        chunk_indices=[173],
        keywords=["DES", "Detached Eddy Simulation", "CFD", "RANS", "LES", "turbulent wake"],
        answer="Detached Eddy Simulation (DES) is a hybrid computational fluid dynamics model that treats attached boundary layers near vehicle walls using computationally efficient Reynolds-Averaged Navier-Stokes (RANS) equations, while switching to Large Eddy Simulation (LES) in detached flow regions to resolve large-scale unsteady turbulent vortices in the wake."
    )

    entries[125] = build_entry(
        125, questions[124]['question'], "verified",
        source="Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx",
        chunk_indices=[172],
        keywords=["Kammback", "vortex generators", "boundary layer", "clean separation", "drag"],
        answer="Kammback designs truncate the rear bodywork at a point where the airflow separates cleanly, mimicking the aerodynamic characteristics of a long tear-drop shape while maintaining practical vehicle dimensions. Vortex generators introduce small high-energy vortices that re-energize the boundary layer, delaying separation and narrowing the wake."
    )

    # Q126 to Q130: FCEV Hydrogen Storage & Stack
    entries[126] = build_entry(
        126, questions[125]['question'], "verified",
        source="FCEV_Hydrogen_Storage_and_Stack_Architecture.docx",
        chunk_indices=[106],
        keywords=["PEM fuel cell", "platinum catalyst", "anode", "protons", "electrons", "current"],
        answer="At the anode of a Polymer Electrolyte Membrane (PEM) fuel cell, a platinum catalyst adsorbs H2 molecules and dissociates them into hydrogen protons (H+) and electrons (e-). The polymer membrane conducts protons across to the cathode while blocking electrons, forcing electrons through an external circuit to generate electric current."
    )

    entries[127] = build_entry(
        127, questions[126]['question'], "verified",
        source="FCEV_Hydrogen_Storage_and_Stack_Architecture.docx",
        chunk_indices=[107],
        keywords=["Air Management System", "chemical filters", "humidifier", "PEM membrane", "conductivity"],
        answer="The Air Management System supplies filtered and humidified oxygen to the cathode. Chemical filters remove airborne pollutants (sulfur, ammonia) that would permanently poison the platinum catalyst, while humidifiers maintain membrane moisture, ensuring optimal proton conductivity and preventing membrane dehydration or dry-out cracking."
    )

    entries[128] = build_entry(
        128, questions[127]['question'], "verified",
        source="FCEV_Hydrogen_Storage_and_Stack_Architecture.docx",
        chunk_indices=[107],
        keywords=["Type IV cylinder", "hydrogen storage", "polymer liner", "carbon fiber", "700 bar"],
        answer="Type IV hydrogen cylinders feature a seamless, blow-molded High-Density Polyethylene (HDPE) or polyamide polymer liner to prevent hydrogen gas permeation, overwrapped with continuous strands of high-tensile carbon fiber embedded in an epoxy resin matrix to withstand internal pressures of 700 bar (10,000 psi) while minimizing weight."
    )

    entries[129] = build_entry(
        129, questions[128]['question'], "verified",
        source="FCEV_Hydrogen_Storage_and_Stack_Architecture.docx",
        chunk_indices=[107],
        keywords=["TPRD", "Thermal Pressure Relief Device", "hydrogen tank", "fire", "venting"],
        answer="A Thermal Pressure Relief Device (TPRD) is a thermally triggered safety release valve incorporating a eutectic alloy or glass bulb that melts when exposed to fire (typically above 110 degrees C), venting the entire compressed hydrogen supply in a controlled vertical plume to prevent catastrophic cylinder explosion."
    )

    entries[130] = build_entry(
        130, questions[129]['question'], "verified",
        source="FCEV_Hydrogen_Storage_and_Stack_Architecture.docx",
        chunk_indices=[108],
        keywords=["FCEV", "hybrid architecture", "lithium-ion buffer battery", "regenerative braking", "transient"],
        answer="FCEVs are hybrid architectures combining a fuel cell stack with a lithium-ion buffer battery because fuel cells cannot accept reverse electrical power from regenerative braking, and experience electro-mechanical transient response lag during sudden acceleration. The battery captures braking energy and supplies instant power bursts."
    )

    # Q131 to Q135: Automotive ECU Architecture
    entries[131] = build_entry(
        131, questions[130]['question'], "verified",
        source="Automotive_ECU_Architecture_and_Microcontrollers.docx",
        chunk_indices=[28],
        keywords=["DCLS", "Dual-Core Lockstep", "ASIL D", "transient faults", "comparator", "microcontrollers"],
        answer="Dual-Core Lockstep (DCLS) executes the identical instruction stream redundantly across two processor cores staggered by a few clock cycles. Hardware comparators continuously check internal register states; if an upset such as a cosmic-ray single-event upset causes a bit flip, a mismatch is detected within cycles, triggering safe states required for ASIL D compliance."
    )

    entries[132] = build_entry(
        132, questions[131]['question'], "verified",
        source="Automotive_ECU_Architecture_and_Microcontrollers.docx",
        chunk_indices=[29],
        keywords=["ECC", "Error-Correcting Code", "SRAM", "single-bit error", "multi-bit error", "safety"],
        answer="Error-Correcting Code (ECC) logic protects critical on-chip SRAM and flash memory sectors by calculating syndrome bits for stored data. ECC automatically detects and transparently corrects single-bit memory corruptions on read access, and detects multi-bit errors to raise a non-maskable safety interrupt, preventing execution of corrupted code."
    )

    entries[133] = build_entry(
        133, questions[132]['question'], "verified",
        source="Automotive_ECU_Architecture_and_Microcontrollers.docx",
        chunk_indices=[29],
        keywords=["AUTOSAR", "RTE", "BSW", "Runtime Environment", "Basic Software", "hardware abstraction"],
        answer="AUTOSAR decouples application software from underlying microcontroller hardware via layered abstraction: Application Software Components (SWCs) interact exclusively with the Runtime Environment (RTE). The RTE routes calls down to Basic Software (BSW) modules (MCAL, ECAL, Services), allowing application code to be reused across different microcontrollers without hardware rewrites."
    )

    entries[134] = build_entry(
        134, questions[133]['question'], "verified",
        source="Automotive_ECU_Architecture_and_Microcontrollers.docx",
        chunk_indices=[30],
        keywords=["Zonal Architectures", "High-Performance Computers", "HPCs", "POSIX", "Ethernet"],
        answer="The automotive industry is shifting from flat distributed ECU networks to Zonal Architectures to eliminate overwhelming wiring harness complexity, weight, and fragmented firmware maintenance. A small number of server-grade High-Performance Computers (HPCs) running POSIX-compliant OSs centralize compute and virtualization, communicating with regional Zone Controllers over Gigabit Ethernet."
    )

    entries[135] = build_entry(
        135, questions[134]['question'], "verified",
        source="Automotive_ECU_Architecture_and_Microcontrollers.docx",
        chunk_indices=[29],
        keywords=["PWM", "Pulse Width Modulation", "duty cycle", "actuators", "fuel injectors"],
        answer="Pulse Width Modulation (PWM) modules allow an engine or motor control unit to precisely manipulate analog actuators by rapidly pulsing digital on/off signals at high frequency while varying the duty cycle (percentage of time high), controlling average voltage, motor speed, solenoid opening, and fuel injection duration."
    )

    # Q136 to Q140: Automotive Tribology
    entries[136] = build_entry(
        136, questions[135]['question'], "verified",
        source="Automotive_Tribology_Friction_and_Wear.docx",
        chunk_indices=[44],
        keywords=["Stribeck curve", "Hydrodynamic Lubrication", "Boundary Lubrication", "friction", "viscosity"],
        answer="The Stribeck curve illustrates the friction coefficient as a function of the Hersey number (viscosity * speed / load). Under normal cruising, full-fluid hydrodynamic lubrication completely separates surfaces; during cold starts, low speeds, or high loads, fluid film collapses into boundary lubrication where microscopic asperities make direct contact, drastically increasing friction and wear."
    )

    entries[137] = build_entry(
        137, questions[136]['question'], "verified",
        source="Automotive_Tribology_Friction_and_Wear.docx",
        chunk_indices=[44],
        keywords=["ZDDP", "Extreme Pressure", "tribofilm", "anti-wear", "valvetrain"],
        answer="Under localized frictional heating and high contact pressure in boundary contact, Zinc Dialkyldithiophosphate (ZDDP) additives decompose to form a resilient, glassy polyphosphate sacrificial tribofilm (iron/zinc phosphate) on metal asperities. This sacrificial layer shears easily, preventing metal-to-metal welding and catastrophic adhesive wear."
    )

    entries[138] = build_entry(
        138, questions[137]['question'], "verified",
        source="Automotive_Tribology_Friction_and_Wear.docx",
        chunk_indices=[45],
        keywords=["DLC", "Diamond-Like Carbon", "low viscosity", "0W-16", "friction reduction"],
        answer="Diamond-Like Carbon (DLC) coatings exhibit extremely high hardness, atomic smoothness, and chemically inert surface properties with low surface energy. Applied to valvetrain cam followers and piston rings, DLC provides boundary friction reduction and prevents micro-scuffing even when ultra-low viscosity engine oils (such as 0W-16 or 0W-8) are utilized."
    )

    entries[139] = build_entry(
        139, questions[138]['question'], "verified",
        source="Automotive_Tribology_Friction_and_Wear.docx",
        chunk_indices=[45],
        keywords=["EDM", "Electrical Discharge Machining", "bearing raceways", "ceramic balls", "pitting"],
        answer="High-speed EV traction motors driven by fast-switching inverter PWM inducers induce parasitic capacitive shaft voltages. When this voltage discharges across the bearing lubricant film, it causes Electrical Discharge Machining (EDM) arcing that pits and flutes bearing raceways. Non-conductive ceramic silicon nitride (Si3N4) balls electrically insulate the bearing to eliminate EDM currents."
    )

    entries[140] = build_entry(
        140, questions[139]['question'], "verified",
        source="Automotive_Tribology_Friction_and_Wear.docx",
        chunk_indices=[46],
        keywords=["hydroplaning", "Elastohydrodynamic Lubrication", "tire-road contact patch", "water film"],
        answer="During dynamic hydroplaning, water builds up in front of the rolling tire faster than tire tread sipes can evacuate it, generating high hydrodynamic fluid pressures. When hydrodynamic lift exceeds tire contact pressure, a full continuous water film separates the rubber from the asphalt (Elastohydrodynamic Lubrication), reducing friction to near zero and causing total loss of steering and braking."
    )

    # Q141 to Q145: Automotive Crashworthiness
    entries[141] = build_entry(
        141, questions[140]['question'], "verified",
        source="Automotive_Crashworthiness_and_Structural_Safety.docx",
        chunk_indices=[24],
        keywords=["crumple zones", "plastic deformation", "impact pulse", "peak deceleration", "kinetic energy"],
        answer="Engineered crumple zones utilize progressive, folding plastic deformation of longitudinal structural members to absorb and dissipate collision kinetic energy over distance. This extends the duration of the impact deceleration pulse (from milliseconds to tenths of a second), dramatically reducing peak G-forces transmitted to occupants."
    )

    entries[142] = build_entry(
        142, questions[141]['question'], "verified",
        source="Automotive_Crashworthiness_and_Structural_Safety.docx",
        chunk_indices=[25],
        keywords=["UHSS", "boron steel", "passenger safety cell", "cabin intrusion", "A-pillar", "B-pillar"],
        answer="Ultra-High-Strength Steels (UHSS), such as hot-stamped boron steels (tensile strength > 1500 MPa), form a rigid, non-deformable safety cage around the occupant cell (A-pillars, B-pillars, roof rails, sill reinforcements). Unlike crumple zones that fold, UHSS resists crushing forces to prevent cabin intrusion during side impacts and rollovers."
    )

    entries[143] = build_entry(
        143, questions[142]['question'], "verified",
        source="Automotive_Crashworthiness_and_Structural_Safety.docx",
        chunk_indices=[25],
        keywords=["pretensioners", "load limiters", "seatbelts", "thoracic injury", "pyrotechnic"],
        answer="Upon collision detection, pyrotechnic pretensioners detonate a micro gas charge that retracts the seatbelt spool by several inches within 10-15 milliseconds, coupling the occupant tightly to the seat. As deceleration peaks, mechanical torsion-bar load limiters deform progressively to yield seatbelt webbing, preventing excessive chest compression and thoracic skeletal fractures."
    )

    entries[144] = build_entry(
        144, questions[143]['question'], "verified",
        source="Automotive_Crashworthiness_and_Structural_Safety.docx",
        chunk_indices=[26],
        keywords=["MEMS accelerometers", "Restraint Control Module", "RCM", "airbag deployment", "20 milliseconds"],
        answer="Solid-state MEMS capacitive accelerometers measure multi-axis high-G deceleration forces and send raw data to the central Restraint Control Module (RCM). The RCM's crash algorithm integrates deceleration over time to determine crash severity within 15-20 milliseconds and commands squib firing pins to deploy airbags before the occupant contacts the steering wheel."
    )

    entries[145] = build_entry(
        145, questions[144]['question'], "verified",
        source="Automotive_Crashworthiness_and_Structural_Safety.docx",
        chunk_indices=[26],
        keywords=["FEA", "Finite Element Analysis", "crash simulation", "LS-DYNA", "nonlinear"],
        answer="Finite Element Analysis (FEA) solvers (such as LS-DYNA) discretize the vehicle CAD geometry into millions of finite elements to model nonlinear material plasticity, strain-rate sensitivity, structural buckling, and contact mechanics under dynamic crash physics, allowing engineers to optimize structural crashworthiness before physical tooling is cut."
    )

    # Q146 to Q150: Software-Defined Vehicle & OTA
    entries[146] = build_entry(
        146, questions[145]['question'], "verified",
        source="Software_Defined_Vehicle_SDV_and_OTA_Updates.docx",
        chunk_indices=[129],
        keywords=["SOA", "SOME/IP", "Service-Oriented Architecture", "SDV", "hardware-agnostic"],
        answer="A Service-Oriented Architecture (SOA) implemented via protocols like SOME/IP (Scalable service-Oriented MiddlewarE over IP) shifts automotive communication from fixed signal-based CAN broadcasting to client-server publish-subscribe services. This abstracts underlying ECU hardware and allows developers to write modular, hardware-agnostic applications that interact through standardized APIs."
    )

    entries[147] = build_entry(
        147, questions[146]['question'], "verified",
        source="Software_Defined_Vehicle_SDV_and_OTA_Updates.docx",
        chunk_indices=[130],
        keywords=["dual-bank", "A/B memory", "OTA rollback", "flash partition", "fail-safe"],
        answer="A dual-bank (A/B) memory architecture divides storage into two identical partitions: Bank A executes the current active software while Bank B flashes the incoming Over-The-Air (OTA) update in the background. After successful installation and cryptographic verification, the bootloader swaps the active bank. If the new software fails boot self-tests, the bootloader immediately rolls back to Bank A."
    )

    entries[148] = build_entry(
        148, questions[147]['question'], "verified",
        source="Software_Defined_Vehicle_SDV_and_OTA_Updates.docx",
        chunk_indices=[130],
        keywords=["CI/CD", "HIL", "Hardware-in-the-Loop", "OTA", "automated testing"],
        answer="CI/CD pipelines connect automated software build stages to Hardware-in-the-Loop (HIL) simulators that replicate vehicle electrical loads, sensor inputs, and CAN/Ethernet networks in real-time. Automated test suites validate software releases against physical target microcontrollers and simulated driving scenarios before approving firmware packages for OTA deployment."
    )

    entries[149] = build_entry(
        149, questions[148]['question'], "verified",
        source="Software_Defined_Vehicle_SDV_and_OTA_Updates.docx",
        chunk_indices=[130],
        keywords=["telemetry", "Digital Twin", "OEM cloud", "predictive maintenance", "machine learning"],
        answer="Connected vehicle telemetry (sensor streams, diagnostic codes, driving dynamics) is continuously uploaded to the OEM's cloud infrastructure to maintain a digital representation ('Digital Twin') of every vehicle. Machine learning models analyze this aggregate fleet data to identify early component degradation for predictive maintenance and to train perception models for ADAS."
    )

    entries[150] = build_entry(
        150, questions[149]['question'], "verified",
        source="Software_Defined_Vehicle_SDV_and_OTA_Updates.docx",
        chunk_indices=[131],
        keywords=["UN R156", "SUMS", "Software Update Management System", "OTA", "compliance"],
        answer="International regulation UN R156 mandates that automakers establish a certified Software Update Management System (SUMS). It requires OEMs to maintain end-to-end traceability of software versions across all vehicle configurations, assess safety impacts, protect updates from cyber tampering, and ensure safe rollback capabilities to maintain regulatory type approval."
    )

    # Compile all records in order 1 to 150
    for qid in range(1, 151):
        if qid not in entries:
            raise ValueError(f"Missing question ID {qid}")
        dataset.append(entries[qid])

    print(f"\nCompleted assembly of {len(dataset)} questions.")

    # Validation checks
    assert len(dataset) == 150, f"Expected 150, got {len(dataset)}"
    status_counts = {"verified": 0, "partially_supported": 0, "not_found": 0}
    mapped_chunk_ids = set()
    covered_sources = set()
    multi_chunk_questions = []

    for item in dataset:
        status_counts[item['ground_truth_status']] += 1
        if item['ground_truth_status'] != 'not_found':
            assert item['expected_source'] is not None
            assert len(item['expected_relevant_chunk_ids']) > 0
            assert len(item['relevant_keywords']) > 0
            assert item['expected_answer'] is not None
            covered_sources.add(item['expected_source'])
            for cid in item['expected_relevant_chunk_ids']:
                assert cid in chunks_by_id, f"Invalid chunk_id {cid}"
                mapped_chunk_ids.add(cid)
            if len(item['expected_relevant_chunk_ids']) > 1:
                multi_chunk_questions.append((item['id'], item['expected_source'], len(item['expected_relevant_chunk_ids'])))
        else:
            assert item['expected_source'] is None
            assert item['expected_answer'] is None
            assert item['expected_relevant_chunk_ids'] == []
            assert item['relevant_keywords'] == []
            assert item['expected_pages'] == []

    print("\n--- VALIDATION SUMMARY ---")
    print(f"Total Questions: {len(dataset)}")
    print(f"Verified: {status_counts['verified']}")
    print(f"Partially Supported: {status_counts['partially_supported']}")
    print(f"Not Found: {status_counts['not_found']}")
    print(f"Covered Source Documents: {len(covered_sources)}")
    print(f"Unique Chunks Mapped: {len(mapped_chunk_ids)}")
    print(f"Multi-Chunk Questions: {len(multi_chunk_questions)}")

    # Write output file
    output_path = Path('tests/retrieval_evaluation_dataset.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully wrote {output_path} ({output_path.stat().st_size} bytes)")

if __name__ == '__main__':
    main()

