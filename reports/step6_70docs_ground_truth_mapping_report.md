# Step 2 — Ground Truth Mapping Audit Report (68-Document Corpus)

> **Benchmark Session:** `benchmark_70docs_v1`  
> **Corpus Directory:** `docs_70/` (68 files, 85 Document objects, 181 chunks)  
> **Input Question File:** `tests/data/retrieval_questions_input.json` (150 questions)  
> **Output Dataset:** `tests/retrieval_evaluation_dataset.json`  
> **Embedding Model:** Google Gemini `models/gemini-embedding-001` (3072 dimensions, zero API calls during Step 2)  

---

## 1. Executive Summary

In this audit, a ground-truth mapping was executed across all **150 questions** provided in `tests/data/retrieval_questions_input.json` against the **181 chunks** in the verified FAISS vector database (`faiss_index/benchmark_70docs_v1/index.pkl`).

### Key Results Summary

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Total Questions Evaluated** | **150** | **100.0%** |
| **Verified (Fully Grounded)** | **128** | **85.33%** |
| **Partially Supported (Incomplete Evidence)** | **5** | **3.33%** |
| **Not Found (Out-of-Corpus / Negative Controls)** | **17** | **11.33%** |
| **Covered Source Documents** | **32** / 68 | 47.06% |
| **Unique Chunks Mapped to Ground Truth** | **70** / 181 | 38.67% |
| **Multi-Chunk Ground Truth Questions** | **15** | 10.00% |

---

## 2. Integrity and Safety Verification

1. **Zero External API Calls:** No calls were made to the Google Gemini API or any external inference provider. Ground truth extraction was performed purely against local serialized FAISS docstores (`index.pkl`).
2. **Zero Modification to FAISS Index:** All index files (`faiss_index/benchmark_70docs_v1/index.faiss` and `index.pkl`) remained strictly read-only and unmutated.
3. **Complete Hash & ID Validation:** Every `chunk_id` in `expected_relevant_chunk_ids` corresponds to an authentic 64-character SHA-256 identifier verified in the FAISS index docstore.
4. **Exact Question Preservation:** All 150 questions preserved their exact wording, casing, and numerical IDs without modification.
5. **No Model Hallucination:** Out-of-corpus queries were explicitly mapped to `ground_truth_status: "not_found"` with `expected_source: null` and `expected_answer: null`.

---

## 3. Analysis of Ground Truth Categories

### 3.1. Verified Questions (128 Questions)
Verified questions are fully and directly answered by explicit factual statements within the corpus. For each verified question:
- A detailed reference answer was extracted verbatim or synthesized exclusively from the corresponding chunk text.
- Exact 64-character SHA-256 chunk IDs were assigned.
- Source document filename and page numbers were recorded.
- Relevant keywords (3-6 domain terms) were cataloged for downstream evaluation.

### 3.2. Partially Supported Questions (5 Questions)
These questions contain premises or sub-questions that are only partially answered in the documents, serving as crucial boundary tests for retrieval faithfulness:

| ID | Question Snippet | Expected Source | Document Evidence Present | Missing / Unsupported Information |
| :---: | :--- | :--- | :--- | :--- |
| **44** | What is the difference between a Horizontal Pod Autoscaler (... | `Configuring K8s HPA for Custom Metrics.pdf` | Full HPA custom metrics pipeline, RabbitMQ scaling, stabilization windows | Cluster Autoscaler (node-level scaling) is entirely absent from the corpus. |
| **78** | How did the Iconoclastic Controversy fracture Byzantine soci... | `byzantine_empire_and_justinian_code.txt` | Theological, linguistic, and political disputes; Great Schism of 1054 between Eastern Orthodox and Roman Catholic churches | Specific history and societal fracture of the Iconoclastic Controversy are not mentioned. |
| **79** | In what ways did Byzantine diplomacy and the 'theme system' ... | `byzantine_empire_and_justinian_code.txt` | Professional standing military successfully repelling Persian, Arab, and Slavic incursions | Specific provincial military 'theme system' and Byzantine diplomatic mechanisms are not detailed. |
| **90** | What were the immediate cultural and architectural impacts o... | `aztec_empire_spanish_conquest.txt` | Fall of Tenochtitlan, surrender of Cuauhtémoc (1521), inception of 300 years of Spanish colonial rule, and Tenochtitlan becoming Mexico City | Specific architectural demolition of the Templo Mayor and construction details of Mexico City atop ruins are omitted. |
| **98** | How did Gosplan (the State Planning Committee) calculate mat... | `soviet_command_economy_and_five_year_plans.txt` | Gosplan directing the command economy, setting production targets, and the inability to efficiently allocate resources | Mathematical method of calculating 'material balances' is omitted. |

### 3.3. Not Found / Out-of-Corpus Questions (17 Questions)
The evaluation input intentionally included negative control questions covering topics outside the 68 documents. These questions have no supporting chunks in the FAISS index:

| ID | Question Topic | Target Technical Concept | Closest Corpus File | Audit Finding |
| :---: | :--- | :--- | :--- | :--- |
| **27** | RESTful Architecture | HATEOAS & 6 architectural constraints | `Idempotency in RESTful API Design.pdf` | File focuses exclusively on idempotency and Idempotency-Key headers; HATEOAS is absent. |
| **28** | Python Concurrency | Global Interpreter Lock (GIL) | `Implementing Asynchronous Generators in Python.pdf` | File covers async def / yield; GIL is absent from corpus. |
| **29** | Backend Architecture | API Gateway vs direct microservice calls | `None` | No API gateway document exists in corpus. |
| **30** | API Rate Limiting | Token Bucket & Leaky Bucket algorithms | `None` | No rate-limiting algorithm document exists in corpus. |
| **31** | Database Concurrency | PostgreSQL MVCC (Multi-Version Concurrency Control) | `Optimizing PostgreSQL JSONB Query Performance.pdf` | File covers JSONB decomposition & GIN/B-tree indexes; MVCC is absent. |
| **32** | In-Memory Caching | Redis latency & durability without AOF/RDB | `Redis Cluster Data Sharding and High Availability.pdf` | File covers Hash Slots & CRC16; AOF/RDB persistence is absent. |
| **33** | Distributed Systems | CAP Theorem trade-offs during partitions | `None` | CAP Theorem is absent from all corpus documents. |
| **34** | Database Reliability | PostgreSQL Write-Ahead Logging (WAL) | `None` | WAL is absent from corpus. |
| **38** | Migration Architecture | Strangler Fig pattern for monoliths | `The Saga Pattern for Distributed Transactions.pdf` | Strangler Fig pattern is absent from all microservices documents. |
| **39** | Service Mesh | mTLS, retries, and telemetry sidecar offloading | `Circuit Breaker Pattern in Microservices.pdf` | File mentions Istio briefly for resilience, but mTLS/telemetry sidecar offloading is absent. |
| **40** | Distributed Consistency | Eventual consistency latency vs strong consistency | `None` | Eventual consistency comparative analysis is absent. |
| **41** | Kubernetes Internals | Control plane reconciliation: API Server & Controller Manager | `Configuring K8s HPA for Custom Metrics.pdf` | File covers HPA custom metrics; control plane reconciliation is absent. |
| **43** | Kubernetes Ingress | Ingress controllers dynamic HTTP/HTTPS routing | `None` | Ingress controller routing rules are absent from corpus. |
| **45** | Kubernetes Multi-Tenancy | Namespaces and ResourceQuotas resource starvation | `None` | Namespaces & ResourceQuotas are absent from corpus. |
| **49** | Cloud Security | AWS/Azure IAM policies for serverless Least Privilege | `DOC-DEV-1003.docx` | Document covers CI/CD and OIDC; serverless IAM policies are absent. |
| **50** | Zero Trust Architecture | Software-Defined Perimeter (SDP) infrastructure hiding | `Zero Trust Network Access (ZTNA) Implementation.pdf` | File covers Identity-Aware Proxies; SDP is absent from corpus. |
| **80** | Byzantine History | Fourth Crusade 1204 sack of Constantinople | `byzantine_empire_and_justinian_code.txt` | Document covers 476 CE to 1453 CE, but Fourth Crusade and 1204 sack are omitted. |

---

## 4. Multi-Chunk Questions Analysis

A total of **15 questions** require synthesizing factual evidence spanning **2 distinct chunks**. This occurs when technical mechanisms span section boundaries or require both architecture and implementation details:

| ID | Question | Source Document | Relevant Chunk IDs | Pages | Evidence Synthesis Details |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **3** | What specific claims are required to be validated ... | `DOC-SEC-601.docx` | `52bdf06f...`, `43b797c8...` | N/A | Evidence spans conceptual definition and implementation mechanics. |
| **26** | How do Python's advanced metaprogramming features,... | `Advanced Metaprogramming with Python Metaclasses.pdf` | `6dabecdc...`, `eceeeb62...` | [0, 1] | Evidence spans conceptual definition and implementation mechanics. |
| **35** | How do Redis clusters utilize hash slots to automa... | `Redis Cluster Data Sharding and High Availability.pdf` | `8fe98fe0...`, `d677ad5a...` | [0] | Evidence spans conceptual definition and implementation mechanics. |
| **36** | What are the core principles of the Saga pattern, ... | `The Saga Pattern for Distributed Transactions.pdf` | `01d5ddb2...`, `b30cf956...` | [0, 1] | Evidence spans conceptual definition and implementation mechanics. |
| **37** | How does the Circuit Breaker pattern prevent casca... | `Circuit Breaker Pattern in Microservices.pdf` | `8997c0ee...`, `e1f0e104...` | [0] | Evidence spans conceptual definition and implementation mechanics. |
| **42** | What is the function of the Raft consensus algorit... | `Paxos and Raft Consensus Algorithms.pdf` | `fa81afb5...`, `292066dc...` | [0] | Evidence spans conceptual definition and implementation mechanics. |
| **44** | What is the difference between a Horizontal Pod Au... | `Configuring K8s HPA for Custom Metrics.pdf` | `e320f233...`, `2a4f819b...` | [0, 1] | Evidence spans conceptual definition and implementation mechanics. |
| **46** | What are the core tenets of the Zero Trust securit... | `Zero Trust Network Access (ZTNA) Implementation.pdf` | `6896e97a...`, `abbd14f4...` | [0] | Evidence spans conceptual definition and implementation mechanics. |
| **47** | How does continuous authentication in a Zero Trust... | `Zero Trust Network Access (ZTNA) Implementation.pdf` | `6896e97a...`, `abbd14f4...` | [0] | Evidence spans conceptual definition and implementation mechanics. |
| **65** | How did the state-sponsored industrialization mant... | `DOC-HIS-2318.docx` | `aeb5d6a8...`, `2a08c3f6...` | N/A | Evidence spans conceptual definition and implementation mechanics. |
| **83** | How did the widespread use of mercenary armies dur... | `thirty_years_war_and_westphalia.txt` | `27bd1db7...`, `a2205e36...` | N/A | Evidence spans conceptual definition and implementation mechanics. |
| **90** | What were the immediate cultural and architectural... | `aztec_empire_spanish_conquest.txt` | `8c8f6d1f...`, `a07375b5...` | N/A | Evidence spans conceptual definition and implementation mechanics. |
| **92** | What strategic military tactics did Toussaint Louv... | `the_haitian_revolution.txt` | `1b4bcd0b...`, `c1600d89...` | N/A | Evidence spans conceptual definition and implementation mechanics. |
| **97** | What were the devastating humanitarian and agricul... | `soviet_command_economy_and_five_year_plans.txt` | `dda33a0b...`, `8816857c...` | N/A | Evidence spans conceptual definition and implementation mechanics. |
| **98** | How did Gosplan (the State Planning Committee) cal... | `soviet_command_economy_and_five_year_plans.txt` | `dda33a0b...`, `8816857c...` | N/A | Evidence spans conceptual definition and implementation mechanics. |

---

## 5. Source Document Coverage Matrix

The 133 answerable questions (verified + partially supported) cover **32 distinct technical and historical source documents** across the corpus:

| Source Document | Type | Chunks in Index | Ground Truth Questions | Primary Domain |
| :--- | :---: | :---: | :---: | :--- |
| `DOC-SEC-601.docx` | DOCX | 2 | **5** | Distributed Systems & Architecture |
| `DOC-ML-705.docx` | DOCX | 2 | **5** | Distributed Systems & Architecture |
| `DOC-NET-802.docx` | DOCX | 2 | **5** | Distributed Systems & Architecture |
| `DOC-ARC-901.docx` | DOCX | 2 | **5** | Distributed Systems & Architecture |
| `DOC-DEV-1003.docx` | DOCX | 2 | **5** | Distributed Systems & Architecture |
| `DOC-HIS-2116.docx` | DOCX | 2 | **5** | History |
| `DOC-HIS-2217.docx` | DOCX | 2 | **5** | History |
| `DOC-HIS-2318.docx` | DOCX | 2 | **5** | History |
| `DOC-HIS-2419.docx` | DOCX | 2 | **5** | History |
| `DOC-HIS-2520.docx` | DOCX | 2 | **5** | History |
| `thirty_years_war_and_westphalia.txt` | TXT | 2 | **5** | History |
| `aztec_empire_spanish_conquest.txt` | TXT | 2 | **5** | History |
| `the_haitian_revolution.txt` | TXT | 2 | **5** | History |
| `soviet_command_economy_and_five_year_plans.txt` | TXT | 2 | **5** | Distributed Systems & Architecture |
| `Automotive_CAN_Bus_Architecture_and_Diagnostics.docx` | DOCX | 4 | **5** | Automotive Engineering |
| `Electric_Vehicle_Battery_Management_Systems_BMS.docx` | DOCX | 5 | **5** | Automotive Engineering |
| `Autonomous_Driving_Sensor_Fusion_and_Perception.docx` | DOCX | 4 | **5** | Automotive Engineering |
| `Internal_Combustion_Engine_Thermodynamics.docx` | DOCX | 4 | **5** | Automotive Engineering |
| `Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx` | DOCX | 4 | **5** | Automotive Engineering |
| `FCEV_Hydrogen_Storage_and_Stack_Architecture.docx` | DOCX | 4 | **5** | Automotive Engineering |
| `Automotive_ECU_Architecture_and_Microcontrollers.docx` | DOCX | 4 | **5** | Automotive Engineering |
| `Automotive_Tribology_Friction_and_Wear.docx` | DOCX | 4 | **5** | Automotive Engineering |
| `Automotive_Crashworthiness_and_Structural_Safety.docx` | DOCX | 4 | **5** | Automotive Engineering |
| `Software_Defined_Vehicle_SDV_and_OTA_Updates.docx` | DOCX | 4 | **5** | Automotive Engineering |
| `byzantine_empire_and_justinian_code.txt` | TXT | 2 | **4** | History |
| `Zero Trust Network Access (ZTNA) Implementation.pdf` | PDF | 2 | **3** | Distributed Systems & Architecture |
| `Advanced Metaprogramming with Python Metaclasses.pdf` | PDF | 2 | **1** | Distributed Systems & Architecture |
| `Redis Cluster Data Sharding and High Availability.pdf` | PDF | 2 | **1** | Distributed Systems & Architecture |
| `The Saga Pattern for Distributed Transactions.pdf` | PDF | 2 | **1** | Distributed Systems & Architecture |
| `Circuit Breaker Pattern in Microservices.pdf` | PDF | 2 | **1** | Distributed Systems & Architecture |
| `Paxos and Raft Consensus Algorithms.pdf` | PDF | 2 | **1** | Distributed Systems & Architecture |
| `Configuring K8s HPA for Custom Metrics.pdf` | PDF | 2 | **1** | Distributed Systems & Architecture |

---

## 6. Complete Question-by-Question Ground Truth Mapping Table

| ID | Question Snippet | Expected Source | Chunks | Pages | Status |
| :---: | :--- | :--- | :---: | :---: | :---: |
|   1 | What is the fundamental difference between authentication and aut... | `DOC-SEC-601.docx` | 1 | - | **verified** |
|   2 | How does the Proof Key for Code Exchange (PKCE) extension dynamic... | `DOC-SEC-601.docx` | 1 | - | **verified** |
|   3 | What specific claims are required to be validated by resource ser... | `DOC-SEC-601.docx` | 2 | - | **verified** |
|   4 | Why is the Implicit Grant type now deprecated in modern OAuth 2.0... | `DOC-SEC-601.docx` | 1 | - | **verified** |
|   5 | How do granular custom scopes in OAuth 2.0 help enforce the Princ... | `DOC-SEC-601.docx` | 1 | - | **verified** |
|   6 | How does the scaled dot-product attention mechanism in Transforme... | `DOC-ML-705.docx` | 1 | - | **verified** |
|   7 | In what way does Multi-Head Attention allow a machine learning mo... | `DOC-ML-705.docx` | 1 | - | **verified** |
|   8 | Why are positional encodings mandatory at the bottom of the encod... | `DOC-ML-705.docx` | 1 | - | **verified** |
|   9 | What role does masked self-attention play in the autoregressive g... | `DOC-ML-705.docx` | 1 | - | **verified** |
|  10 | How do recent advancements like FlashAttention address the quadra... | `DOC-ML-705.docx` | 1 | - | **verified** |
|  11 | How does the Border Gateway Protocol (BGP) evaluate complex path ... | `DOC-NET-802.docx` | 1 | - | **verified** |
|  12 | What is the primary architectural limitation regarding transitivi... | `DOC-NET-802.docx` | 1 | - | **verified** |
|  13 | How do BGP communities allow network administrators to tag routes... | `DOC-NET-802.docx` | 1 | - | **verified** |
|  14 | Why does a hub-and-spoke networking architecture utilizing Transi... | `DOC-NET-802.docx` | 1 | - | **verified** |
|  15 | What distinguishes eBGP (External BGP) from iBGP (Internal BGP) w... | `DOC-NET-802.docx` | 1 | - | **verified** |
|  16 | In an Event-Driven Architecture, how does Apache Kafka utilize pa... | `DOC-ARC-901.docx` | 1 | - | **verified** |
|  17 | How do consumer groups in Kafka facilitate massive scalability, a... | `DOC-ARC-901.docx` | 1 | - | **verified** |
|  18 | What specific guarantees does Kafka's transactional API provide t... | `DOC-ARC-901.docx` | 1 | - | **verified** |
|  19 | Why is Log Compaction an essential retention strategy for maintai... | `DOC-ARC-901.docx` | 1 | - | **verified** |
|  20 | What operational burdens are eliminated by modern Kafka deploymen... | `DOC-ARC-901.docx` | 1 | - | **verified** |
|  21 | Why are CI/CD pipelines considered prime targets for supply chain... | `DOC-DEV-1003.docx` | 1 | - | **verified** |
|  22 | How does utilizing OpenID Connect (OIDC) to federate identity bet... | `DOC-DEV-1003.docx` | 1 | - | **verified** |
|  23 | What is the operational difference between Static Application Sec... | `DOC-DEV-1003.docx` | 1 | - | **verified** |
|  24 | How does applying the Principle of Least Privilege to ephemeral C... | `DOC-DEV-1003.docx` | 1 | - | **verified** |
|  25 | What role do frameworks like In-Toto and Sigstore play in generat... | `DOC-DEV-1003.docx` | 1 | - | **verified** |
|  26 | How do Python's advanced metaprogramming features, such as metacl... | `Advanced Metaprogramming with Python Metaclasses.pdf` | 2 | [0, 1] | **verified** |
|  27 | What are the primary architectural constraints of a RESTful API, ... | *None* | 0 | - | **not_found** |
|  28 | How does Python's Global Interpreter Lock (GIL) impact the perfor... | *None* | 0 | - | **not_found** |
|  29 | In the context of backend architecture, what are the trade-offs o... | *None* | 0 | - | **not_found** |
|  30 | How can implementing rate limiting algorithms like Token Bucket o... | *None* | 0 | - | **not_found** |
|  31 | What specific advantages does PostgreSQL's Multi-Version Concurre... | *None* | 0 | - | **not_found** |
|  32 | How does Redis achieve its ultra-low latency, and what are the du... | *None* | 0 | - | **not_found** |
|  33 | When designing a distributed database system, how does the CAP th... | *None* | 0 | - | **not_found** |
|  34 | What is the role of Write-Ahead Logging (WAL) in PostgreSQL, and ... | *None* | 0 | - | **not_found** |
|  35 | How do Redis clusters utilize hash slots to automatically partiti... | `Redis Cluster Data Sharding and High Availability.pdf` | 2 | [0] | **verified** |
|  36 | What are the core principles of the Saga pattern, and how does it... | `The Saga Pattern for Distributed Transactions.pdf` | 2 | [0, 1] | **verified** |
|  37 | How does the Circuit Breaker pattern prevent cascading failures i... | `Circuit Breaker Pattern in Microservices.pdf` | 2 | [0] | **verified** |
|  38 | Why is the Strangler Fig pattern considered a low-risk strategy f... | *None* | 0 | - | **not_found** |
|  39 | What role does Service Mesh architecture play in offloading mutua... | *None* | 0 | - | **not_found** |
|  40 | In system design, how does implementing eventual consistency redu... | *None* | 0 | - | **not_found** |
|  41 | How do Kubernetes control plane components, such as the API Serve... | *None* | 0 | - | **not_found** |
|  42 | What is the function of the Raft consensus algorithm within etcd,... | `Paxos and Raft Consensus Algorithms.pdf` | 2 | [0] | **verified** |
|  43 | How do Kubernetes Ingress controllers dynamically route external ... | *None* | 0 | - | **not_found** |
|  44 | What is the difference between a Horizontal Pod Autoscaler (HPA) ... | `Configuring K8s HPA for Custom Metrics.pdf` | 2 | [0, 1] | **partially_supported** |
|  45 | How do Kubernetes namespaces and ResourceQuotas provide logical i... | *None* | 0 | - | **not_found** |
|  46 | What are the core tenets of the Zero Trust security model, and ho... | `Zero Trust Network Access (ZTNA) Implementation.pdf` | 2 | [0] | **verified** |
|  47 | How does continuous authentication in a Zero Trust architecture e... | `Zero Trust Network Access (ZTNA) Implementation.pdf` | 2 | [0] | **verified** |
|  48 | Why is micro-segmentation critical in a modern cloud environment,... | `Zero Trust Network Access (ZTNA) Implementation.pdf` | 1 | [0] | **verified** |
|  49 | How do Identity and Access Management (IAM) policies in AWS or Az... | *None* | 0 | - | **not_found** |
|  50 | What is the role of a Software-Defined Perimeter (SDP) in hiding ... | *None* | 0 | - | **not_found** |
|  51 | How did James Watt's improvements to the steam engine act as a ca... | `DOC-HIS-2116.docx` | 1 | - | **verified** |
|  52 | What were the immediate public health consequences of the rapid, ... | `DOC-HIS-2116.docx` | 1 | - | **verified** |
|  53 | How did the shift from the domestic cottage industry to the centr... | `DOC-HIS-2116.docx` | 1 | - | **verified** |
|  54 | In what ways did the accumulation of industrial capital drive 19t... | `DOC-HIS-2116.docx` | 1 | - | **verified** |
|  55 | How did the mass production of high-quality wrought iron via the ... | `DOC-HIS-2116.docx` | 1 | - | **verified** |
|  56 | How did the cessation of territorial expansion in the 2nd century... | `DOC-HIS-2217.docx` | 1 | - | **verified** |
|  57 | What were the long-term socioeconomic effects of the Roman state ... | `DOC-HIS-2217.docx` | 1 | - | **verified** |
|  58 | How did the 'barbarization' of the Roman legions with Germanic fo... | `DOC-HIS-2217.docx` | 1 | - | **verified** |
|  59 | Why did Diocletian’s division of the empire ultimately doom the W... | `DOC-HIS-2217.docx` | 1 | - | **verified** |
|  60 | In what way did the westward expansion of the Huns act as the pri... | `DOC-HIS-2217.docx` | 1 | - | **verified** |
|  61 | How did the arrival of Commodore Perry's 'Black Ships' expose the... | `DOC-HIS-2318.docx` | 1 | - | **verified** |
|  62 | What were the social consequences of dismantling the rigid four-t... | `DOC-HIS-2318.docx` | 1 | - | **verified** |
|  63 | How did the Meiji oligarchs utilize the Iwakura Mission to rapidl... | `DOC-HIS-2318.docx` | 1 | - | **verified** |
|  64 | In what ways did the Meiji Constitution of 1889 blend Western bic... | `DOC-HIS-2318.docx` | 1 | - | **verified** |
|  65 | How did the state-sponsored industrialization mantra of 'Fukoku K... | `DOC-HIS-2318.docx` | 2 | - | **verified** |
|  66 | How did the deeply regressive tax system of the Ancien Régime pla... | `DOC-HIS-2419.docx` | 1 | - | **verified** |
|  67 | What role did the Enlightenment philosophies of Rousseau and Lock... | `DOC-HIS-2419.docx` | 1 | - | **verified** |
|  68 | How did the outbreak of war against European monarchies accelerat... | `DOC-HIS-2419.docx` | 1 | - | **verified** |
|  69 | What were the primary political and social justifications used by... | `DOC-HIS-2419.docx` | 1 | - | **verified** |
|  70 | How did Napoleon Bonaparte's rise to power consolidate revolution... | `DOC-HIS-2419.docx` | 1 | - | **verified** |
|  71 | How did the logistics of the Silk Road rely on a relay system of ... | `DOC-HIS-2520.docx` | 1 | - | **verified** |
|  72 | In what ways did the Silk Road serve as the primary conduit for t... | `DOC-HIS-2520.docx` | 1 | - | **verified** |
|  73 | How did the westward transmission of Chinese papermaking technolo... | `DOC-HIS-2520.docx` | 1 | - | **verified** |
|  74 | What role did merchant caravans on the Silk Road play as epidemio... | `DOC-HIS-2520.docx` | 1 | - | **verified** |
|  75 | How did the collapse of the Mongol Empire's Pax Mongolica contrib... | `DOC-HIS-2520.docx` | 1 | - | **verified** |
|  76 | How did the Corpus Juris Civilis (Justinian Code) rationalize cen... | `byzantine_empire_and_justinian_code.txt` | 1 | - | **verified** |
|  77 | What strategic advantages allowed the Byzantine Empire and its ca... | `byzantine_empire_and_justinian_code.txt` | 1 | - | **verified** |
|  78 | How did the Iconoclastic Controversy fracture Byzantine society a... | `byzantine_empire_and_justinian_code.txt` | 1 | - | **partially_supported** |
|  79 | In what ways did Byzantine diplomacy and the 'theme system' of mi... | `byzantine_empire_and_justinian_code.txt` | 1 | - | **partially_supported** |
|  80 | What were the catastrophic geopolitical consequences of the Fourt... | *None* | 0 | - | **not_found** |
|  81 | How did the Defenestration of Prague ignite the underlying religi... | `thirty_years_war_and_westphalia.txt` | 1 | - | **verified** |
|  82 | Why did Catholic France, under Cardinal Richelieu, intervene on t... | `thirty_years_war_and_westphalia.txt` | 1 | - | **verified** |
|  83 | How did the widespread use of mercenary armies during the Thirty ... | `thirty_years_war_and_westphalia.txt` | 2 | - | **verified** |
|  84 | In what way did the Peace of Westphalia (1648) establish the mode... | `thirty_years_war_and_westphalia.txt` | 1 | - | **verified** |
|  85 | How did the conclusion of the Thirty Years' War permanently cripp... | `thirty_years_war_and_westphalia.txt` | 1 | - | **verified** |
|  86 | How did the Aztec Empire utilize the 'chinampa' system of agricul... | `aztec_empire_spanish_conquest.txt` | 1 | - | **verified** |
|  87 | What role did the Aztec practice of extracting heavy tribute and ... | `aztec_empire_spanish_conquest.txt` | 1 | - | **verified** |
|  88 | How did the introduction of Old World pathogens, particularly sma... | `aztec_empire_spanish_conquest.txt` | 1 | - | **verified** |
|  89 | In what ways did superior Spanish metallurgy, combined with the u... | `aztec_empire_spanish_conquest.txt` | 1 | - | **verified** |
|  90 | What were the immediate cultural and architectural impacts of the... | `aztec_empire_spanish_conquest.txt` | 2 | - | **partially_supported** |
|  91 | How did the brutal conditions of the French sugar colony of Saint... | `the_haitian_revolution.txt` | 1 | - | **verified** |
|  92 | What strategic military tactics did Toussaint Louverture employ t... | `the_haitian_revolution.txt` | 2 | - | **verified** |
|  93 | How did the ideological rhetoric of the French Revolution inadver... | `the_haitian_revolution.txt` | 1 | - | **verified** |
|  94 | Why did Napoleon Bonaparte's attempt to reinstate slavery in the ... | `the_haitian_revolution.txt` | 1 | - | **verified** |
|  95 | What were the long-term economic consequences of the global diplo... | `the_haitian_revolution.txt` | 1 | - | **verified** |
|  96 | How did Stalin's first Five-Year Plan attempt to radically transf... | `soviet_command_economy_and_five_year_plans.txt` | 1 | - | **verified** |
|  97 | What were the devastating humanitarian and agricultural consequen... | `soviet_command_economy_and_five_year_plans.txt` | 2 | - | **verified** |
|  98 | How did Gosplan (the State Planning Committee) calculate material... | `soviet_command_economy_and_five_year_plans.txt` | 2 | - | **partially_supported** |
|  99 | In what ways did the Soviet command economy struggle with innovat... | `soviet_command_economy_and_five_year_plans.txt` | 1 | - | **verified** |
| 100 | How did the prioritization of heavy industry and military product... | `soviet_command_economy_and_five_year_plans.txt` | 1 | - | **verified** |
| 101 | How does the non-destructive bitwise arbitration mechanism in the... | `Automotive_CAN_Bus_Architecture_and_Diagnostics.docx` | 1 | - | **verified** |
| 102 | Why was the transition from Classical CAN to CAN FD (Flexible Dat... | `Automotive_CAN_Bus_Architecture_and_Diagnostics.docx` | 1 | - | **verified** |
| 103 | In what way do Zonal Architectures simplify physical wiring harne... | `Automotive_CAN_Bus_Architecture_and_Diagnostics.docx` | 1 | - | **verified** |
| 104 | How does the Secure Onboard Communication (SecOC) module utilize ... | `Automotive_CAN_Bus_Architecture_and_Diagnostics.docx` | 1 | - | **verified** |
| 105 | Why is the strict master-slave architecture of the Local Intercon... | `Automotive_CAN_Bus_Architecture_and_Diagnostics.docx` | 1 | - | **verified** |
| 106 | How do advanced Battery Management Systems (BMS) utilize Extended... | `Electric_Vehicle_Battery_Management_Systems_BMS.docx` | 1 | - | **verified** |
| 107 | What is the mechanical difference between passive cell balancing ... | `Electric_Vehicle_Battery_Management_Systems_BMS.docx` | 1 | - | **verified** |
| 108 | How does a BMS rapidly detect the onset of thermal runaway, and w... | `Electric_Vehicle_Battery_Management_Systems_BMS.docx` | 1 | - | **verified** |
| 109 | Why must the BMS manage pre-charge circuits before closing the ma... | `Electric_Vehicle_Battery_Management_Systems_BMS.docx` | 1 | - | **verified** |
| 110 | How does continuous monitoring of high-voltage isolation resistan... | `Electric_Vehicle_Battery_Management_Systems_BMS.docx` | 1 | - | **verified** |
| 111 | What are the inherent environmental limitations of automotive cam... | `Autonomous_Driving_Sensor_Fusion_and_Perception.docx` | 1 | - | **verified** |
| 112 | How do early sensor fusion algorithms utilize deep neural network... | `Autonomous_Driving_Sensor_Fusion_and_Perception.docx` | 1 | - | **verified** |
| 113 | Why is standard GPS insufficient for autonomous lane positioning,... | `Autonomous_Driving_Sensor_Fusion_and_Perception.docx` | 1 | - | **verified** |
| 114 | In autonomous motion planning, how does the system balance comple... | `Autonomous_Driving_Sensor_Fusion_and_Perception.docx` | 1 | - | **verified** |
| 115 | What is the function of a Model Predictive Control (MPC) algorith... | `Autonomous_Driving_Sensor_Fusion_and_Perception.docx` | 1 | - | **verified** |
| 116 | How do the Miller and Atkinson thermodynamic cycles extract more ... | `Internal_Combustion_Engine_Thermodynamics.docx` | 1 | - | **verified** |
| 117 | How does Gasoline Direct Injection (GDI) cool the cylinder charge... | `Internal_Combustion_Engine_Thermodynamics.docx` | 1 | - | **verified** |
| 118 | What role does Variable Geometry Turbine (VGT) technology play in... | `Internal_Combustion_Engine_Thermodynamics.docx` | 1 | - | **verified** |
| 119 | Why do Three-Way Catalytic Converters (TWC) require the Engine Co... | `Internal_Combustion_Engine_Thermodynamics.docx` | 1 | - | **verified** |
| 120 | How do Selective Catalytic Reduction (SCR) systems utilize Diesel... | `Internal_Combustion_Engine_Thermodynamics.docx` | 1 | - | **verified** |
| 121 | How does flow separation at the rear of a vehicle create a chaoti... | `Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx` | 1 | - | **verified** |
| 122 | In what ways do active aerodynamic features, such as deployable r... | `Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx` | 1 | - | **verified** |
| 123 | How does accelerating air under a smooth vehicle undercarriage ge... | `Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx` | 1 | - | **verified** |
| 124 | What role do advanced turbulence models like Detached Eddy Simula... | `Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx` | 1 | - | **verified** |
| 125 | Why are geometric features like Kammbacks and vortex generators u... | `Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx` | 1 | - | **verified** |
| 126 | How does the platinum catalyst at the anode of a PEM fuel cell sp... | `FCEV_Hydrogen_Storage_and_Stack_Architecture.docx` | 1 | - | **verified** |
| 127 | Why is a robust Air Management System containing chemical filters... | `FCEV_Hydrogen_Storage_and_Stack_Architecture.docx` | 1 | - | **verified** |
| 128 | How do Type IV composite cylinders utilize high-density polymer l... | `FCEV_Hydrogen_Storage_and_Stack_Architecture.docx` | 1 | - | **verified** |
| 129 | What is the function of the Thermal Pressure Relief Device (TPRD)... | `FCEV_Hydrogen_Storage_and_Stack_Architecture.docx` | 1 | - | **verified** |
| 130 | Why are Fuel Cell Electric Vehicles (FCEVs) fundamentally hybrid ... | `FCEV_Hydrogen_Storage_and_Stack_Architecture.docx` | 1 | - | **verified** |
| 131 | How does the Dual-Core Lockstep (DCLS) architecture within automo... | `Automotive_ECU_Architecture_and_Microcontrollers.docx` | 1 | - | **verified** |
| 132 | What is the primary function of Error-Correcting Code (ECC) logic... | `Automotive_ECU_Architecture_and_Microcontrollers.docx` | 1 | - | **verified** |
| 133 | How does the AUTOSAR layered software architecture decouple appli... | `Automotive_ECU_Architecture_and_Microcontrollers.docx` | 1 | - | **verified** |
| 134 | Why is the automotive industry shifting from flat ECU networks to... | `Automotive_ECU_Architecture_and_Microcontrollers.docx` | 1 | - | **verified** |
| 135 | How do Pulse Width Modulation (PWM) modules allow an engine contr... | `Automotive_ECU_Architecture_and_Microcontrollers.docx` | 1 | - | **verified** |
| 136 | How does the Stribeck curve illustrate the transition from low-fr... | `Automotive_Tribology_Friction_and_Wear.docx` | 1 | - | **verified** |
| 137 | In what ways do Extreme Pressure (EP) additives like Zinc Dialkyl... | `Automotive_Tribology_Friction_and_Wear.docx` | 1 | - | **verified** |
| 138 | Why are Diamond-Like Carbon (DLC) coatings utilized on valvetrain... | `Automotive_Tribology_Friction_and_Wear.docx` | 1 | - | **verified** |
| 139 | How does the extreme rotational speed of EV traction motors induc... | `Automotive_Tribology_Friction_and_Wear.docx` | 1 | - | **verified** |
| 140 | What physical phenomenon occurs at the tire-road contact patch du... | `Automotive_Tribology_Friction_and_Wear.docx` | 1 | - | **verified** |
| 141 | How do engineered crumple zones utilize progressive plastic defor... | `Automotive_Crashworthiness_and_Structural_Safety.docx` | 1 | - | **verified** |
| 142 | Why are Ultra-High-Strength Steels (UHSS), such as hot-stamped bo... | `Automotive_Crashworthiness_and_Structural_Safety.docx` | 1 | - | **verified** |
| 143 | How do pyrotechnic pretensioners and mechanical load limiters wor... | `Automotive_Crashworthiness_and_Structural_Safety.docx` | 1 | - | **verified** |
| 144 | What is the role of solid-state MEMS accelerometers in supplying ... | `Automotive_Crashworthiness_and_Structural_Safety.docx` | 1 | - | **verified** |
| 145 | How do Finite Element Analysis (FEA) solvers running on supercomp... | `Automotive_Crashworthiness_and_Structural_Safety.docx` | 1 | - | **verified** |
| 146 | How does the transition to a Service-Oriented Architecture (SOA) ... | `Software_Defined_Vehicle_SDV_and_OTA_Updates.docx` | 1 | - | **verified** |
| 147 | In what way does a dual-bank (A/B) memory architecture guarantee ... | `Software_Defined_Vehicle_SDV_and_OTA_Updates.docx` | 1 | - | **verified** |
| 148 | How do Continuous Integration and Continuous Deployment (CI/CD) p... | `Software_Defined_Vehicle_SDV_and_OTA_Updates.docx` | 1 | - | **verified** |
| 149 | What role does the vehicle's telemetry data play in feeding the O... | `Software_Defined_Vehicle_SDV_and_OTA_Updates.docx` | 1 | - | **verified** |
| 150 | Why do international regulations like UN R156 mandate that automa... | `Software_Defined_Vehicle_SDV_and_OTA_Updates.docx` | 1 | - | **verified** |

---

## 7. Phase 3 Readiness & Recommendations

With the ground truth mapping completed and verified:
- The ground-truth dataset `tests/retrieval_evaluation_dataset.json` is ready for the Phase 3 retrieval evaluation pipeline.
- Exact recall, precision@k, MRR, NDCG, and negative-rejection metrics can be measured.
- In accordance with project instructions, **Phase 3 execution will remain paused until user review and approval**.
