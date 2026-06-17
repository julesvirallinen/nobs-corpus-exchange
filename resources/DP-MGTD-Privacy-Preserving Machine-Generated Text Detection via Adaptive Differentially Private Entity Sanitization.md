

# DP-MGTD: Privacy-Preserving Machine-Generated Text Detection via Adaptive Differentially Private Entity Sanitization

Lionel Z. Wang<sup>1,2\*</sup>, Yusheng Zhao<sup>1,3\*</sup>, Jiabin Luo<sup>1,4\*</sup>, Xinfeng Li<sup>1†</sup>, Lixu Wang<sup>1</sup>,  
Yinan Peng<sup>5</sup>, Haoyang Li<sup>2</sup>, XiaoFeng Wang<sup>1</sup>, Wei Dong<sup>1†</sup>

<sup>1</sup>Nanyang Technological University, <sup>2</sup>The Hong Kong Polytechnic University,  
<sup>3</sup>University of Science and Technology of China, <sup>4</sup>Peking University, <sup>5</sup>Hengxin Tech.

## Abstract

The deployment of Machine-Generated Text (MGT) detection systems necessitates processing sensitive user data, creating a fundamental conflict between authorship verification and privacy preservation. Standard anonymization techniques often disrupt linguistic fluency, while rigorous Differential Privacy (DP) mechanisms typically degrade the statistical signals required for accurate detection. To resolve this dilemma, we propose **DP-MGTD**, a framework incorporating an Adaptive Differentially Private Entity Sanitization algorithm. Our approach utilizes a two-stage mechanism that performs noisy frequency estimation and dynamically calibrates privacy budgets, applying Laplace and Exponential mechanisms to numerical and textual entities respectively. Crucially, we identify a counter-intuitive phenomenon where the application of DP noise amplifies the distinguishability between human and machine text by exposing distinct sensitivity patterns to perturbation. Extensive experiments on the MGTBench-2.0 dataset show that our method achieves near-perfect detection accuracy, significantly outperforming non-private baselines while satisfying strict privacy guarantees.

## 1 Introduction

The proliferation of Large Language Models (LLMs), exemplified by GPT-4 (Achiam et al., 2023) and Llama-3 (Dubey et al., 2024), has necessitated the deployment of Machine-Generated Text (MGT) detection systems to safeguard academic integrity and information authenticity (Mitchell et al., 2023; Li et al., 2024; Wu et al., 2024, 2025). Existing detection paradigms are broadly categorized into zero-shot (metric-based) methods and supervised (model-based) methods. Metric-based approaches utilize statistical signals derived from pre-trained models, such as entropy, perplexity, and

log-rank information (Gehrmann et al., 2019; Bao et al., 2023; Mitchell et al., 2023). In contrast, model-based methods fine-tune neural classifiers on labeled datasets to distinguish between human and machine authorship (Ippolito et al., 2020; Solaiman et al., 2019). These methods have achieved commendable performance, predominantly operating under the assumption of full access to raw, cleartext inputs. This assumption creates a critical tension in real-world deployment: users are increasingly required to submit sensitive documents, such as medical records (Kumichev et al., 2024) or proprietary financial reports (Dolphin et al., 2024), to third-party detection services. Consequently, the detection process itself becomes a vector for privacy leakage, exposing Personally Identifiable Information (PII) to potential interception or model memorization (Das et al., 2025; Chen et al., 2025).

Resolving this conflict is algorithmically challenging due to the inherent *privacy-utility trade-off*. Naive anonymization methods, such as masking named entities, disrupt the linguistic fluency and statistical dependencies required by detectors, precipitating a sharp decline in accuracy (Majeed and Lee, 2020). Furthermore, rigorous privacy standards like Differential Privacy (DP) (Dwork, 2006) necessitate injecting noise proportional to the data sensitivity. Excessive noise injection can distort the text distribution to an extent that renders subtle authorship signals undetectable, effectively blinding downstream classifiers.

To address these limitations, we propose a novel **Adaptive Differentially Private Entity Sanitization** framework, **DP-MGTD**, that reconciles rigorous privacy guarantees with high-performance detection. Diverging from uniform noise injection, our approach introduces an **adaptive budget allocation** mechanism. We partition sensitive information into numerical and textual entities, employing a two-stage process: first, we perform noisy frequency estimation to gauge entity density; sec-

\*Equal contribution.

†Corresponding author.

ond, we dynamically calibrate the privacy budget based on these estimates. This allows us to apply the Laplace Mechanism to numerical values and the Exponential Mechanism to textual entities with granular precision, ensuring that the sanitized text remains statistically representative while satisfying strict  $\epsilon$ -DP constraints.

Crucially, our empirical investigation reveals a counter-intuitive phenomenon: the application of our DP mechanism does not degrade detection performance but rather amplifies the distinguishability between human and machine authorship. We observe that machine-generated text exhibits distinct sensitivity patterns to DP-induced perturbations compared to human writing. By exploiting these stability dynamics, our framework transforms the privacy constraint into a discriminative feature. Extensive experiments on the MGTBench-2.0 dataset across STEM, Humanities, and Social Sciences domains demonstrate that our method significantly outperforms non-private baselines, achieving near-perfect detection accuracy in supervised settings while providing formal privacy protection.

Our contributions are summarized as follows:

- We identify the **privacy bottleneck** in current MGT detection services and formulate a general framework for integrating Differential Privacy into both metric-based and model-based detection pipelines.
- We propose an **Adaptive Differentially Private Entity Sanitization** algorithm that utilizes a hybrid noise mechanism (Laplace and Exponential) with frequency-based budget allocation to optimize the trade-off between text utility and entity privacy.
- We conduct extensive experiments on the MGTBench-2.0 dataset across diverse domains and LLMs. Empirical results demonstrate that our method not only provides robust privacy protection but also yields substantial performance improvements over non-private baselines, validating that privacy mechanisms can uncover latent distributional distinctions between human and machine text.

## 2 Related work

### 2.1 Differential Private Text Sanitization

Text sanitization aims to obfuscate sensitive information of the unstructured text while preserv-

ing utility for downstream NLP tasks, leveraging techniques such as differential privacy (DP). Token-level metric LDP approaches include SANTEXT (Yue et al., 2021), which applies the Exponential Mechanism over embedding distances, and the Truncated Exponential Mechanism (TEM) (Carvalho et al., 2023), which optimizes trade-offs by calibrating sampling to local token density. To address fixed-space limitations, CUSTEXT (Chen et al., 2023) supports customized output sets with arbitrary similarity, while CLUSANT (Awon et al., 2025) leverages LLM-assisted clustering for coherent MLDP sanitization. For black-box prompt protection, INFEDPT (Tong et al., 2025) combines perturbation with an extraction module, whereas PREEMPT (Chowdhury et al., 2025) hybridizes format-preserving encryption with metric DP. In our work, we focus on combining different DP mechanisms to word-level adapt to entities in texts, achieving adaptive privacy allocation and DP-preserving text sanitization.

### 2.2 Machine-Generated Text Detection

Existing detection methodologies generally categorize into metric-based and model-based paradigms (He et al., 2024; Liu et al., 2025). Metric-based approaches exploit zero-shot statistical artifacts, premised on the hypothesis that machine text manifests lower perplexity or entropy. Fundamental indicators include average Log-Likelihood (Solaiman et al., 2019), Rank (Gehrmann et al., 2019), and Entropy (Gehrmann et al., 2019; Mitchell et al., 2023). Advanced techniques like GLTR (Gehrmann et al., 2019) analyze top- $k$  token fractions, while Binoculars (Hans et al., 2024) utilizes robust cross-perplexity ratios. In contrast, model-based methods employ supervised classifiers, typically fine-tuning Transformer encoders such as DistilBERT (Sanh et al., 2019) and RoBERTa (Liu et al., 2019) to capture discriminative features. Leveraging these established detectors as baselines, we demonstrate that DP-MGTD consistently enhances performance independent of the underlying detection architecture.

## 3 Preliminaries and Problem Formulation

### 3.1 Task Definition

Let  $\mathcal{X}$  denote the space of discrete text sequences and  $\mathcal{Y} = \{0, 1\}$  be the label space, where 0 represents human-written text and 1 represents

machine-generated text (MGT). The standard MGT detection task aims to learn a decision function  $\mathcal{D} : \mathcal{X} \rightarrow \mathcal{Y}$  that minimizes the classification error on a given dataset.

However, in real-world deployment, the input text  $\mathbf{x} \in \mathcal{X}$  often contains sensitive information. We consider a privacy-preserving setting where the detector  $\mathcal{D}$  does not have direct access to the raw text  $\mathbf{x}$ . Instead, it operates on a sanitized version  $\tilde{\mathbf{x}}$ , produced by a sanitization mechanism  $\mathcal{M} : \mathcal{X} \rightarrow \mathcal{X}$ . The objective is transformed to learning  $\mathcal{D}(\tilde{\mathbf{x}})$  such that it maintains high detection performance while  $\mathcal{M}$  provides rigorous privacy guarantees for the sensitive entities within  $\mathbf{x}$ .

### 3.2 Threat Model and Privacy Goal

We consider a scenario where users submit their texts to a third-party service for MGT detection. The service provider (or an eavesdropper) is modeled as a *semi-honest* (or honest-but-curious) adversary who executes the detection protocol correctly but attempts to infer sensitive information.

**Privacy Goal.** Our primary goal is to protect specific sensitive entities (e.g., names, dates, financial figures) contained in the text. We aim to ensure that the presence, absence, or specific value of any single entity instance does not significantly affect the output distribution of the sanitized text. This indistinguishability prevents the adversary from reconstructing the exact sensitive values with high confidence. To achieve this, we adopt Differential Privacy (DP) as our formal privacy standard.

### 3.3 Differential Privacy Basics.

Differential privacy (DP) (Dwork, 2006) is a rigorous mathematical standard for quantifying privacy guarantees in Section 3.2. It is fundamental in privacy protection, ensuring that the output distribution of an algorithm remains statistically indistinguishable regardless of the presence or absence of any single text entity in the input. The standard DP is defined as follows:

**Definition 3.1** (Differential Privacy). A randomized mechanism  $\mathcal{M}$  satisfies  $\epsilon$ -differential privacy if for any two adjacent inputs  $x$  and  $x'$  (differing by at most one entity), and for any possible output subset  $S \subseteq \text{Range}(\mathcal{M})$ , the following inequality holds:

$$\Pr[\mathcal{M}(x) \in S] \leq \exp(\epsilon) \cdot \Pr[\mathcal{M}(x') \in S]. \quad (1)$$

To implement Definition 3.1, we employ two foundational mechanisms tailored to different data types. The magnitude of noise required is determined by the global sensitivity  $\Delta$  of a query function  $f$ , defined as:

$$\Delta = \max_{x, x'} \|f(x) - f(x')\|_1. \quad (2)$$

The **Laplace Mechanism** is designed for numerical entities. It injects noise drawn from a Laplace distribution  $\text{Lap}(\cdot)$  calibrated to the sensitivity and the privacy budget  $\epsilon$ .

**Lemma 3.1** (Laplace Mechanism). *Given a function  $f : \mathcal{X} \rightarrow \mathbb{R}^k$ , the Laplace Mechanism  $\mathcal{M}_L$  is defined as:*

$$\mathcal{M}_L(x, \epsilon) := f(x) + \eta, \quad (3)$$

where  $\eta \sim \text{Lap}(\Delta/\epsilon)^k$ . This mechanism satisfies  $\epsilon$ -DP.

Conversely, for non-numerical (textual) entities where direct noise addition is infeasible, the **Exponential Mechanism** is used. Given an input  $x$ , a set of candidates  $\mathcal{Y}$ , and a utility scoring function  $s : \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}$ , the sensitivity of  $s$  is given as:

$$\Delta = \max_{y \in \mathcal{Y}} \max_{x, x'} |s(x, y) - s(x', y)|. \quad (4)$$

**Lemma 3.2** (Exponential Mechanism). *Given inputs  $x$ , candidate set  $\mathcal{Y}$ , and score function  $s$ , the Exponential Mechanism  $\mathcal{M}_E(x, \epsilon)$  satisfies  $\epsilon$ -DP by sampling an output  $y \in \mathcal{Y}$  with probability proportional to the scaled score:*

$$\Pr[\mathcal{M}_E(x, \epsilon) = y] \propto \exp\left(\frac{\epsilon s(x, y)}{2\Delta}\right). \quad (5)$$

Finally, to handle multiple entities within a single text, we utilize the composition property of DP.

**Theorem 3.3** (Sequential Composition). *Let  $\mathcal{M}_1, \dots, \mathcal{M}_n$  be a sequence of randomized algorithms where each  $\mathcal{M}_i$  satisfies  $\epsilon_i$ -DP. The combined mechanism  $\mathcal{M}(x) = (\mathcal{M}_1(x), \dots, \mathcal{M}_n(x))$  satisfies  $(\sum_{i=1}^n \epsilon_i)$ -DP.*

## 4 Methodology: DP-MGTD Framework

### 4.1 Overview

As illustrated in Figure 1, our proposed DP-MGTD framework operates as a privacy-preserving pipeline that transforms raw text into a decision label. Formally, the detection function  $\mathcal{D}$  defined

![Figure 1: Overview of DP-MGTD. The diagram illustrates a three-stage pipeline. Stage 1: Noisy Frequency Estimation involves 'Noise Counting' (Entity Counts, Noise Injection) and 'Adaptive Parameter Control'. Stage 2: Adaptive Budget Allocation involves 'Numerical Data (Laplace Mechanism)' and 'Text Data (Exponential Mechanism)', both leading to 'Sanitization'. The final output is 'Sanitized Text', which is processed by 'Downstream Detection Strategies' including 'Metric-based Feature Extraction' and 'Model-based Feature Extraction', leading to 'Machine-Generated Text Detection' and a 'Machine vs Human' comparison.](2763901b7a1fd1b5d704cdc450d12ed0_img.jpg)

Figure 1: Overview of DP-MGTD. The diagram illustrates a three-stage pipeline. Stage 1: Noisy Frequency Estimation involves 'Noise Counting' (Entity Counts, Noise Injection) and 'Adaptive Parameter Control'. Stage 2: Adaptive Budget Allocation involves 'Numerical Data (Laplace Mechanism)' and 'Text Data (Exponential Mechanism)', both leading to 'Sanitization'. The final output is 'Sanitized Text', which is processed by 'Downstream Detection Strategies' including 'Metric-based Feature Extraction' and 'Model-based Feature Extraction', leading to 'Machine-Generated Text Detection' and a 'Machine vs Human' comparison.

Figure 1: Overview of DP-MGTD. The pipeline transforms unprotected input containing sensitive entities into sanitized representations via Adaptive Differentially Private Entity Sanitization. This core module operates in two stages: (1) *Noisy Frequency Estimation* to gauge entity density, and (2) *Adaptive Budget Allocation* to dynamically distribute the privacy budget  $\epsilon_{total}$  across text and numerical data using Exponential and Laplace mechanisms, respectively. The sanitized output serves as the input for downstream Metric-based and Model-based detection strategies, enabling robust distinction between machine-generated and human-written text while preserving privacy.

in Section 3.1 is decomposed into three sequential stages:

$$D(\mathbf{x}) = g(\phi(\mathcal{M}(\mathbf{x}, \epsilon_{total}))), \quad (6)$$

where:

- $\mathcal{M} : \mathcal{X} \times \mathbb{R}^+ \rightarrow \mathcal{X}$  is the **Adaptive Sanitization Module** (Section 4.2). It injects noise into sensitive entities within the input  $\mathbf{x}$  under a global privacy budget  $\epsilon_{total}$  to produce a sanitized version  $\tilde{\mathbf{x}}$ .
- $\phi : \mathcal{X} \rightarrow \mathbb{R}^{d \times K}$  is the **Feature Extraction Module** (Section 4.3). It maps the sanitized text  $\tilde{\mathbf{x}}$  to a temporal feature matrix based on either statistical metrics ( $d = 3$ ) or semantic embeddings ( $d = 768$ ).
- $g : \mathbb{R}^{d \times K} \rightarrow \{0, 1\}$  is the **Classifier**. It aggregates the extracted features to predict the origin of the text (Machine-generated vs. Human-written).

The complete execution flow, including budget calibration and iterative processing, is summarized in **Algorithm 1**.

### 4.2 Adaptive Differentially Private Entity Sanitization

To realize the privacy goals outlined in Section 3.2, we introduce an adaptive budget allocation scheme.

Let  $\Sigma$  be a finite token dictionary. We consider an input text sequence  $\mathbf{x} = \langle x_1, x_2, \dots, x_n \rangle$  where each token  $x_i \in \Sigma$ . The entities within  $\mathbf{x}$  are partitioned into two subsets: numerical entities  $\mathcal{E}_{num}$  and non-numerical (textual) entities  $\mathcal{E}_{text}$ .

To guarantee privacy preservation during budget calibration, we employ a **two-stage noise injection mechanism**. The total privacy budget  $\epsilon_{total}$  is split into two components:  $\epsilon_{cnt}$  for entity counting and  $\epsilon_{sub}$  for value perturbation.

**Stage 1: Noisy Frequency Estimation.** First, we estimate the frequency of each entity type  $\tau$  using the Laplace Mechanism with budget  $\epsilon_{cnt}$ . The noisy count  $\tilde{c}_\tau$  is clamped to ensure validity:

$$\tilde{c}_\tau = \max(1, \mathcal{M}_L(c_\tau, \epsilon_{cnt})). \quad (7)$$

**Stage 2: Adaptive Budget Allocation.** Second, we distribute the remaining budget  $\epsilon_{sub}$  based on these noisy counts. The allocation share is defined as  $\rho_\tau = \Delta_\tau \cdot w_\tau \cdot \tilde{c}_\tau$ , using a predefined weight  $w_\tau$  for importance adjustment. The normalized privacy budget allocated to the  $\tau$ -th component instance is given by:

$$\epsilon_\tau = \epsilon_{sub} \cdot \frac{\rho_\tau}{\tilde{c}_\tau \cdot \sum_{t \in \mathcal{T}} \rho_t}, \quad (8)$$

#### --- **Algorithm 1:** Adaptive DP-MGTD Sanitization and Detection ---

**Data:** Raw text  $x$ , List of privacy budgets  $E$ , Entity categories  $\mathcal{T}$ , Sensitivity  $\{\Delta_\tau\}_{\tau \in \mathcal{T}}$ , Entity weights  $\{w_\tau\}_{\tau \in \mathcal{T}}$ , Classifier  $g$

**Result:** Decision label  $\hat{y} \in \{0, 1\}$

```

1 Identify all entity types  $\tau \in \mathcal{T}$  and their true
  counts  $c_\tau$  in  $x$ ;
2  $V_x \leftarrow \emptyset$ ;
3 for each privacy budget  $\epsilon_{total} \in E$  do
4   Split budget:
   ( $\epsilon_{cnt}, \epsilon_{sub}$ )  $\leftarrow$  Split( $\epsilon_{total}$ );
   // Stage 1: Noisy Counting
   for each entity type  $\tau \in \mathcal{T}$  do
6      $\tilde{c}_\tau \leftarrow \max(1, \mathcal{M}_L(c_\tau, \epsilon_{cnt}))$ ;
7      $\rho_\tau \leftarrow \Delta_\tau \cdot w_\tau \cdot \tilde{c}_\tau$ ;
   // Stage 2: Adaptive Allocation
   & Perturbation
   for each entity type  $\tau \in \mathcal{T}$  do
8      $\epsilon_\tau \leftarrow \epsilon_{sub} \cdot \frac{\rho_\tau}{\sum_{t \in \mathcal{T}} \rho_t}$ ;
9     Retrieve instances  $\{x_i^{(\tau)}\}_{i=1}^{c_\tau}$  from
10     $x$ ;
11    for  $i = 1$  to  $\min(c_\tau, \tilde{c}_\tau)$  do
12      if  $x_i^{(\tau)} \in \mathcal{E}_{num}$  then
13         $\hat{x}_i^{(\tau)} \leftarrow \mathcal{M}_L(x_i^{(\tau)}, \epsilon_\tau)$ ;
14      else if  $x_i^{(\tau)} \in \mathcal{E}_{text}$  then
15         $\hat{x}_i^{(\tau)} \leftarrow \mathcal{M}_E(x_i^{(\tau)}, \epsilon_\tau)$ ;
16    Construct sanitized text  $\hat{x}$  using
    perturbed entities;
    // Feature Extraction (Strategy
    I or II)
17     $v_e \leftarrow \phi(\hat{x})$ ;
18     $V_x \leftarrow V_x \cup \{v_{\epsilon_{total}}\}$ ;
19   $\hat{y} \leftarrow g(V_x)$ ;
20 return  $\hat{y}$ ;
```

---

where  $\mathcal{T}$  represents the set of all entity types:

$$\mathcal{T} = \{\text{CARD, MONEY, DATE, TIME}_h, \text{TIME}_m\} \cup \{\text{TEXT}\}. \quad (9)$$

**Sanitization Execution.** For numerical entities  $x_i \in \mathcal{E}_{num}$ , we apply the Laplace Mechanism (Lemma 3.1). The perturbed value  $\hat{x}^{(\tau)}$  is generated by adding noise scaled to  $\epsilon_\tau$ :

$$\hat{x}_i^{(\tau)} \leftarrow \mathcal{M}_L(x_i^{(\tau)}, \epsilon_\tau). \quad (10)$$

For textual entities  $x_i \in \mathcal{E}_{text}$ , we utilize the Exponential Mechanism (Lemma 3.2) to select a replacement token  $y$  from a candidate set  $\mathcal{Y}$  for each  $x_i$ .

Crucially, to strictly adhere to the budget, if the actual occurrence  $c_\tau > \tilde{c}_\tau$ , we only perturb the first  $\lfloor \tilde{c}_\tau \rfloor$  instances and truncate the rest. This granular allocation strategy satisfies the global  $\epsilon_{total}$ -DP constraint.

**Proof.** For each entity type  $\tau$ , we process at most  $\tilde{c}_\tau$  instances. By Theorem 3.3, the budget consumed in the second stage  $\epsilon_{used}$  is bounded by:

$$\begin{aligned} \epsilon_{used} &\leq \sum_{\tau \in \mathcal{T}} \tilde{c}_\tau \cdot \epsilon_\tau \\ &= \sum_{\tau \in \mathcal{T}} \tilde{c}_\tau \cdot \left( \epsilon_{sub} \cdot \frac{\rho_\tau}{\sum_{t \in \mathcal{T}} \rho_t} \right) \quad (11) \\ &= \frac{\epsilon_{sub}}{\sum_{t \in \mathcal{T}} \rho_t} \cdot \sum_{\tau \in \mathcal{T}} \rho_\tau = \epsilon_{sub}. \end{aligned}$$

The total privacy cost is  $\epsilon_{cnt} + \epsilon_{used} \leq \epsilon_{cnt} + \epsilon_{sub} = \epsilon_{total}$ . Thus, the scheme satisfies  $\epsilon_{total}$ -DP. ■

### 4.3 Downstream Detection Strategies

Following the sanitization process, the perturbed text  $\hat{x}$  retains its semantic structure while masking sensitive attributes. The subsequent challenge is to extract robust features  $\phi(\hat{x})$  that enable the classifier  $g$  to distinguish machine-generated text from human-written text. We formulate this as a sequence classification problem over a fixed window size  $K$  (where  $K = 30$ ). We propose two distinct feature extraction paradigms: *Metric-based* and *Model-based*.

#### Strategy I: Metric-based Feature Extraction.

This strategy relies on the hypothesis that machine-generated text exhibits specific statistical artifacts (e.g., lower perplexity, higher consistency). We define a statistical mapping function  $\phi_{stat} : \mathcal{X} \rightarrow \mathbb{R}^{3 \times K}$ . For each token  $\hat{x}_t$  in the sanitized sequence window  $t \in [1, K]$ , we utilize a proxy language model  $M_{proxy}$  to compute a feature vector  $\mathbf{v}_t \in \mathbb{R}^3$ :

$$\mathbf{v}_t = [\text{Conf}(\hat{x}_t), \text{Cohen}(\hat{x}_t), \text{Logit}(\hat{x}_t)]^\top, \quad (12)$$

where:

- $\text{Conf}(\hat{x}_t) = 1 - p\_value(\hat{x}_t)$  represents the model's confidence in the token, derived from the Mann-Whitney U test on the probability distribution.

- $\text{Cohen}(\hat{x}_t)$  measures the effect size (Cohen’s  $d$ ) of the divergence between the proxy model’s top predictions and the observed token.
- $\text{Logit}(\hat{x}_t)$  is the raw unnormalized detection score from  $M_{\text{proxy}}$ .

The resulting feature matrix  $\mathbf{V}_{\text{met}} = [\mathbf{v}_1, \dots, \mathbf{v}_K] \in \mathbb{R}^{3 \times K}$  captures the distributional trajectory of the text.

#### Strategy II: Model-based Feature Extraction.

This strategy exploits high-dimensional semantic representations. We employ a pre-trained Transformer encoder  $\mathcal{E}$  (e.g., DistilBERT or RoBERTa) to map the sanitized text into a dense vector space. Let  $\mathbf{H} \in \mathbb{R}^{L \times d_{\text{model}}}$  be the hidden states output by the encoder for the sequence  $\hat{\mathbf{x}}$ , where  $d_{\text{model}} = 768$ . We extract the representations corresponding to the first  $K$  tokens:

$$\mathbf{V}_{\text{emb}} = \text{Truncate}(\mathcal{E}(\hat{\mathbf{x}}), K) \in \mathbb{R}^{768 \times K}. \quad (13)$$

Unlike Strategy I, which relies on scalar statistics,  $\mathbf{V}_{\text{emb}}$  preserves the contextual embeddings of the sanitized entities and their surrounding context.

**Classification.** Finally, the extracted feature matrix  $\mathbf{V}$  (either  $\mathbf{V}_{\text{met}}$  or  $\mathbf{V}_{\text{emb}}$ ) is flattened and passed to the binary classifier  $g$ :

$$\hat{y} = g(\text{Flatten}(\mathbf{V})), \quad (14)$$

where  $g$  is implemented as a learnable classifier that minimizes the classification error. This separation of  $\phi$  and  $g$  allows us to evaluate how different feature granularities (i.e. statistical vs. semantic) respond to the noise introduced by  $\mathcal{M}$ .

## 5 Experiments

### 5.1 Experimental Setup

**Baselines.** We benchmark our proposed DP-MGTD against two primary categories of detection methods. **Metric-based** approaches operate in a zero-shot setting, deriving statistical features directly from the probability distributions of the source language model. Representative methods in this category include Log-Likelihood (Solaiman et al., 2019), Rank and Rank-GLTR (Gehrmann et al., 2019), Entropy (Gehrmann et al., 2019), LRR (Su et al., 2023), Fast-DetectGPT (Bao et al., 2023), and Binoculars (Hans et al., 2024). **Model-based** approaches involve training supervised classifiers

on labeled datasets. We employ RoBERTa-F and DistilBERT-F (Ippolito et al., 2019) as representative baselines for this category.

**Models and Datasets.** To evaluate the generalization capability of the detectors, we select five LLMs with diverse architectures to serve as text generators: Llama-3.1-70b (Dubey et al., 2024), Mixtral-8x7b (Jiang et al., 2024), GPT-3.5 (Ye et al., 2023), GPT-4o-mini (Menick et al., 2024), and MoonShot-8k (Xu et al., 2024). We construct our evaluation benchmark based on the MGTBench-2.0 dataset (Liu et al., 2025). Within this framework, we employ the aforementioned LLMs to generate synthetic MGT across three primary domains: **STEM** (encompassing Physics, Mathematics, Computer Science, Biology, Chemistry, Electrical Engineering, Medicine, and Statistics), **Humanities** (covering Art, History, Literature, Philosophy, and Law), and **Social Sciences** (spanning Education, Management, and Economy). These MGT samples are paired with human-written content sourced from Wikipedia, arXiv, and Project Gutenberg.

**Implementation Details.** We implement our detection framework using Python 3.12 and PyTorch 2.5.1. For the LLMs (e.g., Llama-3.1-70b) used for generating MGT, we utilize FP16 precision to optimize memory, while detection metrics are computed in FP64 to ensure numerical stability.

### 5.2 Methodological Details

**Entity Extraction and Sanitization.** We utilize the spaCy library (en\_core\_web\_sm) to identify sensitive entities, categorizing them into *Numerical Entities*, and *Textual Entities* to capture the sensitivity of these entities, we apply DP noise. We define a perturbation trajectory by varying the privacy budget  $\epsilon$  across 30 distinct levels, linearly spaced from 0.1 to 2.0. The privacy budget is dynamically allocated based on entity type and frequency (Details Illustrated in Appendix A.1).

**Feature Extraction and Classification.** For each input text, we generate  $d = 30$  sanitized variants corresponding to the  $\epsilon$  levels. We then extract three statistical features for each variant: the raw detection metric (e.g., Log-Likelihood), the statistical confidence derived from the Mann-Whitney U test ( $1 - p$ -value), and the effect size (Cohen’s  $d$ ). This results in a  $30 \times 3$  feature matrix per sample, capturing the sensitivity dynamics of the text. These features are flattened into a 90-dimensional

| Method                      | Setting | Llama-3.1-70b |              |              | Mixtral-8x7b |              |              | GPT-4o-mini  |              |              | GPT-3.5      |              |              | MoonShot-8k  |              |              |
|-----------------------------|---------|---------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
|                             |         | ST.           | Hu.          | So.          | ST.          | Hu.          | So.          | ST.          | Hu.          | So.          | ST.          | Hu.          | So.          | ST.          | Hu.          | So.          |
| <i>Metric-based Methods</i> |         |               |              |              |              |              |              |              |              |              |              |              |              |              |              |              |
| LL                          | Base    | 0.689         | 0.802        | 0.755        | 0.668        | 0.776        | 0.739        | 0.655        | 0.659        | 0.732        | 0.598        | 0.718        | 0.655        | 0.692        | 0.760        | 0.734        |
|                             | Ours    | <b>0.809</b>  | <b>0.880</b> | <b>0.879</b> | <b>0.768</b> | <b>0.842</b> | <b>0.853</b> | <b>0.795</b> | <b>0.803</b> | <b>0.842</b> | <b>0.743</b> | <b>0.830</b> | <b>0.861</b> | <b>0.801</b> | <b>0.877</b> | <b>0.886</b> |
| Rank                        | Base    | 0.538         | 0.696        | 0.654        | 0.489        | 0.690        | 0.609        | 0.643        | 0.574        | 0.669        | 0.421        | 0.592        | 0.589        | 0.618        | 0.687        | 0.632        |
|                             | Ours    | <b>0.797</b>  | <b>0.825</b> | <b>0.818</b> | <b>0.778</b> | <b>0.806</b> | <b>0.795</b> | <b>0.768</b> | <b>0.811</b> | <b>0.805</b> | <b>0.772</b> | <b>0.829</b> | <b>0.817</b> | <b>0.809</b> | <b>0.840</b> | <b>0.830</b> |
| Rank_GLTR                   | Base    | 0.655         | 0.800        | 0.719        | 0.590        | 0.781        | 0.739        | 0.588        | 0.660        | 0.685        | 0.544        | 0.723        | 0.669        | 0.591        | 0.726        | 0.688        |
|                             | Ours    | <b>0.830</b>  | <b>0.898</b> | <b>0.882</b> | <b>0.772</b> | <b>0.847</b> | <b>0.844</b> | <b>0.783</b> | <b>0.798</b> | <b>0.809</b> | <b>0.742</b> | <b>0.849</b> | <b>0.849</b> | <b>0.822</b> | <b>0.866</b> | <b>0.884</b> |
| ENTROPY                     | Base    | 0.668         | 0.730        | 0.682        | 0.655        | 0.706        | 0.712        | 0.629        | 0.679        | 0.674        | 0.609        | 0.669        | 0.596        | 0.672        | 0.693        | 0.638        |
|                             | Ours    | <b>0.810</b>  | <b>0.833</b> | <b>0.833</b> | <b>0.781</b> | <b>0.832</b> | <b>0.851</b> | <b>0.762</b> | <b>0.825</b> | <b>0.832</b> | <b>0.773</b> | <b>0.826</b> | <b>0.829</b> | <b>0.805</b> | <b>0.826</b> | <b>0.840</b> |
| Binoculars                  | Base    | 0.843         | 0.888        | 0.872        | 0.835        | 0.853        | 0.876        | 0.720        | 0.768        | 0.800        | 0.667        | 0.788        | 0.756        | 0.920        | 0.854        | 0.904        |
|                             | Ours    | <b>0.897</b>  | <b>0.912</b> | <b>0.894</b> | <b>0.881</b> | <b>0.875</b> | <b>0.908</b> | <b>0.794</b> | <b>0.825</b> | <b>0.846</b> | <b>0.790</b> | <b>0.838</b> | <b>0.842</b> | <b>0.935</b> | <b>0.902</b> | <b>0.921</b> |
| <i>Model-based Methods</i>  |         |               |              |              |              |              |              |              |              |              |              |              |              |              |              |              |
| DistilBert-F                | Base    | 0.622         | 0.584        | 0.533        | 0.612        | 0.559        | 0.558        | 0.588        | 0.548        | 0.470        | 0.625        | 0.573        | 0.581        | 0.546        | 0.555        | 0.418        |
|                             | Ours    | <b>0.997</b>  | <b>0.995</b> | <b>0.973</b> | <b>0.989</b> | <b>0.992</b> | <b>0.993</b> | <b>0.986</b> | <b>0.996</b> | <b>0.993</b> | <b>0.986</b> | <b>0.988</b> | <b>0.973</b> | <b>0.996</b> | <b>0.992</b> | <b>0.991</b> |
| Roberta-F                   | Base    | 0.667         | 0.667        | 0.645        | 0.667        | 0.667        | 0.667        | 0.667        | 0.667        | 0.667        | 0.667        | 0.667        | 0.667        | 0.667        | 0.667        | 0.577        |
|                             | Ours    | <b>0.995</b>  | <b>0.998</b> | <b>0.979</b> | <b>0.992</b> | <b>0.993</b> | <b>0.992</b> | <b>0.988</b> | <b>0.998</b> | <b>0.991</b> | <b>0.989</b> | <b>0.988</b> | <b>0.974</b> | <b>0.994</b> | <b>0.990</b> | <b>0.993</b> |

Table 1: Experimental results for MGT. We report F1 scores across five LLMs and three domains: STEM (ST.), Humanities (Hu.), and Social Sciences (So.). The table compares our proposed framework (Ours) against standard baselines (Base) within both metric-based and model-based categories. **Bold** values indicate the superior performance, while the **colored** values denote the performance gain ( $\Delta$ ) achieved by our method over the baseline.

vector and fed into a time-series classifier. Unless otherwise stated, we employ a 2-layer LSTM classifier (hidden size 64) for the final binary decision (Details Illustrated in [Appendix A.4](#)).

### 5.3 Main Results

The detection performance across five LLMs and three domains is summarized in [Table 1](#). The experimental evidence indicates that our DP-MGTD framework consistently outperforms baseline methods in both metric-based and model-based settings. Beyond the numerical improvements, the results reveal three fundamental properties of our approach:

**Unlocking Latent Separability in Supervised Detection.** The most significant observation lies in the model-based methods (DistilBERT-F and RoBERTa-F). In the baseline setting, these classifiers struggle to distinguish human from machine text, yielding F1 scores between 0.53 and 0.67. This suggests that the static semantic features of advanced LLMs have become nearly indistinguishable from human writing. In contrast, DP-MGTD achieves near-perfect separation with F1 scores consistently exceeding 0.99. This quantum leap

indicates that while the *static surface* of MGT mimics human distribution, its *dynamic behavior* under DP-based sanitization is distinct. The supervised models effectively learn to classify the stability patterns exposed by our perturbation, transforming a difficult text classification task into a highly separable feature recognition task.

**Amplification of Weak Distributional Signals.** Our framework demonstrates a restorative effect on weaker zero-shot metrics. Methods relying on raw probability rankings, such as Rank and Rank\_GLTR, exhibit poor baseline performance (e.g., 0.421 for GPT-3.5 in STEM), often failing to outperform random guessing. However, applying DP-MGTD results in substantial performance gains, with improvements exceeding 35% in absolute F1 score. This finding implies that raw likelihood rankings are often poorly calibrated for detection due to RLHF alignment. By measuring the relative change in these rankings after sanitization, our method recovers a robust detection signal. Even for strong baselines like Binoculars, the consistent positive performance gain ( $\Delta$ ) confirms that our sensitivity-based features provide orthogonal

| Method              | $d = 10$     |              |              | $d = 20$     |              |              | $d = 30$     |              |              | $d = 60$     |              |              |
|---------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
|                     | ST.          | Hu.          | So.          | ST.          | Hu.          | So.          | ST.          | Hu.          | So.          | ST.          | Hu.          | So.          |
| <i>Metric-based</i> |              |              |              |              |              |              |              |              |              |              |              |              |
| LL                  | 0.742        | 0.822        | 0.860        | 0.741        | 0.829        | 0.861        | 0.743        | 0.830        | 0.861        | 0.738        | 0.823        | 0.855        |
| Rank                | 0.765        | 0.820        | 0.810        | 0.769        | 0.822        | 0.815        | 0.772        | 0.829        | 0.817        | 0.762        | 0.814        | 0.812        |
| Rank_GLTR           | 0.733        | <b>0.840</b> | <b>0.842</b> | 0.742        | <b>0.844</b> | <b>0.845</b> | 0.742        | <b>0.849</b> | <b>0.849</b> | 0.741        | <b>0.843</b> | <b>0.845</b> |
| ENTROPY             | 0.763        | 0.819        | 0.821        | 0.768        | 0.820        | 0.824        | 0.773        | 0.826        | 0.829        | 0.765        | 0.818        | 0.822        |
| Log-Rank            | 0.742        | 0.781        | 0.792        | 0.744        | 0.786        | 0.799        | 0.750        | 0.796        | 0.804        | 0.743        | 0.783        | 0.787        |
| Binoculars          | <b>0.781</b> | 0.831        | 0.835        | <b>0.784</b> | 0.835        | 0.837        | <b>0.790</b> | 0.838        | 0.842        | <b>0.784</b> | 0.833        | 0.836        |
| <i>Model-based</i>  |              |              |              |              |              |              |              |              |              |              |              |              |
| DistilBert-F        | 0.980        | 0.979        | 0.965        | 0.983        | 0.983        | 0.970        | 0.986        | <b>0.988</b> | 0.973        | 0.981        | 0.980        | 0.968        |
| Roberta-F           | <b>0.983</b> | <b>0.982</b> | <b>0.967</b> | <b>0.986</b> | <b>0.984</b> | <b>0.970</b> | <b>0.989</b> | <b>0.988</b> | <b>0.974</b> | <b>0.984</b> | <b>0.983</b> | <b>0.971</b> |

Table 2: Impact of the perturbation dimension  $d$  on detection performance (F1 score) across different domains. Experiments are conducted under the GPT-3.5 setting.  $d$  represents the number of  $\epsilon$ -perturbations applied per sample. **Bold** indicates the best performance in each category of detection methods.

information to standard likelihood metrics.

**Robustness Across Model Architectures and Domains.** The efficacy of DP-MGTD remains invariant across diverse LLM architectures and subject domains. We observe consistent improvements whether the source is a dense model (Llama-3.1-70b), a mixture-of-experts model (Mixtral-8x7b), or a proprietary black-box model (GPT-4o-mini). This universality suggests that the vulnerability to DP-based perturbation is not an artifact of specific training data or model size but rather an intrinsic property of current autoregressive generation mechanisms. Furthermore, the high performance across STEM, Humanities, and Social Sciences indicates that our method does not rely on domain-specific keywords but captures fundamental structural differences between human and machine composition.

### 5.4 Sensitivity Analysis of Perturbation Granularity

We investigate the impact of the perturbation dimension  $d$  (defined as the resolution of the  $\epsilon$  grid used for sanitization) on the discriminative power of our framework. Table 2 details the detection performance under the GPT-3.5 setting as  $d$  varies from 10 to 60. The results reveal two critical insights regarding the nature of the extracted signals:

**Robustness to Granularity.** A key finding is the method’s stability across varying resolutions. Even at a coarse granularity of  $d = 10$ , the model-based detectors (DistilBERT-F and RoBERTa-F) achieve F1 scores exceeding 0.96 across all domains. This high baseline performance suggests that the sensitivity fingerprint of MGT is a *strong signal* feature. Unlike subtle statistical artifacts that require high-dimensional feature engineering to uncover, the DP-based perturbations expose fundamental structural

vulnerabilities in machine text that are detectable even with a sparse sampling of  $\epsilon$ . This implies that DP-MGTD is inherently robust to hyperparameter selection, reducing the need for extensive fine-tuning in practical deployments.

**Signal Saturation and Optimal Resolution.** While performance improves as  $d$  increases to 30, capturing finer nuances of the perturbation response, we observe a plateau or slight regression at  $d = 60$ . For instance, in the Humanities domain, Rank\_GLTR peaks at 0.849 ( $d = 30$ ) before dropping to 0.843 ( $d = 60$ ). This phenomenon indicates *signal saturation*, where the core discriminative information is fully captured within the first 30 dimensions. Extending the feature space beyond this point introduces redundancy without providing orthogonal information, potentially leading to the curse of dimensionality for the classifier.

## 6 Conclusion

In this work, we presented DP-MGTD, a privacy-preserving framework that effectively reconciles the tension between data confidentiality and authorship verification. By implementing an adaptive differentially private entity sanitization mechanism, our approach secures sensitive information while preserving essential linguistic dependencies. Crucially, our empirical findings reveal that the sensitivity of text to DP-based perturbations serves as a robust discriminative feature, transforming the privacy constraint into an enhancement for detection accuracy. Extensive evaluations on MGTBench-2.0 demonstrate that our method significantly outperforms existing baselines across diverse domains and architectures. This study establishes a new paradigm for secure MGT detection, suggesting that privacy mechanisms can uncover latent distri-

butional distinctions between human and machine intelligence. Future work will explore the theoretical bounds of this stability phenomenon and extend the framework to broader generative tasks.

## Limitations

The primary limitation of this study lies in the theoretical formalization of the observed counter-intuitive phenomenon where differential privacy noise enhances detection accuracy. While our empirical results consistently demonstrate this effect, we have not yet established a comprehensive mathematical proof to fully explain the interaction between specific noise distributions and the decision boundaries of MGT detectors. Future work is required to isolate the causal factors driving this performance gain. Additionally, our comparative analysis is currently constrained to standard masking strategies and basic perturbation baselines. A more exhaustive evaluation involving a wider array of state-of-the-art privacy-preserving text generation methods and diverse detector architectures would strengthen the generalizability of our claims. Furthermore, the proposed framework relies heavily on the presence of named entities. Consequently, its efficacy in scenarios involving highly abstract reasoning or texts with sparse entity occurrences remains to be fully verified, as the privacy budget allocation mechanism depends on entity density. Finally, while we validated our approach on the MGTBench-2.0 dataset, the robustness of the method across low-resource languages or highly specialized domains such as medical or legal texts warrants further investigation.

## Ethics Statement

This research adheres to the ethical guidelines regarding data privacy and responsible AI development. The experiments conducted in this study utilize the MGTBench-2.0 dataset, which comprises public domain texts and content generated by large language models. We explicitly state that no private user data or real-world personally identifiable information was collected, stored, or processed during the training and evaluation phases. The proposed entity sanitization framework is intended to protect user privacy in downstream applications; however, we acknowledge the potential risk that similar obfuscation techniques could be repurposed to evade detection systems for malicious intent. To mitigate this, we advocate for the development of

detectors that are robust to such perturbations, as demonstrated by our findings. Furthermore, we have considered the computational impact of our method. The adaptive privacy budget allocation is designed to be computationally efficient, ensuring that the integration of privacy guarantees does not incur a prohibitive carbon footprint compared to standard model inference. We also recognize that the underlying language models used for entity replacement may carry inherent biases, and future deployment of this framework should include rigorous fairness audits to prevent the propagation of such biases in sanitized outputs.

## GenAI Usage Statement

We clarify the use of generative AI in this study as follows: (1) Research Methodology: As detailed in Sections 5.1, LLMs were utilized for MGT data generation. (2) Writing and Coding: LLMs were used as productivity tools to optimize the codebase and refine the writing expression of the paper. All final outputs were critically reviewed by the authors.

## References

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*.
- Ahmed Musa Awon, Yun Lu, Shera Potka, and Alex Thomo. 2025. Clusant: Differentially private and semantically coherent text sanitization. In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 3676–3693.
- Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2023. Fast-detectgpt: Efficient zero-shot detection of machine-generated text via conditional probability curvature. *arXiv preprint arXiv:2310.05130*.
- Ricardo Silva Carvalho, Theodore Vasiloudis, Oluwaseyi Feyisetan, and Ke Wang. 2023. Tem: High utility metric differential privacy on text. In *Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)*, pages 883–890. SIAM.
- Chaoran Chen, Daodao Zhou, Yanfang Ye, Toby Jia-jun Li, and Yaxing Yao. 2025. Clear: Towards contextual llm-empowered privacy policy analysis and risk generation for large language model applications. In *Proceedings of the 30th International Conference on Intelligent User Interfaces*, pages 277–297.

- Sai Chen, Fengran Mo, Yanhao Wang, Cen Chen, Jian-Yun Nie, Chengyu Wang, and Jamie Cui. 2023. A customized text sanitization mechanism with differential privacy. In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 5747–5758.
- Amrita Roy Chowdhury, David Glukhov, Divyam Anshumaan, Prasad Chalasani, Nicolas Papernot, Somesh Jha, and Mihir Bellare. 2025. Pr  $\epsilon\epsilon$  mpt: Sanitizing sensitive prompts for llms. *arXiv preprint arXiv:2504.05147*.
- Badhan Chandra Das, M Hadi Amini, and Yanzhao Wu. 2025. Security and privacy challenges of large language models: A survey. *ACM Computing Surveys*, 57(6):1–39.
- Rian Dolphin, Joe Dursun, Jonathan Chow, Jarrett Blankenship, Katie Adams, and Quinton Pike. 2024. Extracting structured insights from financial news: An augmented llm driven approach. *arXiv preprint arXiv:2407.15788*.
- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and 1 others. 2024. The llama 3 herd of models. *arXiv e-prints*, pages arXiv–2407.
- Cynthia Dwork. 2006. Differential privacy. In *International colloquium on automata, languages, and programming*, pages 1–12. Springer.
- Sebastian Gehrmann, Hendrik Strobelt, and Alexander M Rush. 2019. Glt: Statistical detection and visualization of generated text. *arXiv preprint arXiv:1906.04043*.
- Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024. Spotting llms with binoculars: Zero-shot detection of machine-generated text. *arXiv preprint arXiv:2401.12070*.
- Xinlei He, Xinyue Shen, Zeyuan Chen, Michael Backes, and Yang Zhang. 2024. Mtgbench: Benchmarking machine-generated text detection. In *Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security*, pages 2251–2265.
- Daphne Ippolito, Daniel Duckworth, Chris Callison-Burch, and Douglas Eck. 2019. Automatic detection of generated text is easiest when humans are fooled. *arXiv preprint arXiv:1911.00650*.
- Daphne Ippolito, Daniel Duckworth, Chris Callison-Burch, and Douglas Eck. 2020. Automatic detection of generated text is easiest when humans are fooled. In *Proceedings of the 58th annual meeting of the association for computational linguistics*, pages 1808–1822.
- Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, and 1 others. 2024. Mixtral of experts. *arXiv preprint arXiv:2401.04088*.
- Gleb Kuznetsov, Pavel Blinov, Yulia Kuzkina, Vasily Goncharov, Galina Zubkova, Nikolai Zenovkin, Aleksei Goncharov, and Andrey Savchenko. 2024. Medsyn: Llm-based synthetic medical text generation framework. In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*, pages 215–230. Springer.
- Yafu Li, Qintong Li, Leyang Cui, Wei Bi, Zhilin Wang, Longyue Wang, Linyi Yang, Shuming Shi, and Yue Zhang. 2024. Mage: Machine-generated text detection in the wild. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 36–53.
- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*.
- Yule Liu, Zhiyuan Zhong, Yifan Liao, Zhen Sun, Jingyi Zheng, Jiaheng Wei, Qingyuan Gong, Fenghua Tong, Yang Chen, Yang Zhang, and 1 others. 2025. On the generalization and adaptation ability of machine-generated text detectors in academic writing. In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2*, pages 5674–5685.
- Abdul Majeed and Sungchang Lee. 2020. Anonymization techniques for privacy preserving data publishing: A comprehensive survey. *IEEE access*, 9:8512–8545.
- Jacob Menick, Kevin Lu, Shengjia Zhao, E Wallace, H Ren, H Hu, N Stathas, and F Petroski Such. 2024. Gpt-4o mini: advancing cost-efficient intelligence. *Open AI: San Francisco, CA, USA*.
- Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. 2023. Detectgpt: Zero-shot machine-generated text detection using probability curvature. In *International conference on machine learning*, pages 24950–24962. PMLR.
- Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.
- Irene Solaiman, Miles Brundage, Jack Clark, Amanda Askell, Ariel Herber-Voss, Jeff Wu, Alec Radford, Gretchen Krueger, Jong Wook Kim, Sarah Kreps, and 1 others. 2019. Release strategies and the social impacts of language models. *arXiv preprint arXiv:1908.09203*.

Jinyan Su, Terry Yue Zhuo, Di Wang, and Preslav Nakov. 2023. Detectllm: Leveraging log rank information for zero-shot detection of machine-generated text. *arXiv preprint arXiv:2306.05540*.

Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu, Tianwei Zhang, and Zhikun Zhang. 2025. Infedpt: Privacy-preserving inference for black-box large language models. *IEEE Transactions on Dependable and Secure Computing*.

Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Lidia Sam Chao, and Derek Fai Wong. 2025. A survey on llm-generated text detection: Necessity, methods, and future directions. *Computational Linguistics*, 51(1):275–338.

Junchao Wu, Runzhe Zhan, Derek Wong, Shu Yang, Xinyi Yang, Yulin Yuan, and Lidia Chao. 2024. Detectrl: Benchmarking llm-generated text detection in real-world scenarios. *Advances in Neural Information Processing Systems*, 37:100369–100401.

Liuchang Xu, Shuo Zhao, Qingming Lin, Luyao Chen, Qianqian Luo, Sensen Wu, Xinyue Ye, Hailin Feng, and Zhenhong Du. 2024. Evaluating large language models on spatial tasks: A multi-task benchmarking study. *arXiv preprint arXiv:2408.14438*.

Junjie Ye, Xuantang Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhuan Cui, Zeyang Zhou, Chao Gong, Yang Shen, and 1 others. 2023. A comprehensive capability analysis of gpt-3 and gpt-3.5 series models. *arXiv preprint arXiv:2303.10420*.

Xiang Yue, Minxin Du, Tianhao Wang, Yaliang Li, Huan Sun, and Sherman SM Chow. 2021. Differential privacy for text analytics via natural text sanitization. *arXiv preprint arXiv:2106.01221*.

# A Detailed Experimental Settings

### A.1 Entity Extraction and Sensitivity

We use the en\_core\_web\_sm model from spaCy for Named Entity Recognition (NER). Entities are classified into two categories with distinct sensitivity configurations:

**Numerical Entities.** We identify types such as CARDINAL, MONEY, DATE, and TIME. The sensitivity for these entities corresponds to their valid numerical range. For instance, DATE (day of month) has a sensitivity of 29, while TIME\_minutes has a sensitivity of 59. For unbounded numbers like CARDINAL and MONEY, we assign a capped sensitivity of 10,000 to prevent excessive noise.

**Textual Entities.** We identify types including PERSON, GPE, ORG, PRODUCT, EVENT, WORK\_OF\_ART, FAC, and LAW. Sensitivity is defined by the size of the candidate replacement pool. For example, the

sensitivity for PERSON is determined by the size of our compiled list of common names.

**Budget Allocation.** The total privacy budget  $\epsilon_{\text{total}}$  is distributed among entities using a weighted scheme. The weight  $w_i$  for an entity type  $i$  is calculated as:

$$w_i = w_{\text{base}} \times \log(\text{sensitivity}_i + 1) \times (\text{count}_i + 1) \quad (15)$$

where  $w_{\text{base}}$  is an empirical constant (e.g., 0.3 for numbers, 0.25 for persons). This ensures that more sensitive and frequent entities receive a larger share of the privacy budget, resulting in lower noise levels for critical information.

### A.2 Perturbation Mechanisms

**Laplace Mechanism (Numerical).** For a numerical value  $x$ , we add noise sampled from a Laplace distribution:

$$x' = x + \text{Laplace}\left(0, \frac{\Delta f}{\epsilon_i}\right) \quad (16)$$

where  $\Delta f$  is the sensitivity and  $\epsilon_i$  is the allocated budget. We apply post-processing to ensure validity (e.g., ensuring time minutes remain in  $[0, 59]$ ).

**Exponential Mechanism (Textual).** For a textual entity  $t$ , we select a replacement  $t'$  from a candidate set  $\mathcal{C}$  with probability proportional to the privacy score:

$$P(t'|t) = \frac{\exp\left(\frac{\epsilon_i \cdot u(t, t')}{2\Delta u}\right)}{\sum_{z \in \mathcal{C}} \exp\left(\frac{\epsilon_i \cdot u(t, z)}{2\Delta u}\right)} \quad (17)$$

where the utility function  $u$  is binary (1 if  $t' = t$ , 0 otherwise). This simplifies to a probability  $P_{\text{keep}}$  of retaining the original word and  $1 - P_{\text{keep}}$  of sampling uniformly from  $\mathcal{C} \setminus \{t\}$ .

### A.3 Data Preprocessing and Filtering

To ensure data quality, we filter the MGTBench-2.0 dataset using the following criteria:

- **Length constraints:** Text length must be between 100 and 15,000 characters.
- **Entity requirements:** Samples must contain at least one numerical entity and one textual entity.
- **Density check:** The entity density (number of entities divided by text length) must exceed a threshold of 0.003.

Outliers in the feature space are removed using the Interquartile Range (IQR) method before training the classifiers.

### A.4 Classifier Hyperparameters

The time-series classifier used in our main experiments is a Long Short-Term Memory (LSTM) network. The specific configuration is as follows:

- **Architecture:** 2-layer LSTM.
- **Hidden Size:** 64.
- **Dropout:** 0.2.
- **Input Dimension:** 90 (30 time steps  $\times$  3 features).
- **Training/Test Split:** 80% training, 20% testing, stratified by class.
- **Feature Scaling:** Standard normalization (zero mean, unit variance) applied per feature.

We also explored other classifiers (e.g., SVM with RBF kernel, Random Forest) and found the LSTM to offer the most robust performance for the sequential sensitivity features.