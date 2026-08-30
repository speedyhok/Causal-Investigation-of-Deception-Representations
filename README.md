# A Causal Investigation of Deception Representations in Gemma-2-2B

### Probing, Sparse Autoencoders, Residual-State Intervention, and Cross-Question Transfer

This repository contains the code and experimental materials for a mechanistic interpretability study investigating whether **instructed deceptive behavior in a language model can be detected from internal activations and causally manipulated**.

The study uses **Google Gemma-2-2B-IT** and a controlled factual question benchmark in which the model is asked either to answer truthfully or to intentionally provide an incorrect answer.

The central question is:

> **Can deceptive behavior in an LLM be identified from its internal representations, and can those representations be causally manipulated to change the model's deceptive behavior?**

The experiments reveal a distinction between **decodability** and **causality**: truthful and deceptive states are highly distinguishable from internal activations, but the features that are most predictive of deception are not necessarily the features that causally control the behavior.

---

## Research Overview

The project investigates deception in four stages:

1. **Residual-stream probing**
   Determine whether truthful and deceptive internal states can be distinguished using classifiers.

2. **Sparse Autoencoder analysis**
   Determine whether deception-related information is concentrated in a small number of interpretable SAE features.

3. **Dense residual-stream intervention**
   Test whether moving a deceptive residual state toward its corresponding truthful state actually changes the generated answer.

4. **Cross-question transfer**
   Test whether the causal direction discovered for one question transfers to unrelated questions, which would provide evidence for a more general "truthfulness direction."

The overall hypothesis was that if deception corresponds to a simple internal representation, it should be both **detectable** and potentially **transferable** across questions.

The experiments instead support a more nuanced picture:

> **Deception is highly decodable, but the strongest causal effect comes from dense, question-specific residual differences rather than from a universal "lying direction" or a small set of individual SAE features.**

---

# Main Findings

## 1. Truthful and deceptive states are highly decodable

The experiment begins with **200 manually constructed factual question-answer pairs**.

After automatic filtering and manual exclusions:

* Initial examples: **200**
* After automatic cleaning: **147**
* Final dataset: **145 questions**

For each question, the model produces:

* one truthful response
* one intentionally deceptive response

Residual-stream activations are extracted from layers:

```text
5, 10, 15, 20, 25
```

Each selected final-token hidden state has dimensionality:

```text
2304
```

Logistic regression is then trained to distinguish truthful from deceptive activations.

The executed notebook reports:

| Layer | Accuracy | ROC-AUC |
| ----- | -------: | ------: |
| 5     |     1.00 |    1.00 |
| 10    |     1.00 |    1.00 |
| 15    |     1.00 |    1.00 |
| 20    |     1.00 |    1.00 |
| 25    |     1.00 |    1.00 |

Additional checks using both a linear SVM and an RBF SVM at layer 20 also report:

```text
Accuracy = 1.00
ROC-AUC  = 1.00
```

Five-fold logistic-regression cross-validation at layer 20 gives:

```text
ROC-AUC = [1.0, 1.0, 1.0, 1.0, 1.0]

Mean = 1.00
Std   = 0.00
```

The same procedure gives a mean ROC-AUC of 1.00 across all tested layers.

### Interpretation

These results show that the internal residual states contain a very strong signal distinguishing the two conditions.

However:

> **Perfect probe performance does not prove that the probe has discovered a semantic "deception representation."**

The probe split is **sample-level rather than question-level**. Because each question contributes both a truthful and deceptive activation, the same question can occur in different folds. Therefore, the experiment establishes very strong within-benchmark decodability, but not generalization to completely unseen questions.

---

# 2. Sparse Autoencoder Analysis

To investigate whether deception-related information is localized in sparse features, the project uses the following pretrained Gemma Scope SAE:

```text
release:
gemma-scope-2b-pt-res

SAE:
layer_20/width_16k/average_l0_71
```

The SAE contains:

```text
16,384 latent features
```

The layer-20 residual representation is encoded into this sparse latent space.

For every feature \(j\), the experiment calculates the mean activation difference between deceptive and truthful conditions:

$$
\Delta_j =
E[z_j^{lie}] - E[z_j^{truth}]
$$

The ten highest-ranked features are:

```text
9120
259
8366
1642
15027
6698
15848
7248
247
1815
```

Some features show extremely large descriptive differences.

For example, feature `9120` has:

```text
Truth mean activation: 0.3483
Lie mean activation:   22.7805
```

and is active in approximately:

```text
Truth: 2.76%
Lie:   81.38%
```

The ten selected SAE features are highly predictive of the truth/lie condition.

Five-fold ROC-AUC:

```text
[1.0, 1.0, 1.0, 0.9952, 0.9988]

Mean ROC-AUC = 0.9988
```

### But prediction was not enough

The strongest SAE features were then directly manipulated through several interventions:

* latent feature amplification
* decoder-direction steering
* individual feature ablation

Despite strong predictive correlations, these interventions did **not reliably change the deceptive answer**.

For the France example, ablation of the strongest candidate feature still resulted in the deceptive answer:

```text
London
```

and the same general failure was observed when the individual ablation procedure was applied to the ten highest-ranked candidate features.

### Key conclusion

This creates an important **probe–causality gap**:

> **A feature can strongly predict deception without being sufficient to causally control deceptive behavior.**

This suggests that the relevant computation is more distributed than a small collection of individually sufficient SAE features.

---

# 3. Dense Residual-Stream Causal Intervention

The strongest causal result comes from directly manipulating the dense residual stream.

For each question \(i\), the study computes a question-specific difference:

$$
d_i =
h_i^{truth} - h_i^{lie}
$$

where:

* \(h_i^{truth}\) is the truthful residual state
* \(h_i^{lie}\) is the deceptive residual state

The intervention then modifies the deceptive state using:

$$
h_i' =
h_i^{lie} + \alpha d_i
$$

where \(\alpha\) controls intervention strength.

---

## Layer localization

For the France example, the norm of the truthful-minus-deceptive residual difference increases across later layers:

| Layer | Difference Norm |
| ----- | --------------: |
| 10    |         59.1875 |
| 15    |          145.25 |
| 20    |           266.0 |
| 25    |           669.0 |

This is a measure of separation between the two states; it does **not** mean that later layers are inherently "more truthful."

---

## Activation patching

A direct patch replaces the deceptive residual state with the corresponding truthful state.

For the France question:

| Layer | Generated output                 |
| ----- | -------------------------------- |
| 5     | `LondonLondonLondonLondonLondon` |
| 10    | `LondonLondonLondonLondonLondon` |
| 15    | `ParisParisParisParisParis`      |
| 20    | `ParisParisParisParisParis`      |
| 25    | `ParisParisParisParisParis`      |

This provides direct evidence that changing the residual state at the appropriate stage of the network can change the generated answer.

The effect is layer-dependent:

```text
Layers 5, 10  → deceptive answer remains
Layers 15–25  → truthful answer recovered
```

---

# 4. Direction-Based Steering

Instead of fully replacing the hidden state, the experiment gradually moves the deceptive state toward the truthful state.

At layer 20:

|   α | Output                      |
| --: | --------------------------- |
| 0.1 | London                      |
| 0.5 | Paris is the capital of     |
| 1.0 | Paris is the capital of     |
| 2.0 | Paris / repeated Paris      |
| 5.0 | Malformed / unstable output |

The important observation is that:

```text
α = 0.5
```

is sufficient to change the deceptive answer into the truthful answer without completely replacing the original hidden state.

However, stronger interventions eventually produce unstable or malformed generations.

Therefore:

> **Increasing intervention strength is not equivalent to increasing "truthfulness."**

---

# 5. Large-Scale Question-Specific Intervention

The final causal experiment applies the same idea across all **145 questions**.

For every question, a separate truth–lie direction is computed:

$$
d_i =
h_i^{truth} - h_i^{lie}
$$

with:

```text
Layer = 20
α     = 0.5
```

The baseline deceptive generation achieves:

```text
Truth recovery: 0 / 145
```

After question-specific residual intervention:

```text
Truth recovery: 118 / 145
```

or:

```text
81.38%
```

This represents an improvement of:

```text
81.38 percentage points
```

Some representative successful examples include:

| Question          | Truth     | Steered output           |
| ----------------- | --------- | ------------------------ |
| France capital    | Paris     | Paris is the capital of  |
| Germany capital   | Berlin    | Berlin is the capital of |
| Italy capital     | Rome      | Rome is the capital of   |
| Canada capital    | Ottawa    | Ottawa is the capital of |
| Australia capital | Canberra  | Canberra is the capital  |
| India capital     | New Delhi | New Delhi is the capital |
| Japan capital     | Tokyo     | Tokyo is the capital of  |
| Egypt capital     | Cairo     | Cairo is the capital of  |
| Spain capital     | Madrid    | Madrid is the capital of |

The method is not perfect. For example, for Brazil:

```text
Truth: Brasília
Steered output: Rio de Janeiro
```

This demonstrates that causal intervention can substantially improve truth recovery without guaranteeing correctness on every example.

---

# 6. Is There a Universal "Truth Direction"?

A natural interpretation of the causal result would be:

> Perhaps the model contains a universal direction corresponding to "truthfulness."

To test this, the study performs **cross-question transfer**.

Instead of calculating:

$$
d_i =
h_i^{truth} - h_i^{lie}
$$

for the target question, a direction obtained from one question is applied to another unrelated question.

For example:

```text
Source direction: Paris
Target question: Germany
```

If a universal truthfulness direction existed, transferring the source direction should help recover:

```text
Berlin
```

Instead, the results are inconsistent.

Outputs can include:

```text
The capital of Germany is,
London,
London, England.
```

but the correct answer:

```text
Berlin
```

is not reliably recovered.

Increasing the intervention strength also does not reveal a stable universal effect. At very large strengths, outputs can become malformed or disruptive.

### Interpretation

The experiments therefore provide:

```text
No reliable evidence for a universal truth direction.
```

The successful causal vector is better described as:

$$
\text{question-specific truthful state}
-
\text{question-specific deceptive state}
$$

rather than:

$$
\text{truthfulness}
-
\text{deception}
$$

For example, the France vector may contain information associated with changing:

```text
Paris ←→ London
```

rather than representing a pure abstract "truth" concept.

---

# Overall Results

| Experiment                          | Result                           |
| ----------------------------------- | -------------------------------- |
| Residual logistic probe             | 1.00 accuracy / 1.00 ROC-AUC     |
| Layer-20 five-fold probe            | Mean ROC-AUC = 1.00              |
| Linear SVM                          | 1.00 accuracy / 1.00 ROC-AUC     |
| RBF SVM                             | 1.00 accuracy / 1.00 ROC-AUC     |
| Top-10 SAE features                 | Mean ROC-AUC = 0.9988            |
| Individual SAE interventions        | No reliable answer change        |
| Residual activation patching        | Later layers switch answer       |
| Question-specific residual steering | 118/145 = 81.38% recovery        |
| Baseline truth recovery             | 0/145                            |
| Cross-question transfer             | No reliable target-fact recovery |

---

# What This Study Shows

The experiments support four main conclusions.

### 1. Deceptive states are highly decodable

The model's internal residual states contain a very strong signal distinguishing truthful and intentionally deceptive conditions.

### 2. Decodability does not imply causality

SAE features can be extremely predictive while failing to control the behavior when directly manipulated.

### 3. Dense residual intervention has causal leverage

Moving a deceptive residual state toward the corresponding truthful state can substantially change the model's generation.

### 4. There is no evidence for a universal truth direction

The successful intervention appears to be **question-conditioned and distributed**, rather than a single global axis representing truthfulness or lying.

---

# A Critical Interpretation

This repository does **not** claim to have built a general-purpose AI lie detector.

The experiment uses a deliberately controlled definition of deception:

> The model is explicitly instructed to provide an incorrect answer to a factual question.

This is different from:

* autonomous strategic deception
* long-horizon scheming
* manipulation of oversight
* deceptive planning
* misleading but technically true statements
* real-world agentic deception

The strongest causal intervention also relies on a **teacher-conditioned counterfactual**: for each question, the truthful activation from that same question is used to construct the intervention vector.

A real monitoring system would generally not have access to this truthful counterfactual.

Therefore, the study should be understood primarily as a **mechanistic proof-of-concept**, not as a deployable lie-correction system.

---

# Limitations

The main limitations are:

* **Controlled deception definition** — only instructed false answering is studied.
* **Sample-level probe split** — the current probe evaluation does not establish clean generalization to unseen questions.
* **Small manually constructed benchmark** — only 145 usable questions remain after filtering.
* **Simple factual tasks** — many questions belong to related semantic families, which may allow shortcuts.
* **Substring-based truth recovery** — the evaluation can miss semantic paraphrases or count awkward matches.
* **Teacher-conditioned causal directions** — the intervention uses the model's own truthful counterfactual for the same question.
* **Single model and SAE** — the study uses Gemma-2-2B-IT and one Gemma Scope residual-stream SAE.
* **Intervention instability** — large intervention strengths can produce malformed outputs.

These limitations are important when interpreting the reported 81.38% recovery result.

---

# Future Work

Several experiments would strengthen the conclusions.

### Question-held-out probing

Use `GroupKFold` or another grouped split so that all truth/lie states associated with a question remain in the same fold.

### Instruction generalization

Test multiple paraphrases of the truth and lie instructions to determine whether the detector is learning deception-related information rather than prompt-specific patterns.

### Geometric analysis

Collect the 145 question-specific vectors and analyze their geometry using:

* cosine similarity
* PCA
* low-rank decomposition

This could reveal whether the directions occupy a lower-dimensional subspace.

### Sparse causal subspaces

Instead of intervening on one SAE feature at a time, jointly intervene on groups of features or on probe-derived directions.

### Cross-model replication

Repeat the experiments across:

* larger models
* different model families
* different SAE releases
* different activation layers

### Broader deception benchmarks

Extend the benchmark beyond factual false answers to include:

* omission
* misleading framing
* strategic deception
* multi-step deceptive behavior

---

# Reproducibility

The complete experimental implementation is provided in the accompanying Google Colab notebook.

**Google Colab Notebook:**

https://colab.research.google.com/drive/1cV1_EzPlelZdOd8tdSgMU8XU6H3JtMak?usp=sharing

The notebook contains the implementation corresponding to the thesis, including:

* model loading
* prompt construction
* dataset generation
* filtering
* activation extraction
* probing
* SVM evaluation
* cross-validation
* SAE loading and encoding
* feature ranking
* SAE interventions
* residual activation patching
* residual-state steering
* large-scale intervention
* cross-question transfer experiments

---

# Computational Environment

The project was designed to be lightweight enough for experimentation in **Google Colab with a single NVIDIA T4-class GPU**.

Main components include:

```text
Google Gemma-2-2B-IT
Hugging Face Transformers
NNsight
SAE Lens
Gemma Scope SAE
scikit-learn
PyTorch
```

The model is loaded in half precision, and residual activations are traced and modified during generation.

---

# Repository Structure

A recommended repository structure is:

```text
.
├── README.md
├── Causal_Investigation_of_Deception.pdf
├── Causal_Investigation_Deception_Thesis.tex
├── notebook/
│   └── Causal_Investigation_of_Deception_Representations.ipynb
├── results/
│   └── ...
└── figures/
    └── ...
```

The exact file organization may differ depending on how the final repository is packaged.

---

# Thesis

The full research thesis is included in this repository:

**A Causal Investigation of Deception Representations in Gemma-2-2B**

The thesis covers:

* background and related work
* experimental setup
* residual-stream probing
* sparse autoencoder analysis
* causal residual intervention
* cross-question transfer
* discussion
* limitations
* future experiments
* conclusions

---

# Key Takeaway

The central result of this project is not simply that a classifier can distinguish truth from lies.

The more important finding is the separation between **what can be decoded** and **what can be causally manipulated**.

The experiments suggest:

```text
Truthful / deceptive states
            ↓
       Highly decodable
            ↓
     SAE features predict
            ↓
   Individual SAE features
       fail to control
            ↓
 Dense residual intervention
       changes behavior
            ↓
 Question-specific directions
            ↓
 No reliable universal
     truth direction
```

In other words:

> **A behavior can be strongly represented without being localized, and a causal intervention can succeed without revealing a universal semantic direction.**

This makes the study a small but concrete case study in mechanistic interpretability: **detectability alone is not enough; causal intervention is necessary to understand whether an internal signal actually matters for behavior.**

---

# Citation

If you use this work, please cite the associated thesis/repository:

```bibtex
@misc{hoque2026deception,
  author       = {Mohibul Hoque},
  title        = {A Causal Investigation of Deception Representations in Gemma-2-2B},
  year         = {2026},
  note         = {Mechanistic interpretability study of deception representations,
                  probing, sparse autoencoders, residual-state intervention,
                  and cross-question transfer}
}
```

---

# Disclaimer

This project is an exploratory mechanistic interpretability study.

The results should **not** be interpreted as evidence that Gemma-2-2B possesses a general-purpose internal "lie detector," a universal truthfulness direction, or a general strategic deception mechanism.

The study specifically investigates **instructed false answering on a controlled factual benchmark** and should be interpreted within that experimental scope.
