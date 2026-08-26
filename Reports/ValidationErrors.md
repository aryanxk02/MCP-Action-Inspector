# Validation Error Analysis

## Scope and method

This review examines the incorrectly predicted validation examples produced by the DistilBERT classifier. The supplied error listing contains **27 validation errors**, which is sufficient to satisfy the requirement to manually review at least 20 errors.

The classifier predicts one of six mutually exclusive classes:

- `Read`
- `Write`
- `Execute`
- `Destructive`
- `Financial`
- `Other`

The model input is constructed from the tool `name`, `description`, and serialized `input_schema`, with a maximum sequence length of 256 tokens. The model was trained for one epoch using standard cross-entropy, without class weighting or oversampling. These design choices are important when interpreting the errors below.

## Summary of the reviewed errors

The reviewed errors are not random. They cluster strongly around a small number of semantic boundaries:

1. **Read vs Write is the dominant failure mode.** Many tools that expose a capability through an API-style description are predicted as the opposite side of the read/write boundary.
2. **Execute is frequently collapsed into Read or Write.** The model appears to focus on the tool's input/output wording rather than the operational fact that it causes an action to occur outside the model.
3. **Rare/high-risk classes are poorly separated.** `Financial` examples are predicted as `Destructive`, while a clearly destructive registry tool is predicted as `Read`.
4. **Tool descriptions contain mixed semantics.** Several MCP tools expose multiple operations in one description. The model has to infer the dominant action category from a list of capabilities rather than a single clean action.
5. **Long descriptions and schemas make the classification problem harder.** The project truncates inputs at 256 tokens, so decisive information near the end of a description/schema may not reach the classifier.

The overall validation accuracy of 93.02% therefore needs to be interpreted alongside the much lower macro F1 of 0.7079. The error examples illustrate why: the majority `Read` class is relatively easy to predict, while minority and semantically adjacent classes remain difficult.

## Reviewed examples

### 1. Read vs Write confusion

This is the clearest recurring pattern in the reviewed errors.

| Tool | Actual | Predicted | Why it is difficult |
|---|---|---|---|
| `veo_image_to_video` | Write | Read | It describes generating/returning a video, which can look like an information-producing API rather than a state-changing operation. |
| `handoff_evidence` | Write | Read | "Transfer approved evidence" is an action, but the description is short and contains no strong write verb such as create/update/delete. |
| `remember` | Write | Read | The description says to store a decision/fact, but the model can associate the surrounding language with information retrieval. |
| `track_usage` | Write | Read | Recording telemetry is a state-changing operation, but the word "Record" can be overwhelmed by the informational nature of telemetry. |
| `response_type` | Write | Read | The name is abstract and the description is largely metadata-oriented, making the write semantics difficult to identify. |
| `fork_repository` | Write | Read | Forking creates a new repository, but the model may associate repositories with reading/browsing. |
| `browser_mouse_wheel` | Read | Write | Browser interaction is an external action, but the project labels this example as `Read`; this creates an inherently difficult boundary for a text-only classifier. |
| `aps_sign_attribution_consent` | Read | Write | The description contains the strong action verb "adds", making a write prediction understandable even though the supplied label is `Read`. |
| `respond_to_intro` | Read | Write | "Approve to share" and "decline" look like state-changing actions. |
| `rotate_key` | Read | Write | Key rotation clearly sounds like a state change, so the supplied `Read` label is difficult to infer from surface text alone. |
| `sub_delegate` | Read | Write | "Sub-delegate authority" strongly resembles a state-changing authorization operation. |
| `archive_need` | Read | Write | "Archive" normally implies a mutation, which conflicts with the supplied `Read` label. |
| `archive_offer` | Read | Write | Same semantic issue as `archive_need`. |
| `create_payment_intent` | Read | Write | "Create" strongly suggests mutation. |
| `register` | Read | Write | Registration usually creates persistent state and therefore looks like a write. |
| `register_webhook` | Read | Write | Registering a webhook is clearly an external configuration change. |
| `submit_delivery` | Read | Write | Submission normally creates or changes server-side state. |
| `ai-safety-guard` | Read | Write | The description contains operational instructions and may resemble a configuration/action tool. |
| `associate_time_series_to_asset_property` | Read | Write | "Associate" is a strong state-changing verb. |

### Interpretation

The important observation is that some of these are not simply model mistakes. They expose a **label-boundary problem** in the dataset itself. If `register`, `create_payment_intent`, `archive_need`, `rotate_key`, and `sub_delegate` are intentionally labelled `Read`, then the model cannot reliably recover that policy from natural language because the surface semantics point strongly toward `Write`.

This suggests that the annotation guideline needs to define the taxonomy in terms of **observable runtime behavior**, not merely the wording of the tool description. Otherwise the classifier will learn inconsistent examples and reproduce those inconsistencies.

A concrete improvement is to document a decision rule such as:

> Classify by the primary externally observable effect of the tool. Creating, modifying, deleting, registering, submitting, associating, rotating, or storing persistent state should be considered a write-like operation unless an explicit dataset policy says otherwise.

If the existing labels intentionally follow a different policy—for example, classifying by information-flow direction rather than side effects—that policy should be written explicitly and applied consistently to both training and validation data.

## 2. Execute vs Read/Write confusion

Several errors show that `Execute` is difficult to distinguish from ordinary retrieval or mutation.

| Tool | Actual | Predicted | Failure pattern |
|---|---|---|---|
| `aps_compute_compute_axis_weights` | Execute | Read | The description describes computation and returns a value, so it resembles a pure read/query operation. |
| `crypto_dtd` | Execute | Read | It calculates a score and returns analysis, which looks informational despite requiring computation. |
| `smart_translate_workflow` | Execute | Read | It performs a multi-step workflow but also returns a translation and quality metrics, making it look like a read-style API. |
| `gateway_target_synchronize` | Execute | Write | Synchronization changes or refreshes an external catalog, so the model sees write-like behavior instead of execution semantics. |

There is also a broader ambiguity here: **what exactly does `Execute` mean?** If `Execute` means "runs a computation/workflow" regardless of whether the external world changes, then `aps_compute_compute_axis_weights`, `crypto_dtd`, and `smart_translate_workflow` have strong Execute evidence. If `Execute` means "runs an external command or side effect," then purely computational tools may legitimately belong elsewhere.

### Concrete improvement

Define `Execute` using an explicit operational rule. For example:

> `Execute` = the tool performs a computation, workflow, command, or multi-step operation whose primary purpose is to carry out an operation rather than merely retrieve or persist data.

Then add hard-negative training examples where the same nouns appear in different action contexts:

- `get_balance` → Read
- `calculate_balance` → Execute
- `update_balance` → Write
- `delete_account` → Destructive

This teaches the model that the **verb and operational effect**, rather than the domain noun, determines the category.

## 3. Minority/high-risk class failures

The reviewed errors also show problems with classes that have fewer training examples or stronger semantic overlap.

### Financial → Destructive

Two related examples are:

- `cancel_metadata_transfer_job` — actual `Financial`, predicted `Destructive`
- `create_metadata_transfer_job` — actual `Financial`, predicted `Destructive`

The descriptions concern metadata transfer operations and contain words such as "cancel", "bulk import/export", and "job creation". A model can easily associate `cancel` with destructive behavior or `create` with state mutation without recognizing the intended financial taxonomy.

This is a classic example where **domain classification and action classification interact**. If `Financial` is supposed to be based on financial impact, the training data should contain many explicit examples where financial operations are contrasted with ordinary destructive/write operations.

### Destructive → Read

The `registry` example is particularly important:

- `registry` — actual `Destructive`, predicted `Read`

Its description contains actions including `install`, `uninstall`, `activate`, and `deactivate`. These are strong state-changing/destructive signals, but the tool is also described as a registry and search/discovery mechanism. The model appears to have been attracted to the retrieval-oriented part of the description.

This demonstrates that **multi-operation tools need special handling**. A description containing both `find_tool` and `uninstall` is not semantically equivalent to a simple read-only tool.

### Concrete improvements

1. Add class-weighted loss or balanced sampling so rare classes contribute more strongly during training.
2. Oversample `Financial`, `Other`, and other minority classes rather than relying on one-epoch standard cross-entropy.
3. Add targeted hard-negative examples such as:
   - financial read vs financial write
   - financial write vs destructive
   - registry discovery vs registry installation/uninstallation
   - execute computation vs read retrieval
4. Report per-class precision, recall, and F1 after every training run, not just accuracy and aggregate F1.
5. Use macro F1 for model selection, as already done, but also inspect minority-class recall explicitly.

## 4. Multi-operation descriptions

A recurring source of ambiguity is that an MCP tool may expose many operations under one tool description.

The `registry` tool is the strongest example. It contains retrieval operations such as `find_tool`, `find_tools`, `get_schema`, `list`, and `browse`, but also state-changing operations such as `install`, `uninstall`, `activate`, and `deactivate`.

A single six-way label forces the model to compress all of these semantics into one category. The same issue appears in tools such as `gateway_target_synchronize` and `smart_translate_workflow`, where a description combines multiple steps and effects.

### Concrete improvement

Introduce a deterministic preprocessing feature that extracts the **action-bearing verbs** and operation names from the description/schema. For example:

```text
operation verbs: find, list, get, install, uninstall, activate, deactivate
mutation verbs: install, uninstall, activate, deactivate
retrieval verbs: find, list, get, browse
```

This does not have to replace the transformer input. It can be added as a second signal or used to create additional training examples.

A stronger long-term design would classify each individual operation first and then aggregate the results to a tool-level label using an explicit policy such as:

1. Destructive if any exposed operation is destructive and the tool is classified by maximum risk.
2. Financial if its primary operation has financial impact.
3. Execute if execution is the dominant behavior.
4. Write if the tool primarily changes persistent state.
5. Read otherwise.

The exact precedence should be determined by the project specification rather than assumed by the model.

## 5. Long-input truncation

The current model uses a maximum sequence length of 256 tokens and truncates from the tail. This is particularly relevant because the model input includes the serialized `input_schema` after the name and description.

That means a long description can consume most or all of the 256-token budget before the schema is reached. If the schema contains the decisive argument or operation information, that evidence may never reach the classifier.

This is especially concerning for descriptions that contain many operations, examples, or verbose documentation.

### Concrete improvements

- Increase the maximum sequence length if CPU/memory budget permits.
- Put the most informative fields first: name, concise description, operation names, then schema details.
- Extract only semantically useful schema fields rather than serializing the entire schema.
- Add a preprocessing step that creates a compact representation such as:

```text
name: registry
description: MCP server registry
actions: find_tool, find_tools, get_schema, list, install, uninstall, activate, deactivate
```

This can preserve important information without requiring a much longer context window.

## 6. Recommended model/data improvements

### Priority 1 — Fix annotation consistency

Before changing the model, audit examples where the supplied label contradicts the obvious runtime effect. The reviewed examples include several apparent cases:

- `create_payment_intent`: `Read` despite `create`
- `register`: `Read` despite registration
- `register_webhook`: `Read` despite registration
- `archive_need`: `Read` despite archive
- `archive_offer`: `Read` despite archive
- `rotate_key`: `Read` despite key rotation
- `sub_delegate`: `Read` despite delegation
- `associate_time_series_to_asset_property`: `Read` despite association

If these labels are intentional, document the rule that makes them `Read`. If they are annotation errors, correct them and retrain.

### Priority 2 — Handle class imbalance

The current SLM uses standard cross-entropy without class weighting or oversampling. Change the training procedure to use either:

- weighted cross-entropy based on inverse/effective class frequency, or
- a balanced sampler/oversampling strategy.

This should particularly help `Financial` and `Other` and reduce the tendency to fall back to the majority `Read` class.

### Priority 3 — Train longer and use early stopping

The model is trained for only one epoch. One epoch is a reasonable fast baseline, but it gives the classifier limited opportunity to learn subtle boundaries such as `Read` vs `Write` and `Execute` vs `Read`.

A practical experiment would be 3–5 epochs with the existing validation-macro-F1 checkpointing. Keep the learning rate conservative and stop when validation macro F1 stops improving.

### Priority 4 — Add hard-negative examples

Construct pairs that differ by only one semantic dimension:

```text
Read:     get_account
Write:    update_account
Execute:  calculate_account_score
Destructive: delete_account
Financial: transfer_money
```

Then create similar examples across the MCP domain. Hard negatives are likely to be more valuable than simply adding more random training examples.

### Priority 5 — Add an uncertainty/abstention policy

The current classifier always returns one of the six classes using `argmax`. That is useful for deterministic evaluation but risky for ambiguous cases.

For deployment, record the top probability and margin between the top two classes. If confidence is low, route the example for human review rather than treating the prediction as certain.

For example:

```text
prediction: Write
confidence: 0.43
runner_up: Read
margin: 0.04
status: REVIEW
```

This is particularly appropriate for `Read`/`Write` and `Execute`/`Read` boundaries.

## Conclusion

The reviewed 27 validation errors show that the model's weaknesses are concentrated rather than random. The most important problem is the **Read/Write boundary**, followed by **Execute vs Read/Write**, and then **minority/high-risk class confusion** such as Financial vs Destructive.

The errors also reveal a deeper issue: several examples have surface descriptions whose apparent runtime behavior conflicts with their supplied labels. This means that improving the model alone is unlikely to fully solve the problem. The highest-value first step is to make the annotation policy explicit and audit contradictory labels.

After that, the most promising model improvements are **class-weighted/balanced training, hard-negative examples, better preservation of operation/schema information, and longer training with macro-F1-based checkpointing**. For deployment, an uncertainty threshold and human-review path should be added because some MCP tools genuinely expose multiple behaviors that cannot be represented cleanly by a single categorical label.