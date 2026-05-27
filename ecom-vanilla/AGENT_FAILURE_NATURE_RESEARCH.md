# Research Notes: Why Agents Miss Instructions They Have Read

## Research Question

I want to understand the nature of mistakes where an agent has access to the
right instructions, appears to read them, but still violates them in the final
answer.

The goal is not to add a new reminder for every discovered misunderstanding.
That approach does not scale. The goal is to understand what kind of failure this
is and what general agent design patterns reduce it.

## Example

In an ECOM task, the agent found the correct product and answered the factual
question correctly. The grader still rejected the answer:

```text
answer missing required reference '/proc/catalog/FST-3SJKL8BF.json'
```

The agent had read `/AGENTS.MD`, including this instruction:

```text
When responding with reference - provide full path in the repo to the object.
```

But it returned broad references:

```text
/proc/catalog/products
/proc/catalog/product_properties
/proc/catalog/inventory
```

instead of the exact object path:

```text
/proc/catalog/FST-3SJKL8BF.json
```

## Initial Diagnosis

This does not look like a simple "the model did not read the instructions"
failure.

It also does not look like a simple "the model is unable to understand the
instruction" failure. The model understood that references were needed and
provided some.

The better description is:

```text
instruction operationalization failure
```

The model understood the instruction abstractly, but did not convert it into a
reliable procedure during tool use.

The instruction said:

```text
provide full path in the repo to the object
```

The operational version should have been:

```text
When using SQL to identify an object, preserve or recover its exact path column.
Before finalizing, ensure every grounding reference points to the concrete object
mentioned in the answer.
```

The agent did not make that conversion.

## What Actually Broke

The agent solved the task through SQL. That was a reasonable tool choice.

But the SQL queries selected fields such as:

```text
sku, name, brand, series, model, properties, inventory
```

They did not select:

```text
products.path
```

So the agent's working memory contained:

```text
SKU FST-3SJKL8BF matches the requested product.
```

But it did not contain:

```text
/proc/catalog/FST-3SJKL8BF.json
```

At finalization time, it had a correct claim but incomplete provenance. It then
filled `grounding_refs` with nearby conceptual sources instead of the exact
object.

## Hypothesis

Many agent failures of this kind are not failures of instruction visibility.
They are failures to preserve task contracts across a multi-step chain:

```text
instruction -> tool choice -> evidence collection -> working memory -> final answer
```

The instruction is present at the beginning, but its requirements are not
maintained as an invariant throughout the loop.

## Failure Pattern

This class of error has several recurring properties:

1. The instruction is abstract.

   It describes the desired final property, but not the intermediate operations
   required to guarantee that property.

2. The tool result is lossy.

   SQL answers the factual question but may omit identity/provenance fields
   needed later.

3. The model collapses provenance.

   It treats "where I searched" as equivalent to "the object I am referencing".

4. The schema is too permissive.

   `grounding_refs: List[str]` accepts broad directories, table names, and exact
   object paths equally.

5. There is no final audit.

   Nothing checks whether the final output satisfies the original contract.

## Why More Reminders Are Not Enough

Adding a reminder such as:

```text
Remember to include exact product paths.
```

may fix this specific task. But it creates a brittle collection of special cases.
The next failure may involve customers, baskets, payments, policies, employee
records, or some other object type.

The deeper issue is not "product paths". The deeper issue is that the agent lacks
a general mechanism for carrying contractual obligations into its tool strategy
and final validation.

## Better Research Direction

The useful direction is to turn important instructions into invariants.

Examples:

```text
Every final factual claim must have concrete supporting evidence.

Every referenced object must have an exact object identity, not just a table,
directory, or search location.

If a tool result proves a claim but lacks the required evidence identity, the
agent must perform another step before finalizing.
```

These are not ECOM-specific reminders. They are general properties of a robust
agent loop.

## Possible Design Patterns

### Evidence Stack

Maintain an explicit internal structure:

```text
claim -> evidence fields -> source object path -> confidence
```

The final answer can only cite evidence already present in the stack.

### Contract Extraction

At task start, convert instructions into a small set of active obligations:

```text
required answer format
required references
security constraints
required object identity
```

These obligations should be checked before finalization.

### Finalization Gate

Before accepting `report_completion`, run a deterministic or model-assisted
validator:

```text
Are grounding refs concrete object paths?
Do they correspond to objects named in the answer?
Are broad directories being used as substitutes for object refs?
```

If validation fails, feed the failure back into the agent instead of submitting.

### Tool Result Discipline

When using indirect tools such as SQL, preserve identity columns by default:

```text
path, id, sku, file, object reference
```

This is not a special product rule. It is a general rule: if a tool returns facts
about objects, also return the object's identity.

## Working Theory

The model can read and understand instructions, but understanding is not enough.
In a multi-step agent, instructions need to become durable state and executable
checks.

The mistake is best viewed as:

```text
contract drift during tool-mediated reasoning
```

The contract existed in context, but drifted away from the agent's operational
state as it solved the task.

## Practical Implication

Do not respond to every failure by adding another narrow instruction.

Instead, identify which contract was violated, then ask:

```text
Could this contract be represented as state?
Could it be checked before final output?
Could tool results be shaped so the required evidence cannot be lost?
```

That is more likely to improve the agent broadly than a growing list of custom
ECOM reminders.

## Run-Level Evidence: Qwen 20260527_180319

The 50-task run
`runs/20260527_180319_Qwen_Qwen3.5-397B-A17B-fast_score_8.51_git732d2eb.log`
shows that the same underlying failure is broader than product paths.

Summary from score details:

```text
perfect tasks: 3
partial tasks: 3
zero-score tasks: 44
no answer from length-limit parsing failure: 8
expected-outcome mismatches: 19
missing required references: 12
invalid references: 2
strict answer-format failures: 4
```

These are not independent root causes, but they reveal several contract types
that were not preserved by the agent loop.

### 1. Provenance Contract Drift

The original example was an object-reference failure. The run contains the same
class in several forms:

```text
answer missing required reference '/docs/returns.md'
answer missing required reference '/docs/security.md'
answer missing required reference '/docs/checkout.md'
answer contains invalid reference '/proc/baskets/basket_001.json'
grounding_refs=['products.sku', 'product_properties.sku', 'inventory.store_id']
```

The agent often knew the relevant fact, but the final references were either:

- broad schema/table identifiers,
- plausible but nonexistent paths,
- object paths without the policy document that authorized the decision.

This is not one "remember to cite X" problem. It is one provenance invariant:

```text
Every final claim or action must carry concrete source identities through the
whole reasoning chain.
```

For policy decisions, the concrete source identity is not only the business
object. It also includes the policy document used to justify the decision.

### 2. Outcome Taxonomy Drift

Many failures had a good or partly good user-facing message but selected the
wrong ECOM outcome:

```text
expected OUTCOME_OK, got OUTCOME_ERR_INTERNAL
expected OUTCOME_OK, got OUTCOME_NONE_CLARIFICATION
expected OUTCOME_DENIED_SECURITY, got OUTCOME_OK
expected OUTCOME_NONE_UNSUPPORTED, got OUTCOME_OK
```

The model treated `outcome` as a natural-language mood label instead of a
strict protocol field. For example:

- "I found the answer but it is negative" became clarification or internal
  error.
- "A request is disallowed" sometimes became OK because the agent produced an
  explanation.
- "A tool action is technically possible but not authorized" was confused with
  unsupported or internal failure.

This suggests a separate invariant:

```text
Outcome is not inferred from response tone. It is selected from a deterministic
decision table based on task state: answered, needs user input, unsupported,
security denied, or internal runtime failure.
```

This should be handled by code or a finalization gate, not by another prose
reminder.

### 3. Strict Format Contract Drift

Some tasks were factually close but failed because the final answer did not obey
the requested exact format:

```text
Answer should be "[QTY:7]", but at least it contains it
Answer should be "count : 10", but at least it contains it
Answer should be "[QTY:4]"
Answer should contain '<NO>'
```

The agent solved the semantic task but did not preserve the final output
contract as an active constraint. This is the same pattern as reference drift:
the requirement existed in context but was not carried into finalization.

A scalable fix is not to add one reminder per format. It is to extract:

```text
final_answer_shape = exact | token_required | prose_allowed
```

Then validate the final message against that shape before submitting.

### 4. Policy Discovery Drift

Several missing-reference failures were policy-source failures:

```text
/docs/ops-policy-notes/catalogue-count-wood-drywall-screws-2025-06-22.md
/docs/catalogue-addenda/2024-07-17-reporting-pipe-fittings-salzburg.md
/docs/current-updates/catalogue-counting-2024-07-17-work-tops-vienna.md
/docs/payments/3ds.md
/docs/returns.md
/docs/security.md
```

The root `AGENTS.MD` told the agent:

```text
When you apply a policy from `docs`, include that policy document as a grounding
reference in the final response.
When starting work, make sure to run tree -L 2 on docs folder.
```

But the harness only forced `tree -L 2 /`, `/AGENTS.MD`, `/bin/date`, and
`/bin/id`. The instruction to inspect docs remained a soft obligation. The
agent often did not convert it into a concrete startup action or later source
audit.

This should become an executable preflight:

```text
If task domain may involve policy, add docs tree/search/read to the evidence
plan before answering or acting.
If policy language appears in final answer, require at least one `/docs/...`
grounding ref.
```

### 5. Tool Affordance Drift

The agent sometimes had the right runtime capabilities but reached for the wrong
interface model.

In the archive task it tried:

```text
/bin/sh -c cat /archive/payment_batch_export_DapM8nE7pk.tsv
/bin/cat /archive/payment_batch_export_DapM8nE7pk.tsv
```

Those runtime tools did not exist, so it declared the task unsupported, even
though the agent wrapper exposes a `read` tool for arbitrary file paths.

In the SQL incident task, it hit:

```text
sql: write /tmp/ecom-sql-spool: no space left on device
```

Then it tried shell cleanup and `/bin/codex cat`, instead of reading the
documented incident workaround through the runtime `read` tool.

This is not a missing ecommerce rule. It is an affordance-model failure:

```text
The agent's internal idea of "how to read a file" diverged from the actual
tool surface.
```

Mitigations should be general:

- maintain a capability map in state,
- route common shell intentions like "cat file" to the wrapper's `read` action,
- after `NOT_FOUND` for a shell tool, retry using native file/search/list tools,
- make failed-tool recovery part of the loop rather than relying on the model to
  improvise.

### 6. Side-Effect Contract Drift

The run includes action tasks where the agent selected the wrong authorization
or mutation behavior:

```text
expected outcome OUTCOME_DENIED_SECURITY, got OUTCOME_OK
expected no file changes
payment fraud refs recovered ~8% EUR from fraud amount
```

This is the action-side version of provenance drift. Before any write,
checkout, discount, refund, or payment mutation, the agent should have a typed
side-effect contract:

```text
requested_action
actor_identity
required_role_or_policy
object_ids
precondition_evidence
expected_mutation
rollback_or_no-op expectation
```

The finalization gate should verify that the performed mutation matches the
contract. If the task asks for investigation only, expected mutation is empty.

### 7. Structured-Output Failure

Eight tasks failed before answer submission:

```text
Could not parse response content as the length limit was reached
```

At commit `732d2eb`, the agent used `client.beta.chat.completions.parse` with a
large Pydantic union as `response_format=NextStep`. Some model/provider
combinations expanded or failed inside that constrained format and hit the
16,384-token completion limit.

This is a protocol robustness issue, not an ECOM reasoning issue. A better
pattern is:

```text
Use a simpler action envelope.
Parse normal JSON manually.
Retry with compact validation feedback.
Keep tool history as explicit state/action/result text if native tool-call
messages confuse the provider.
```

The later implementation direction moved toward this: it added explicit JSON
extraction/retry logic and records each step as state, plan, action, and result.

## The Broader Failure Class

The best name for the broader class is:

```text
contract drift across the agent control loop
```

The violated contracts differ by task:

- reference/provenance contract,
- policy-authority contract,
- outcome-taxonomy contract,
- strict-answer-format contract,
- tool-affordance contract,
- side-effect/authorization contract.

But the mechanism is the same. The contract is present somewhere in text, yet it
is not represented as durable state, used to shape tool calls, or checked before
final submission.

## Avoiding Rule Collection

The scalable response is to avoid encoding every discovered failure as another
natural-language instruction. Instead, build a small number of reusable control
mechanisms.

### Contract Extraction

At task start, extract a compact contract:

```text
answer_shape
required_tokens
object_types
required_identity_fields
policy_domains
allowed_side_effects
authorization_requirements
freshness/source requirements
```

This contract should be visible in every loop iteration and updated when new
evidence changes the task state.

### Evidence Ledger

Do not let facts float as prose. Store them as:

```text
claim
object_type
object_id
object_path
supporting_fields
policy_path
tool_result_step
```

The final answer can only cite paths already present in the ledger.

### Tool-Result Shaping

When querying object tables, include identity and path columns by default:

```text
products.path
customers.path
baskets.path
payments.path
returns.path
stores.path
employees.path
```

When a table lacks paths, join to the table that has them or perform a recovery
lookup before finalizing.

### Deterministic Outcome Mapper

Replace free-form outcome choice with a decision table:

```text
answered successfully -> OUTCOME_OK
needs user choice/input -> OUTCOME_NONE_CLARIFICATION
cannot be done with available system capability -> OUTCOME_NONE_UNSUPPORTED
request is unauthorized or unsafe -> OUTCOME_DENIED_SECURITY
runtime/tool failure prevented completion -> OUTCOME_ERR_INTERNAL
```

The model may propose the outcome, but the gate should be able to override or
reject inconsistent choices.

### Finalization Gate

Before `report_completion`, run validation:

```text
Does message match exact requested format?
Are required tokens present?
Do grounding refs exist?
Are refs concrete object paths or required policy docs?
Do refs correspond to objects named in the answer?
If policy was applied, is the policy doc cited?
If action was performed, was it authorized and expected?
Is outcome consistent with the decision table?
```

Failed validation should become another agent step, not a submitted answer.

### Capability-Aware Recovery

Represent available tools as a map:

```text
read file -> read
list directory -> list/tree
search content -> search
query DB -> exec /bin/sql
checkout -> exec /bin/checkout
discount -> exec /bin/discount
payments/refunds -> exec /bin/payments
```

If the model asks for `/bin/sh`, `/bin/cat`, `ls`, or `rm`, normalize that
intent to the available wrapper tool or reject it before dispatch.

## Design Principle

The agent should not be a prompt plus a transcript. It should be a control loop
with explicit contracts:

```text
task text + instructions
  -> active contract
  -> evidence/tool plan
  -> evidence ledger
  -> side-effect plan
  -> finalization validator
  -> report_completion
```

This converts scattered rules into a small number of durable invariants. The
goal is not to make the model remember more. The goal is to make important
requirements impossible to silently drop.
