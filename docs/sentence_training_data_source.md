# Data source and license record

The sentence labels and associated reasoning context used to build the 5,000-row
PSSN training set were imported from the public release accompanying the paper
[*Reasoning Models Know What's Important, and Encode It in Their Activations*](https://github.com/technion-cs-nlp/reasoning-importance).
The paper repository does not show an explicit repository license file, so its
public availability should not be treated as permission to redistribute its
derived labels or generated traces.

The release uses HARP and MATH-500. The [HARP repository](https://github.com/aadityasingh/HARP)
and the [original MATH repository](https://github.com/hendrycks/math) provide MIT
licenses. Those licenses do not automatically resolve rights for the paper's
derived annotations, model-generated reasoning traces, or all source contest
questions. The separate source-notes folder contains an aggregate inventory of
source files, their checksums, source-level metadata, and selected counts by model
and dataset. It does not contain a row-ID-to-source-record crosswalk; individual
training rows cannot be joined to a particular source record through the exported
metadata.

Use the imported records for research only with clear citation and provenance.
Before publicly redistributing the derived dataset, check each model's terms and
obtain permission or institutional legal guidance. This is a research provenance
note, not legal advice.

The output's two label names are a mapping for PSSN's model-facing schema. They do
not mean PSSN generated these labels: the source experiment defines removability
using its own procedure, which is not the PSSN 2-of-3 rule. Report this set as
transferred supervision and keep evaluation separate from PSSN-origin labels.
