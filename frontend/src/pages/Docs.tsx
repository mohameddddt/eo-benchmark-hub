import { Link } from "react-router-dom";
import { CodeBlock } from "../components/ui/ui";
import { STATIC } from "../lib/api";

const SECTIONS = [
  ["overview", "Overview"], ["workflow", "Submitting a model"], ["formats", "File formats"], ["protocol", "Evaluation protocol"],
  ["robustness", "EuroSAT-C"], ["api", "API & tokens"], ["selfhost", "Running the hub"], ["faq", "FAQ"],
];

export default function Docs() {
  return (
    <div className="container">
      <div className="page-head"><h1>Documentation</h1>
        <p className="lead">How benchmarks are built, how submissions are scored, and how to get a model onto the boards.</p></div>
      <div className="docs-grid">
        <nav className="toc" aria-label="On this page">{SECTIONS.map(([id, t]) => <a key={id} href={`#${id}`}>{t}</a>)}</nav>
        <article className="prose">
          <h2 id="overview" style={{ marginTop: 0 }}>Overview</h2>
          <p>EO-Bench Hub evaluates <b>predictions</b>, not code. You train wherever you like — the reference models were trained on free Kaggle T4 GPUs — run inference on a benchmark's hidden test images, and upload the prediction file. The server scores it against labels that never leave the server, with a versioned evaluator that is identical for everyone.</p>
          <p>Each benchmark fixes four things: a dataset, a deterministic split (<code>split_v2</code>), an evaluator version and a primary metric. Changing any of them creates a new version rather than silently moving the goalposts.</p>

          <h2 id="workflow">Submitting a model</h2>
          <ol className="soft">
            <li>Open a benchmark's <b>Dataset</b> tab and download the public manifest (sample ids and splits), or fork the reference Kaggle notebook — it computes the split with exactly the server's code.</li>
            <li>Train on <code>train</code>, select models on <code>val</code>, sanity-check on <code>test_public</code> (labels released). Never train on either test split.</li>
            <li>Predict every <code>test_hidden</code> id and write the file described on the benchmark's <b>Submit</b> tab.</li>
            <li>Upload it (web or API). Malformed files are rejected with a readable report and don't use quota.</li>
            <li>Results are private. Add a model card and publish to appear on the leaderboard.</li>
          </ol>

          <h2 id="formats">File formats</h2>
          <h3>Scene classification (EuroSAT, RESISC45)</h3>
          <p>CSV with one row per hidden id. Probability columns are optional but recommended — they unlock calibration metrics.</p>
          <CodeBlock code={"sample_id,predicted_label,prob_AnnualCrop,prob_Forest,…\nes-3f9a0c1d2e4b,Forest,0.0012,0.9931,…"} />
          <h3>Robustness (EuroSAT-C)</h3>
          <p>Same CSV without probabilities, covering each clean hidden id and its 21 corrupted variants <code>&lt;id&gt;@&lt;corruption&gt;-&lt;severity&gt;</code>.</p>
          <h3>Multi-label classification (BigEarthNet-S2)</h3>
          <p>CSV with one row per hidden patch: <code>sample_id,prob_00,…,prob_18</code>, one probability per class in canonical (alphabetical) order. Primary metric is macro mAP, so thresholds don't matter for ranking; F1 and subset accuracy use 0.5.</p>
          <h3>Semantic segmentation (LandCover.ai)</h3>
          <p>ZIP of single-channel PNGs named <code>&lt;tile_id&gt;.png</code>, pixel values = class index (0 background, 1 building, 2 woodland, 3 water, 4 road), 512×512.</p>

          <h2 id="protocol">Evaluation protocol</h2>
          <h3>Splits and ids</h3>
          <p>Splits are stratified per class with a fixed seed (70/10/10/10) or, for LandCover.ai, the official tile split with the official test half held out. Sample ids are <b>content hashes of the image bytes</b>. The first version of the hub used filenames such as <code>AnnualCrop_1</code> — the label was in the id, and predicting the id prefix scored 100% on the "hidden" split. Content hashes are reproducible from the official download and carry no label information. Exact byte-duplicates are removed so no image can sit in train and test at once.</p>
          <h3>Hidden is not secret</h3>
          <p>All benchmarks are built on public datasets, so a determined cheater could recover test labels by matching pixels. The hub therefore guarantees <i>protocol integrity</i> (same ids, same versioned metric code for everyone), no trivial leaks, and transparency: published entries need a model card, and entries the maintainers reproduced from the linked notebook carry a <b>Verified</b> badge.</p>
          <h3>Confidence intervals and tiers</h3>
          <p>Every primary metric ships with a 95% percentile bootstrap interval (1,000 resamples, fixed seed — re-evaluating reproduces it exactly). Images are the resampling unit for classification; tiles for segmentation, because pixels within a tile are strongly correlated; base images (with all their corrupted variants) for EuroSAT-C.</p>
          <p>The leaderboard groups entries into <b>tiers</b>: the leader's interval opens tier 1 and every entry whose interval reaches into it joins. Entries in one tier are not distinguishable at this test-set size — a 0.2-point gap on 2,700 images is noise.</p>
          <h3>Leaderboard rules</h3>
          <ul>
            <li>One row per (researcher, model name); repeated submissions keep their best score.</li>
            <li>Weekly quota per benchmark (default 20 scored submissions). Rejected files are free.</li>
            <li>Evaluators are versioned (<code>scene_classification/0.2.0</code>, …). A metric fix bumps the version and maintainers can re-score existing submissions.</li>
          </ul>

          <h2 id="robustness">EuroSAT-C</h2>
          <p>Seven corruptions modelled on real failure modes of optical EO pipelines, each at three severities: Gaussian sensor noise, defocus blur, thin haze (spatially varying veil), opaque cumulus clouds, JPEG downlink compression, coarser ground sampling and band-to-band misregistration. Corruptions are pure functions of (image, sample id, corruption, severity), so you regenerate the exact test set locally:</p>
          <CodeBlock lang="python" code={"from eobench_eval.corruptions import all_variants, corrupt, variant_id\n\nfor corruption, severity in all_variants():\n    x = corrupt(img_uint8_hwc, sample_id, corruption, severity)\n    vid = variant_id(sample_id, corruption, severity)   # 'es-…@haze-2'"} />
          <p>Primary metric is mean corruption accuracy (mCA). Relative robustness = mCA ÷ clean accuracy shows how much of its clean accuracy a model keeps.</p>

          <h2 id="api">API &amp; tokens</h2>
          <p>Create a token in <Link to="/account">your account</Link>. Tokens authenticate the same endpoints as the website; interactive docs live at <code>/api/docs</code> on a running server.</p>
          <CodeBlock lang="bash" code={"export EOB_TOKEN=eob_…\ncurl -X POST $HUB/api/benchmarks/eurosat/submissions \\\n  -H \"Authorization: Bearer $EOB_TOKEN\" \\\n  -F file=@predictions.csv -F model_name=\"ConvNeXt-T\"\n\ncurl $HUB/api/submissions/<id> -H \"Authorization: Bearer $EOB_TOKEN\""} />

          <h2 id="selfhost">Running the hub</h2>
          <p>The default configuration needs only Python and Node — SQLite and an in-process evaluation thread. Docker Compose (Postgres, Redis, Celery workers) remains available for a multi-user deployment.</p>
          <CodeBlock lang="bash" code={"pip install -r backend/requirements.txt -e evaluation\npython scripts/seed.py                      # benchmarks + baselines\ncd frontend && npm install && npm run build && cd ..\nuvicorn app.main:app --app-dir backend    # http://localhost:8000"} />
          {STATIC && <p className="small muted">You are viewing a static snapshot; the commands above give you the full interactive hub.</p>}

          <h2 id="faq">FAQ</h2>
          <h3>Why are some 98% and 99% models in the same tier?</h3>
          <p>Because on 2,700 test images their 95% intervals overlap. Ranking them apart would be ranking noise.</p>
          <h3>Why are my results private?</h3>
          <p>So you can iterate without publicly probing the test set. Publish when you have a model card worth sharing.</p>
          <h3>Can I use external data or pretraining?</h3>
          <p>Yes — declare it in the model card. The leaderboard shows each entry's pretraining so comparisons stay honest.</p>
        </article>
      </div>
    </div>
  );
}
