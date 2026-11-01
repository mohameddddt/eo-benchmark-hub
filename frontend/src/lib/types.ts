// Mirrors backend/app/schemas.py
export type TaskType =
  | "scene_classification"
  | "scene_classification_robustness"
  | "semantic_segmentation"
  | "multilabel_classification";

export interface UserRef { id: string; name: string; affiliation: string | null }
export interface Me extends UserRef { email: string; website: string | null; bio: string | null; role: string }

export interface MetricInfo { name: string; label: string; description: string; higher_is_better: boolean; unit: string }

export interface BenchmarkSummary {
  slug: string; name: string; tagline: string; task_type: TaskType; modality: string;
  primary_metric: string; primary_metric_label: string; n_classes: number; n_entries: number;
  ready: boolean; top: { submission_id: string; model_name: string; score: number } | null; position: number;
}

export interface FormatSpec {
  file_type?: string; header?: string; rows?: string; example?: string; generator?: string;
  columns?: { name: string; description: string }[];
}

export interface BenchmarkDetail extends BenchmarkSummary {
  description: string; dataset_url: string; paper_url: string; starter_url: string; citation: string;
  license: string; classes: string[] | null; config: Record<string, any> | null;
  stats: {
    splits?: Record<string, number>; fingerprint?: string; n_samples?: number;
    class_counts?: Record<string, number>; hidden_pixel_share?: Record<string, number>;
    evaluated_images?: number; duplicates_dropped?: number;
  } | null;
  split_version: string; evaluator_version: string; metrics: MetricInfo[]; format_spec: FormatSpec;
  weekly_quota: number;
}

export interface LeaderboardCard {
  architecture: string; family: string; pretraining: string; num_params: number | null;
  gflops: number | null; throughput: number | null; input_size: number | null;
  external_data: boolean; notebook_url: string;
}

export interface LeaderboardEntry {
  rank: number; tier: number; submission_id: string; model_name: string; user: UserRef;
  verified: boolean; created_at: string; score: number; ci_low: number | null; ci_high: number | null;
  metrics: Record<string, number>; card: LeaderboardCard | null;
}

export interface ResultOut { metric_name: string; value: number; ci_low: number | null; ci_high: number | null }

export interface ModelCard {
  architecture: string; family: string; backbone: string; num_params: number | null; gflops: number | null;
  throughput: number | null; throughput_setup: string; input_size: number | null; pretraining: string;
  external_data: boolean; epochs: number | null; optimizer: string; augmentation: string; hardware: string;
  training_time: string; framework: string; repo_url: string; paper_url: string; notebook_url: string;
  notes: string; training_log?: Record<string, number>[] | null; extra?: Record<string, any> | null;
}

export type Status = "UPLOADED" | "QUEUED" | "VALIDATING" | "EVALUATING" | "COMPLETED" | "REJECTED" | "FAILED";

export interface Submission {
  id: string; status: Status;
  benchmark: { slug: string; name: string; task_type: TaskType; primary_metric: string };
  user: UserRef; model_name: string; description: string; file_format: string; file_size: number;
  original_filename: string; checksum: string; error_report: { errors?: string[]; traceback?: string } | null;
  evaluator_version: string; is_published: boolean; verified: boolean; created_at: string;
  evaluated_at: string | null; results: ResultOut[]; details: Record<string, any> | null;
  model_card: ModelCard | null; rank: number | null; n_ranked: number | null;
}

export interface Stats {
  benchmarks: number; task_types: number; models: number; evaluations: number; researchers: number;
  leaders: { benchmark_slug: string; benchmark_name: string; metric_label: string; model_name: string;
             submission_id: string; score: number; runner_up_gap: number | null }[];
  recent: { submission_id: string; model_name: string; benchmark_slug: string; benchmark_name: string;
            user_name: string; metric_label: string; score: number; evaluated_at: string | null }[];
}

export interface PublicUser {
  id: string; name: string; affiliation: string | null; website: string | null; bio: string | null;
  created_at: string;
  entries: { benchmark_slug: string; benchmark_name: string; metric_label: string; rank: number; of: number;
             model_name: string; score: number; submission_id: string; verified: boolean }[];
}

export interface ApiToken { id: string; name: string; prefix: string; created_at: string; last_used_at: string | null }

// media/<slug>/assets.json written by the Kaggle kernels
export interface ClassificationAssets {
  classes: Record<string, string[]>;
  gallery: { sample_id: string; image: string; label: string; kind: "mistake" | "correct";
             predictions: Record<string, { label: string; confidence: number }> }[];
  corruption_examples: { index: number; label: string }[];
  thumb_px: number;
  model_names?: Record<string, string>;
}
export interface SegmentationAssets {
  samples: { sample_id: string; image: string; gt: string; class_fractions: number[];
             predictions: Record<string, { mask: string; miou: number }> }[];
  palette: number[][]; classes: string[];
  model_names?: Record<string, string>;
}
