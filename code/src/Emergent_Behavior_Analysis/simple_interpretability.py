"""
simple_interpretability.py
==========================
Simple, hypothesis-friendly interpretability pipeline for BONES-SEED G1 data.

Outputs:
  - actor_task_features.csv
  - top_features_per_task_attr.json
  - dominant_attr_per_task_feature.json
	- task_attr_top3_summary.csv
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


FPS = 120.0
N_G1_CHANNELS = 35


def norm_relpath(path: str) -> str:
	return str(path).replace("\\", "/").strip().lower()


def load_g1_cache(cache_dir: str) -> Optional[Dict]:
	meta_path = os.path.join(cache_dir, "metadata.json")
	index_path = os.path.join(cache_dir, "motion_index.csv")
	data_path = os.path.join(cache_dir, "motion_data.f32")
	if not all(os.path.exists(p) for p in [meta_path, index_path, data_path]):
		return None

	with open(meta_path, "r", encoding="utf-8") as f:
		meta = json.load(f)
	idx_df = pd.read_csv(index_path)
	index_map = {
		norm_relpath(str(r["path"])): (int(r["offset"]), int(r["length"]))
		for _, r in idx_df.iterrows()
	}
	return {
		"data_path": data_path,
		"index_map": index_map,
		"total_frames": int(meta["total_frames"]),
		"num_channels": int(meta["num_channels"]),
	}


def read_g1_cached(cache_info: Dict, rel_path: str, mmap_obj=None):
	key = norm_relpath(rel_path)
	pos = cache_info["index_map"].get(key)
	if pos is None:
		return None, mmap_obj

	if mmap_obj is None:
		mmap_obj = np.memmap(
			cache_info["data_path"],
			dtype=np.float32,
			mode="r",
			shape=(cache_info["total_frames"], cache_info["num_channels"]),
		)

	offset, length = pos
	arr = np.array(mmap_obj[offset: offset + length], dtype=np.float32, copy=True)
	return arr, mmap_obj


def read_g1_csv(path: str) -> np.ndarray:
	df = pd.read_csv(path)
	if "Frame" in df.columns:
		df = df.drop(columns=["Frame"])
	return df.to_numpy(dtype=np.float32)


def binary_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
	"""Compute ROC AUC for binary labels without sklearn dependency."""
	y_true = np.asarray(y_true)
	y_score = np.asarray(y_score)
	mask = np.isfinite(y_true) & np.isfinite(y_score)
	y_true = y_true[mask]
	y_score = y_score[mask]

	if y_true.size < 2:
		return None

	classes = np.unique(y_true)
	if classes.size != 2:
		return None

	y_bin = (y_true == classes.max()).astype(np.int64)
	n_pos = int(y_bin.sum())
	n_neg = int((1 - y_bin).sum())
	if n_pos == 0 or n_neg == 0:
		return None

	order = np.argsort(y_score)
	sorted_scores = y_score[order]
	ranks = np.empty_like(sorted_scores, dtype=np.float64)

	i = 0
	while i < sorted_scores.size:
		j = i + 1
		while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
			j += 1
		avg_rank = (i + 1 + j) / 2.0
		ranks[i:j] = avg_rank
		i = j

	full_ranks = np.empty_like(ranks)
	full_ranks[order] = ranks
	sum_pos_ranks = float(full_ranks[y_bin == 1].sum())
	auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
	return float(auc)


def compute_multi_axis_features(angles: np.ndarray, fps: float) -> Tuple[float, float, float]:
	# angles: (T, n_axes)
	vel = np.diff(angles, axis=0) * fps
	vel_mag = np.linalg.norm(vel, axis=1)
	mean_vel = float(np.mean(vel_mag))
	peak_vel = float(np.max(vel_mag))
	rom_axes = np.max(angles, axis=0) - np.min(angles, axis=0)
	rom = float(np.linalg.norm(rom_axes))
	return mean_vel, rom, peak_vel


def compute_single_axis_features(angles: np.ndarray, fps: float) -> Tuple[float, float, float]:
	# angles: (T,)
	vel = np.abs(np.diff(angles) * fps)
	mean_vel = float(np.mean(vel))
	peak_vel = float(np.max(vel))
	rom = float(np.max(angles) - np.min(angles))
	return mean_vel, rom, peak_vel


def merged_lr_features(
	x: np.ndarray,
	left_cols: List[int],
	right_cols: List[int],
	fps: float,
) -> Tuple[float, float, float]:
	if len(left_cols) == 1:
		l_mean, l_rom, l_peak = compute_single_axis_features(x[:, left_cols[0]], fps)
	else:
		l_mean, l_rom, l_peak = compute_multi_axis_features(x[:, left_cols], fps)

	if len(right_cols) == 1:
		r_mean, r_rom, r_peak = compute_single_axis_features(x[:, right_cols[0]], fps)
	else:
		r_mean, r_rom, r_peak = compute_multi_axis_features(x[:, right_cols], fps)

	return (
		float((l_mean + r_mean) / 2.0),
		float((l_rom + r_rom) / 2.0),
		float((l_peak + r_peak) / 2.0),
	)


def compute_clip_features(x: np.ndarray, fps: float = FPS) -> Dict[str, float]:
	"""Compute 27 simple features (9 joint groups x 3 stats)."""
	feats: Dict[str, float] = {}

	# root_translate: cols 0,1,2
	mean_vel, rom, peak_vel = compute_multi_axis_features(x[:, [0, 1, 2]], fps)
	feats["root_translate_mean_vel"] = mean_vel
	feats["root_translate_rom"] = rom
	feats["root_translate_peak_vel"] = peak_vel

	# root_rotate: cols 3,4,5
	mean_vel, rom, peak_vel = compute_multi_axis_features(x[:, [3, 4, 5]], fps)
	feats["root_rotate_mean_vel"] = mean_vel
	feats["root_rotate_rom"] = rom
	feats["root_rotate_peak_vel"] = peak_vel

	# hip: left 6,7,8; right 12,13,14
	mean_vel, rom, peak_vel = merged_lr_features(x, [6, 7, 8], [12, 13, 14], fps)
	feats["hip_mean_vel"] = mean_vel
	feats["hip_rom"] = rom
	feats["hip_peak_vel"] = peak_vel

	# knee: left 9; right 15
	mean_vel, rom, peak_vel = merged_lr_features(x, [9], [15], fps)
	feats["knee_mean_vel"] = mean_vel
	feats["knee_rom"] = rom
	feats["knee_peak_vel"] = peak_vel

	# ankle: left 10,11; right 16,17
	mean_vel, rom, peak_vel = merged_lr_features(x, [10, 11], [16, 17], fps)
	feats["ankle_mean_vel"] = mean_vel
	feats["ankle_rom"] = rom
	feats["ankle_peak_vel"] = peak_vel

	# waist: cols 18,19,20
	mean_vel, rom, peak_vel = compute_multi_axis_features(x[:, [18, 19, 20]], fps)
	feats["waist_mean_vel"] = mean_vel
	feats["waist_rom"] = rom
	feats["waist_peak_vel"] = peak_vel

	# shoulder: left 21,22,23; right 28,29,30
	mean_vel, rom, peak_vel = merged_lr_features(x, [21, 22, 23], [28, 29, 30], fps)
	feats["shoulder_mean_vel"] = mean_vel
	feats["shoulder_rom"] = rom
	feats["shoulder_peak_vel"] = peak_vel

	# elbow: left 24; right 31
	mean_vel, rom, peak_vel = merged_lr_features(x, [24], [31], fps)
	feats["elbow_mean_vel"] = mean_vel
	feats["elbow_rom"] = rom
	feats["elbow_peak_vel"] = peak_vel

	# wrist: left 25,26,27; right 32,33,34
	mean_vel, rom, peak_vel = merged_lr_features(x, [25, 26, 27], [32, 33, 34], fps)
	feats["wrist_mean_vel"] = mean_vel
	feats["wrist_rom"] = rom
	feats["wrist_peak_vel"] = peak_vel

	return feats


def load_clip_motion(
	row: pd.Series,
	data_root: str,
	cache_info: Optional[Dict],
	mmap_obj,
	min_seq_len: int,
) -> Tuple[Optional[np.ndarray], object]:
	path_col = "move_g1_mujoco_path"
	rel_path = str(row[path_col])

	try:
		if cache_info is not None:
			x, mmap_obj = read_g1_cached(cache_info, rel_path, mmap_obj)
			if x is None:
				x = read_g1_csv(os.path.join(data_root, rel_path))
		else:
			x = read_g1_csv(os.path.join(data_root, rel_path))
	except Exception:
		return None, mmap_obj

	if x is None:
		return None, mmap_obj

	x = np.asarray(x, dtype=np.float32)
	if x.ndim != 2 or x.shape[0] < min_seq_len or x.shape[1] < N_G1_CHANNELS:
		return None, mmap_obj

	x = np.nan_to_num(x[:, :N_G1_CHANNELS], nan=0.0, posinf=0.0, neginf=0.0)
	return x, mmap_obj


def load_all_manifests(splits_dir: str) -> pd.DataFrame:
	parts = []
	for split in ["train", "val", "test"]:
		fp = os.path.join(splits_dir, f"{split}_manifest.csv")
		if not os.path.exists(fp):
			raise FileNotFoundError(f"Missing manifest: {fp}")
		sdf = pd.read_csv(fp)
		sdf["split"] = split
		parts.append(sdf)
	return pd.concat(parts, ignore_index=True)


def filter_tasks(df: pd.DataFrame, min_demos: int, min_actors: int) -> pd.DataFrame:
	if "content_type_of_movement" not in df.columns:
		raise KeyError("Manifest is missing content_type_of_movement column.")
	if "actor_uid" not in df.columns:
		raise KeyError("Manifest is missing actor_uid column.")

	dff = df[df["content_type_of_movement"].notna()].copy()
	stats_df = (
		dff.groupby("content_type_of_movement")
		.agg(num_clips=("content_type_of_movement", "size"), num_actors=("actor_uid", "nunique"))
		.reset_index()
		.sort_values(["num_clips", "num_actors"], ascending=[False, False])
	)

	keep = stats_df[
		(stats_df["num_clips"] >= min_demos) & (stats_df["num_actors"] >= min_actors)
	].copy()

	print("\nTask summary (after thresholds):")
	if keep.empty:
		print("  No tasks passed filtering.")
	else:
		for _, row in keep.iterrows():
			task = str(row["content_type_of_movement"])
			print(f"  {task:45s} clips={int(row['num_clips']):5d} actors={int(row['num_actors']):4d}")

	keep_tasks = set(keep["content_type_of_movement"].tolist())
	return dff[dff["content_type_of_movement"].isin(keep_tasks)].copy()


def extract_actor_task_features(
	df: pd.DataFrame,
	data_root: str,
	cache_info: Optional[Dict],
	min_seq_len: int,
	progress_every: int,
) -> pd.DataFrame:
	required_cols = [
		"move_g1_mujoco_path",
		"actor_uid",
		"actor_gender",
		"actor_age_yr",
		"actor_height_cm",
		"actor_weight_kg",
		"content_type_of_movement",
	]
	for c in required_cols:
		if c not in df.columns:
			raise KeyError(f"Manifest is missing required column: {c}")

	valid = df["move_g1_mujoco_path"].notna()
	for c in [
		"actor_uid",
		"actor_gender",
		"actor_age_yr",
		"actor_height_cm",
		"actor_weight_kg",
		"content_type_of_movement",
	]:
		valid &= df[c].notna()
	dff = df[valid].reset_index(drop=True)

	rows = []
	mmap_obj = None
	skipped = 0
	t0 = time.time()

	for i in range(len(dff)):
		row = dff.iloc[i]
		x, mmap_obj = load_clip_motion(row, data_root, cache_info, mmap_obj, min_seq_len)
		if x is None:
			skipped += 1
			continue

		feats = compute_clip_features(x, fps=FPS)
		feats["actor_uid"] = row["actor_uid"]
		feats["actor_gender"] = str(row["actor_gender"])
		feats["actor_age_yr"] = float(row["actor_age_yr"])
		feats["actor_height_cm"] = float(row["actor_height_cm"])
		feats["actor_weight_kg"] = float(row["actor_weight_kg"])
		feats["content_type_of_movement"] = str(row["content_type_of_movement"])
		rows.append(feats)

		if progress_every > 0 and (i + 1) % progress_every == 0:
			elapsed = max(1e-6, time.time() - t0)
			rate = (i + 1) / elapsed
			print(f"  [{i+1:>6d}/{len(dff)}] {rate:6.1f} clips/s  skipped={skipped}")

	print(f"Processed clips: {len(rows):,}, skipped: {skipped:,}, time: {time.time() - t0:.1f}s")
	if not rows:
		return pd.DataFrame()

	clip_df = pd.DataFrame(rows)

	id_cols = [
		"actor_uid",
		"content_type_of_movement",
		"actor_gender",
		"actor_age_yr",
		"actor_height_cm",
		"actor_weight_kg",
	]
	feat_cols = [c for c in clip_df.columns if c not in id_cols]

	agg_feats = (
		clip_df.groupby(["content_type_of_movement", "actor_uid"], as_index=False)[feat_cols]
		.mean()
	)

	attrs = (
		clip_df.groupby(["content_type_of_movement", "actor_uid"], as_index=False)[
			["actor_gender", "actor_age_yr", "actor_height_cm", "actor_weight_kg"]
		]
		.first()
	)

	actor_task_df = agg_feats.merge(
		attrs,
		on=["content_type_of_movement", "actor_uid"],
		how="left",
	)
	return actor_task_df


def correlation_with_target(feature_vals: np.ndarray, target_vals: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
	mask = np.isfinite(feature_vals) & np.isfinite(target_vals)
	if mask.sum() < 5:
		return None, None
	f = feature_vals[mask]
	t = target_vals[mask]
	if np.std(f) < 1e-12 or np.std(t) < 1e-12:
		return None, None
	r, p = sp_stats.pearsonr(f, t)
	if not np.isfinite(r) or not np.isfinite(p):
		return None, None
	return float(r), float(p)


def partial_correlation_with_target(
	feature_vals: np.ndarray,
	target_vals: np.ndarray,
	control_vals: np.ndarray,
	min_n: int = 30,
) -> Tuple[Optional[float], Optional[float]]:
	"""Partial Pearson r between feature and target after residualizing both on control_vals.

	control_vals: shape (N, k) — the variables to partial out.
	Returns (partial_r, partial_p), or (None, None) when data are insufficient.
	"""
	mask = np.isfinite(feature_vals) & np.isfinite(target_vals)
	for c in range(control_vals.shape[1]):
		mask &= np.isfinite(control_vals[:, c])
	if mask.sum() < min_n:
		return None, None

	f = feature_vals[mask]
	t = target_vals[mask]
	C = control_vals[mask]

	if np.std(f) < 1e-12 or np.std(t) < 1e-12:
		return None, None

	# Design matrix: intercept + controls
	X = np.column_stack([np.ones(C.shape[0]), C])

	coef_f, _, _, _ = np.linalg.lstsq(X, f, rcond=None)
	resid_f = f - X @ coef_f

	coef_t, _, _, _ = np.linalg.lstsq(X, t, rcond=None)
	resid_t = t - X @ coef_t

	if np.std(resid_f) < 1e-12 or np.std(resid_t) < 1e-12:
		return None, None

	r, p = sp_stats.pearsonr(resid_f, resid_t)
	if not np.isfinite(r) or not np.isfinite(p):
		return None, None
	return float(r), float(p)


def analyze_top_features_per_task_attr(actor_task_df: pd.DataFrame) -> Dict:
	attr_cols = ["actor_age_yr", "actor_height_cm", "actor_weight_kg", "gender_numeric"]
	non_feat = {"actor_uid", "actor_gender", "content_type_of_movement", *attr_cols}
	feat_cols = [c for c in actor_task_df.columns if c not in non_feat]

	out = {}
	for task, tdf in actor_task_df.groupby("content_type_of_movement"):
		out[str(task)] = {}
		for attr in attr_cols:
			scores = []
			tv = pd.to_numeric(tdf[attr], errors="coerce").to_numpy(dtype=float)
			for feat in feat_cols:
				fv = pd.to_numeric(tdf[feat], errors="coerce").to_numpy(dtype=float)
				r, p = correlation_with_target(fv, tv)
				if r is None:
					continue

				item = {"feature": feat, "r": r, "p": p, "abs_r": abs(r)}
				if attr == "gender_numeric":
					auc = binary_auc_from_scores(tv, fv)
					item["auc"] = auc
				scores.append(item)

			scores = sorted(scores, key=lambda d: abs(d["r"]), reverse=True)[:5]
			# remove helper field before saving
			for s in scores:
				s.pop("abs_r", None)
			out[str(task)][attr] = scores
	return out


def analyze_top_features_per_task_attr_partial(
	actor_task_df: pd.DataFrame,
	min_n_partial: int = 30,
) -> Dict:
	"""Compute partial correlations between features and each target attribute.

	Controls for each target attribute are the other three attributes.
	Results ranked by |partial_r| when available (N >= min_n_partial), else |r|.
	Stored items contain only partial_r/partial_p (and auc for gender).
	"""
	attr_cols = ["actor_age_yr", "actor_height_cm", "actor_weight_kg", "gender_numeric"]
	non_feat = {"actor_uid", "actor_gender", "content_type_of_movement", *attr_cols}
	feat_cols = [c for c in actor_task_df.columns if c not in non_feat]

	out = {}
	for task, tdf in actor_task_df.groupby("content_type_of_movement"):
		out[str(task)] = {}
		for attr in attr_cols:
			scores = []
			tv = pd.to_numeric(tdf[attr], errors="coerce").to_numpy(dtype=float)
			control_attrs = [a for a in attr_cols if a != attr]
			ctrl_matrix = np.column_stack([
				pd.to_numeric(tdf[c], errors="coerce").to_numpy(dtype=float)
				for c in control_attrs
			])

			for feat in feat_cols:
				fv = pd.to_numeric(tdf[feat], errors="coerce").to_numpy(dtype=float)
				raw_r, _ = correlation_with_target(fv, tv)
				if raw_r is None:
					continue
				partial_r, partial_p = partial_correlation_with_target(
					fv, tv, ctrl_matrix, min_n=min_n_partial
				)

				sort_key = abs(partial_r) if partial_r is not None else abs(raw_r)
				item = {
					"feature": feat,
					"partial_r": partial_r,
					"partial_p": partial_p,
					"_sort_key": sort_key,
				}
				if attr == "gender_numeric":
					item["auc"] = binary_auc_from_scores(tv, fv)
				scores.append(item)

			scores = sorted(scores, key=lambda d: d["_sort_key"], reverse=True)[:5]
			for s in scores:
				s.pop("_sort_key", None)
			out[str(task)][attr] = scores
	return out


def analyze_dominant_attr_per_task_feature(actor_task_df: pd.DataFrame) -> Dict:
	attr_cols = ["actor_age_yr", "actor_height_cm", "actor_weight_kg", "gender_numeric"]
	non_feat = {"actor_uid", "actor_gender", "content_type_of_movement", *attr_cols}
	feat_cols = [c for c in actor_task_df.columns if c not in non_feat]

	out = {}
	for task, tdf in actor_task_df.groupby("content_type_of_movement"):
		out[str(task)] = {}
		for feat in feat_cols:
			fv = pd.to_numeric(tdf[feat], errors="coerce").to_numpy(dtype=float)
			best = None
			for attr in attr_cols:
				tv = pd.to_numeric(tdf[attr], errors="coerce").to_numpy(dtype=float)
				r, p = correlation_with_target(fv, tv)
				if r is None:
					continue

				item = {
					"attribute": attr,
					"r": r,
					"p": p,
					"abs_r": abs(r),
				}
				if attr == "gender_numeric":
					item["auc"] = binary_auc_from_scores(tv, fv)

				if best is None or item["abs_r"] > best["abs_r"]:
					best = item

			if best is not None:
				best.pop("abs_r", None)
				out[str(task)][feat] = best
	return out


def analyze_dominant_attr_per_task_feature_partial(
	actor_task_df: pd.DataFrame,
	min_n_partial: int = 30,
) -> Dict:
	"""Find the dominant attribute per feature using partial correlations.

	Dominant attribute is chosen by |partial_r| when available (N >= min_n_partial), else |r|.
	Stored items contain only partial_r/partial_p (and auc for gender).
	"""
	attr_cols = ["actor_age_yr", "actor_height_cm", "actor_weight_kg", "gender_numeric"]
	non_feat = {"actor_uid", "actor_gender", "content_type_of_movement", *attr_cols}
	feat_cols = [c for c in actor_task_df.columns if c not in non_feat]

	out = {}
	for task, tdf in actor_task_df.groupby("content_type_of_movement"):
		out[str(task)] = {}
		for feat in feat_cols:
			fv = pd.to_numeric(tdf[feat], errors="coerce").to_numpy(dtype=float)
			best = None
			for attr in attr_cols:
				tv = pd.to_numeric(tdf[attr], errors="coerce").to_numpy(dtype=float)
				control_attrs = [a for a in attr_cols if a != attr]
				ctrl_matrix = np.column_stack([
					pd.to_numeric(tdf[c], errors="coerce").to_numpy(dtype=float)
					for c in control_attrs
				])

				raw_r, _ = correlation_with_target(fv, tv)
				if raw_r is None:
					continue
				partial_r, partial_p = partial_correlation_with_target(
					fv, tv, ctrl_matrix, min_n=min_n_partial
				)

				sort_key = abs(partial_r) if partial_r is not None else abs(raw_r)
				item = {
					"attribute": attr,
					"partial_r": partial_r,
					"partial_p": partial_p,
					"_sort_key": sort_key,
				}
				if attr == "gender_numeric":
					item["auc"] = binary_auc_from_scores(tv, fv)

				if best is None or item["_sort_key"] > best["_sort_key"]:
					best = item

			if best is not None:
				best.pop("_sort_key", None)
				out[str(task)][feat] = best
	return out


def print_top_features_summary(results: Dict) -> None:
	print("\n=== Top 5 Features Per Task/Attribute ===")
	for task, task_res in results.items():
		print(f"\nTask: {task}")
		for attr, feats in task_res.items():
			print(f"  Attribute: {attr}")
			if not feats:
				print("    (no valid correlations)")
				continue
			for f in feats:
				auc_txt = ""
				if "auc" in f and f["auc"] is not None:
					auc_txt = f", auc={f['auc']:.4f}"
				print(f"    {f['feature']:<28s} r={f['r']:+.4f} p={f['p']:.3g}{auc_txt}")


def print_dominant_summary(results: Dict) -> None:
	print("\n=== Dominant Attribute Per Task/Feature ===")
	for task, task_res in results.items():
		print(f"\nTask: {task}")
		shown = 0
		for feat, info in task_res.items():
			auc_txt = ""
			if "auc" in info and info["auc"] is not None:
				auc_txt = f", auc={info['auc']:.4f}"
			print(
				f"  {feat:<28s} -> {info['attribute']:<16s} "
				f"r={info['r']:+.4f} p={info['p']:.3g}{auc_txt}"
			)
			shown += 1
			if shown >= 10:
				print(f"  ... ({len(task_res)} total features)")
				break


def print_top_features_partial_summary(results: Dict) -> None:
	print("\n=== Top 5 Features Per Task/Attribute (Partial Correlations) ===")
	for task, task_res in results.items():
		print(f"\nTask: {task}")
		for attr, feats in task_res.items():
			print(f"  Attribute: {attr}")
			if not feats:
				print("    (no valid correlations)")
				continue
			for f in feats:
				auc_txt = ""
				if "auc" in f and f["auc"] is not None:
					auc_txt = f", auc={f['auc']:.4f}"
				if f.get("partial_r") is not None:
					pr_txt = f"partial_r={f['partial_r']:+.4f} partial_p={f['partial_p']:.3g}"
				else:
					pr_txt = "partial_r=n/a (low-N)"
				print(f"    {f['feature']:<28s} {pr_txt}{auc_txt}")


def print_dominant_partial_summary(results: Dict) -> None:
	print("\n=== Dominant Attribute Per Task/Feature (Partial Correlations) ===")
	for task, task_res in results.items():
		print(f"\nTask: {task}")
		shown = 0
		for feat, info in task_res.items():
			auc_txt = ""
			if "auc" in info and info["auc"] is not None:
				auc_txt = f", auc={info['auc']:.4f}"
			if info.get("partial_r") is not None:
				pr_txt = f"partial_r={info['partial_r']:+.4f} partial_p={info['partial_p']:.3g}"
			else:
				pr_txt = "partial_r=n/a (low-N)"
			print(
				f"  {feat:<28s} -> {info['attribute']:<16s} {pr_txt}{auc_txt}"
			)
			shown += 1
			if shown >= 10:
				print(f"  ... ({len(task_res)} total features)")
				break


def _format_topk_items(items: List[Dict]) -> str:
	if not items:
		return ""
	parts = []
	for item in items[:3]:
		feat = str(item.get("feature", ""))
		r = item.get("r", None)
		p = item.get("p", None)
		auc = item.get("auc", None)
		if r is None or p is None:
			continue
		piece = f"{feat} (r={float(r):+.3f}, p={float(p):.3g}"
		if auc is not None:
			piece += f", auc={float(auc):.3f}"
		piece += ")"
		parts.append(piece)
	return " | ".join(parts)


def _format_topk_items_partial(items: List[Dict]) -> str:
	"""Format top-k items showing only partial_r/partial_p."""
	if not items:
		return ""
	parts = []
	for item in items[:3]:
		feat = str(item.get("feature", ""))
		partial_r = item.get("partial_r", None)
		partial_p = item.get("partial_p", None)
		auc = item.get("auc", None)
		if partial_r is None:
			piece = f"{feat} (pr=n/a)"
		else:
			piece = f"{feat} (pr={float(partial_r):+.3f}, p={float(partial_p):.3g}"
			if auc is not None:
				piece += f", auc={float(auc):.3f}"
			piece += ")"
		parts.append(piece)
	return " | ".join(parts)


def save_task_attr_top3_summary_csv(
	actor_task_df: pd.DataFrame,
	filtered_df: pd.DataFrame,
	top_features_per_task_attr: Dict,
	output_csv: str,
) -> None:
	clip_counts = (
		filtered_df.groupby("content_type_of_movement")
		.size()
		.rename("num_clips")
		.reset_index()
	)
	actor_counts = (
		actor_task_df.groupby("content_type_of_movement")["actor_uid"]
		.nunique()
		.rename("num_actors")
		.reset_index()
	)

	summary = clip_counts.merge(actor_counts, on="content_type_of_movement", how="outer")
	summary["num_clips"] = summary["num_clips"].fillna(0).astype(int)
	summary["num_actors"] = summary["num_actors"].fillna(0).astype(int)

	rows = []
	for _, row in summary.iterrows():
		task = str(row["content_type_of_movement"])
		task_block = top_features_per_task_attr.get(task, {})
		rows.append(
			{
				"content_type_of_movement": task,
				"num_clips": int(row["num_clips"]),
				"num_actors": int(row["num_actors"]),
				"top3_age": _format_topk_items(task_block.get("actor_age_yr", [])),
				"top3_height": _format_topk_items(task_block.get("actor_height_cm", [])),
				"top3_weight": _format_topk_items(task_block.get("actor_weight_kg", [])),
				"top3_gender": _format_topk_items(task_block.get("gender_numeric", [])),
			}
		)

	out_df = pd.DataFrame(rows).sort_values(
		by=["num_actors", "num_clips", "content_type_of_movement"],
		ascending=[False, False, True],
	)
	out_df.to_csv(output_csv, index=False)


def save_task_attr_top3_summary_partial_csv(
	actor_task_df: pd.DataFrame,
	filtered_df: pd.DataFrame,
	top_features_partial: Dict,
	output_csv: str,
) -> None:
	"""Save a per-task summary CSV showing top-3 features with both r and partial_r."""
	clip_counts = (
		filtered_df.groupby("content_type_of_movement")
		.size()
		.rename("num_clips")
		.reset_index()
	)
	actor_counts = (
		actor_task_df.groupby("content_type_of_movement")["actor_uid"]
		.nunique()
		.rename("num_actors")
		.reset_index()
	)

	summary = clip_counts.merge(actor_counts, on="content_type_of_movement", how="outer")
	summary["num_clips"] = summary["num_clips"].fillna(0).astype(int)
	summary["num_actors"] = summary["num_actors"].fillna(0).astype(int)

	rows = []
	for _, row in summary.iterrows():
		task = str(row["content_type_of_movement"])
		task_block = top_features_partial.get(task, {})
		rows.append(
			{
				"content_type_of_movement": task,
				"num_clips": int(row["num_clips"]),
				"num_actors": int(row["num_actors"]),
				"top3_age": _format_topk_items_partial(task_block.get("actor_age_yr", [])),
				"top3_height": _format_topk_items_partial(task_block.get("actor_height_cm", [])),
				"top3_weight": _format_topk_items_partial(task_block.get("actor_weight_kg", [])),
				"top3_gender": _format_topk_items_partial(task_block.get("gender_numeric", [])),
			}
		)

	out_df = pd.DataFrame(rows).sort_values(
		by=["num_actors", "num_clips", "content_type_of_movement"],
		ascending=[False, False, True],
	)
	out_df.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Simple per-task interpretability analysis for G1")
	parser.add_argument("--data_root", type=str, default="..")
	parser.add_argument("--splits_dir", type=str, default="../artifacts/splits")
	parser.add_argument("--g1_cache_dir", type=str, default="../artifacts/cache/g1_motions")
	parser.add_argument("--output_dir", type=str, default="./simple_interpretability_results")
	parser.add_argument("--min_seq_len", type=int, default=30)
	parser.add_argument("--min_demos", type=int, default=10)
	parser.add_argument("--min_actors", type=int, default=5)
	parser.add_argument(
		"--min_actors_partial",
		type=int,
		default=30,
		help="Minimum actors per task required to compute partial correlations (default: 30).",
	)
	parser.add_argument("--progress_every", type=int, default=500)
	parser.add_argument("--preset", type=str, default="full", choices=["full", "fast"])
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.preset == "fast":
		args.progress_every = 200

	os.makedirs(args.output_dir, exist_ok=True)

	print("Loading manifests...")
	manifest_df = load_all_manifests(args.splits_dir)
	print(f"Total rows from all splits: {len(manifest_df):,}")

	print("Applying task filters...")
	filtered_df = filter_tasks(manifest_df, min_demos=args.min_demos, min_actors=args.min_actors)
	print(f"Rows after task filtering: {len(filtered_df):,}")
	if filtered_df.empty:
		print("No valid rows to process after filtering. Exiting.")
		return

	print("Loading G1 cache metadata...")
	cache_info = load_g1_cache(args.g1_cache_dir)
	if cache_info is None:
		print("Cache not found or incomplete. Falling back to CSV reads.")
	else:
		print(
			"Cache loaded:",
			f"frames={cache_info['total_frames']:,}",
			f"channels={cache_info['num_channels']}",
		)

	print("Extracting clip features and aggregating per (task, actor)...")
	actor_task_df = extract_actor_task_features(
		filtered_df,
		data_root=args.data_root,
		cache_info=cache_info,
		min_seq_len=args.min_seq_len,
		progress_every=args.progress_every,
	)

	if actor_task_df.empty:
		print("No actor-task features extracted. Exiting.")
		return

	# Gender numeric mapping (sorted alphabetical values).
	genders = sorted(actor_task_df["actor_gender"].dropna().unique().tolist())
	gender_map = {g: i for i, g in enumerate(genders)}
	actor_task_df["gender_numeric"] = actor_task_df["actor_gender"].map(gender_map)
	print("Gender mapping:", gender_map)

	actor_task_csv = os.path.join(args.output_dir, "actor_task_features.csv")
	actor_task_df.to_csv(actor_task_csv, index=False)
	print(f"Saved actor-task feature matrix: {actor_task_csv}")

	top5 = analyze_top_features_per_task_attr(actor_task_df)
	dominant = analyze_dominant_attr_per_task_feature(actor_task_df)

	top5_json = os.path.join(args.output_dir, "top_features_per_task_attr.json")
	dom_json = os.path.join(args.output_dir, "dominant_attr_per_task_feature.json")
	with open(top5_json, "w", encoding="utf-8") as f:
		json.dump(top5, f, indent=2)
	with open(dom_json, "w", encoding="utf-8") as f:
		json.dump(dominant, f, indent=2)

	top3_csv = os.path.join(args.output_dir, "task_attr_top3_summary.csv")
	save_task_attr_top3_summary_csv(
		actor_task_df=actor_task_df,
		filtered_df=filtered_df,
		top_features_per_task_attr=top5,
		output_csv=top3_csv,
	)

	print_top_features_summary(top5)
	print_dominant_summary(dominant)

	# --- Partial correlations (unique_corr/) ---
	unique_corr_dir = os.path.join(args.output_dir, "unique_corr")
	os.makedirs(unique_corr_dir, exist_ok=True)
	print(f"\nComputing partial correlations (min_actors_partial={args.min_actors_partial})...")

	top5_partial = analyze_top_features_per_task_attr_partial(
		actor_task_df, min_n_partial=args.min_actors_partial
	)
	dominant_partial = analyze_dominant_attr_per_task_feature_partial(
		actor_task_df, min_n_partial=args.min_actors_partial
	)

	top5_partial_json = os.path.join(unique_corr_dir, "top_features_per_task_attr_partial.json")
	dom_partial_json = os.path.join(unique_corr_dir, "dominant_attr_per_task_feature_partial.json")
	with open(top5_partial_json, "w", encoding="utf-8") as f:
		json.dump(top5_partial, f, indent=2)
	with open(dom_partial_json, "w", encoding="utf-8") as f:
		json.dump(dominant_partial, f, indent=2)

	top3_partial_csv = os.path.join(unique_corr_dir, "task_attr_top3_summary_partial.csv")
	save_task_attr_top3_summary_partial_csv(
		actor_task_df=actor_task_df,
		filtered_df=filtered_df,
		top_features_partial=top5_partial,
		output_csv=top3_partial_csv,
	)

	print_top_features_partial_summary(top5_partial)
	print_dominant_partial_summary(dominant_partial)

	print("\nSaved outputs:")
	print(f"  {actor_task_csv}")
	print(f"  {top5_json}")
	print(f"  {dom_json}")
	print(f"  {top3_csv}")
	print(f"  {top5_partial_json}")
	print(f"  {dom_partial_json}")
	print(f"  {top3_partial_csv}")


if __name__ == "__main__":
	main()
