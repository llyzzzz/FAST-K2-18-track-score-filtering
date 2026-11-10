#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
add_beam_track_vote_score_fast.py

功能
----
对候选 CSV 中的每一行候选，计算一个“沿 BLISS 漂移线的加权投票分数”。

前提
----
你已经提前生成了一个阈值 H5：
    OFF_UPPER = OFF_mean + k * OFF_std
例如：
    K2-18b_OFF_UPPER_K2.h5

本脚本只读取：
    1) M01.h5
    2) OFF_UPPER.h5

核心逻辑
--------
1. 对每个候选，取出 M01 和 OFF_UPPER 的对应频率窗口；
2. 逐像素比较：
       vote = (M01 > OFF_UPPER)
   超过阈值记 1 票，否则 0 票；
3. 根据像素到 BLISS 漂移线的距离，给一个高斯权重：
       离漂移线越近，权重越大；
4. 最终得分：
       beam_track_vote_score = sum(weight * vote) / sum(weight)

同时输出一个连续性特征：
       beam_track_continuity
   表示有多少个时间 bin 在线核心区域内至少出现过一次投票。

为了加快速度：
- 先按粗频率 coarse group 分组；
- 每个组只读一次较大的 M01 / OFF_UPPER 块；
- 组内候选直接从内存切片。

输入
----
CSV 至少需要这些列：
    start_freq_MHz
    end_freq_MHz
如果有下面这些列会自动使用：
    bandwidth_Hz
    drift_rate_Hz_per_sec / DriftRate
    SNR

输出新增列
----------
beam_freq_low_MHz
beam_freq_high_MHz
beam_track_vote_score
beam_track_continuity
beam_status
beam_error
"""

import os
import math
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    import hdf5plugin  # noqa: F401
except Exception:
    pass

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================
# 工具函数
# ============================================================

def _attr_to_float(x):
    if isinstance(x, bytes):
        x = x.decode()

    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
        else:
            x = x.flat[0]

    return float(x)


def get_header_value(h5, dset, key):
    if key in dset.attrs:
        return _attr_to_float(dset.attrs[key])
    if key in h5.attrs:
        return _attr_to_float(h5.attrs[key])
    raise KeyError(f"找不到 header 字段: {key}")


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# H5 Reader
# ============================================================

class FastH5Reader:
    """
    快速 H5 读取器：
    - 默认频率轴为最后一维
    - 常见 shape:
        (time, pol, freq)
        (time, freq)
    """

    def __init__(self, path, dataset="data"):
        self.path = path
        self.h5 = h5py.File(path, "r")

        if dataset not in self.h5:
            raise KeyError(f"{path} 中找不到 dataset: {dataset}")

        self.dataset_name = dataset
        self.data = self.h5[dataset]
        self.shape = self.data.shape
        self.ndim = self.data.ndim

        if self.ndim < 2:
            raise ValueError(f"{path} 的 data 维度太低，不支持: shape={self.shape}")

        self.nfreq = self.shape[-1]

        self.fch1 = get_header_value(self.h5, self.data, "fch1")
        self.foff = get_header_value(self.h5, self.data, "foff")

        # 自动兼容 Hz / MHz
        if abs(self.fch1) > 1e5:
            self.fch1 /= 1e6
        if abs(self.foff) > 1.0:
            self.foff /= 1e6

        if self.foff == 0:
            raise ValueError(f"{path} 的 foff=0，无法建立频率坐标")

        self.freq_first = self.fch1
        self.freq_last = self.fch1 + (self.nfreq - 1) * self.foff
        self.freq_min = min(self.freq_first, self.freq_last)
        self.freq_max = max(self.freq_first, self.freq_last)
        self.chan_width_hz = abs(self.foff) * 1e6

        # 时间维长度
        self.ntime = self.shape[0]

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass

    def _make_slice(self, i0, i1):
        slc = [slice(None)] * self.ndim
        slc[-1] = slice(i0, i1)
        return tuple(slc)

    def _read_raw_slice(self, i0, i1):
        slc = self._make_slice(i0, i1)
        arr = self.data[slc]

        # 常见情况：
        # (time, pol, freq) -> 取 pol=0
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        elif arr.ndim == 2:
            pass
        else:
            # 更高维时尽量压缩到 [time, freq]
            arr = np.asarray(arr)
            while arr.ndim > 2:
                arr = arr[:, 0, ...]

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim != 2:
            raise ValueError(f"读取后数组不是 2D: shape={arr.shape}, path={self.path}")

        return arr

    def freq_to_index_bounds(self, f_low_mhz, f_high_mhz):
        if f_high_mhz <= f_low_mhz:
            raise ValueError(f"非法频率范围: {f_low_mhz} - {f_high_mhz} MHz")

        if f_high_mhz < self.freq_min or f_low_mhz > self.freq_max:
            raise ValueError(
                f"请求频率超出文件范围: request={f_low_mhz}-{f_high_mhz} MHz, "
                f"file={self.freq_min}-{self.freq_max} MHz, path={self.path}"
            )

        ia = (f_low_mhz - self.fch1) / self.foff
        ib = (f_high_mhz - self.fch1) / self.foff

        i0 = int(np.floor(min(ia, ib)))
        i1 = int(np.ceil(max(ia, ib))) + 1

        i0 = max(0, i0)
        i1 = min(self.nfreq, i1)

        if i1 <= i0:
            raise ValueError(f"频率窗口为空: {f_low_mhz}-{f_high_mhz} MHz, {self.path}")

        return i0, i1

    def read_freq_chunk(self, f_low_mhz, f_high_mhz):
        i0, i1 = self.freq_to_index_bounds(f_low_mhz, f_high_mhz)
        arr = self._read_raw_slice(i0, i1)
        freq_axis = self.fch1 + np.arange(i0, i1, dtype=np.float64) * self.foff
        return arr, freq_axis, i0, i1


# ============================================================
# 漂移线与打分
# ============================================================

def build_line_freqs(start_mhz, end_mhz, ntime):
    """
    沿时间方向线性插值漂移线
    """
    if ntime <= 1:
        return np.array([0.5 * (start_mhz + end_mhz)], dtype=np.float64)
    return np.linspace(start_mhz, end_mhz, ntime, dtype=np.float64)


def gaussian_weight_from_distance(distance_hz, sigma_hz, cutoff_sigma=3.0):
    """
    高斯权重；超过 cutoff_sigma * sigma_hz 后直接置 0，提高速度。
    """
    w = np.exp(-0.5 * (distance_hz / sigma_hz) ** 2, dtype=np.float64)
    if cutoff_sigma is not None and cutoff_sigma > 0:
        w = np.where(distance_hz <= cutoff_sigma * sigma_hz, w, 0.0)
    return w


def score_one_candidate(
    m01_sub,
    upper_sub,
    freq_axis_mhz,
    start_freq_mhz,
    end_freq_mhz,
    sigma_hz,
    cutoff_sigma=3.0,
    continuity_core_hz=None,
):
    """
    计算一个候选的打分。

    参数
    ----
    m01_sub       : shape (ntime, nfreq_sub)
    upper_sub     : shape (ntime, nfreq_sub)
    freq_axis_mhz : shape (nfreq_sub,)
    start/end     : BLISS 候选线起止频率
    sigma_hz      : 权重宽度
    cutoff_sigma  : 超过几倍 sigma 的像素权重直接为 0
    continuity_core_hz : 连续性统计用的线核心半宽

    返回
    ----
    vote_score, continuity
    """
    ntime, nfreq = m01_sub.shape
    line_freqs = build_line_freqs(start_freq_mhz, end_freq_mhz, ntime)  # (ntime,)

    # 距离矩阵，单位 Hz
    # dist[t, f] = abs(freq_axis[f] - line_freqs[t]) * 1e6
    dist_hz = np.abs(freq_axis_mhz[None, :] - line_freqs[:, None]) * 1e6

    weights = gaussian_weight_from_distance(
        distance_hz=dist_hz,
        sigma_hz=sigma_hz,
        cutoff_sigma=cutoff_sigma,
    )

    weight_sum = np.sum(weights)
    if weight_sum <= 0:
        return np.nan, np.nan

    vote = (m01_sub > upper_sub).astype(np.float32)

    vote_score = float(np.sum(weights * vote) / weight_sum)

    # 连续性：每个时间 bin 在线核心区域是否至少出现一次投票
    if continuity_core_hz is None:
        continuity_core_hz = sigma_hz

    core_mask = dist_hz <= continuity_core_hz
    if np.any(core_mask):
        row_has_vote = np.any((vote > 0) & core_mask, axis=1)
        continuity = float(np.mean(row_has_vote))
    else:
        continuity = np.nan

    return vote_score, continuity


# ============================================================
# 主流程
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fast beam track weighted-vote score using M01 and OFF_UPPER H5."
    )

    p.add_argument("--csv", required=True, help="输入候选 CSV")
    p.add_argument("--m01", required=True, help="M01 H5 文件")
    p.add_argument("--upper", required=True, help="OFF_UPPER H5 文件，即 OFF_mean + 2*OFF_std")
    p.add_argument("--out", required=True, help="输出 CSV")

    p.add_argument("--dataset", default="data", help="H5 中数据集名称，默认 data")

    p.add_argument(
        "--coarse-mhz",
        type=float,
        default=1.0,
        help="按多大 MHz 进行粗分组共享读取，默认 1.0 MHz"
    )

    p.add_argument(
        "--window-pad-hz",
        type=float,
        default=100.0,
        help="每个候选窗口在频率两侧额外扩展的宽度，默认 100 Hz"
    )

    p.add_argument(
        "--sigma-hz",
        type=float,
        default=15.0,
        help="高斯权重 sigma，单位 Hz，默认 15"
    )

    p.add_argument(
        "--cutoff-sigma",
        type=float,
        default=3.0,
        help="超过几倍 sigma 后权重直接置 0，默认 3"
    )

    p.add_argument(
        "--continuity-core-hz",
        type=float,
        default=15.0,
        help="连续性统计时的线核心半宽，默认 15 Hz"
    )

    p.add_argument(
        "--adaptive-sigma",
        action="store_true",
        help="如果候选表中存在 bandwidth_Hz，则 sigma_hz = max(--sigma-hz, bandwidth_Hz / 2)"
    )

    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="只处理前 N 行，用于测试"
    )

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 100)
    print("Fast beam track weighted-vote scoring")
    print("CSV   :", args.csv)
    print("M01   :", args.m01)
    print("UPPER :", args.upper)
    print("OUT   :", args.out)
    print("coarse_mhz      :", args.coarse_mhz)
    print("window_pad_hz   :", args.window_pad_hz)
    print("sigma_hz        :", args.sigma_hz)
    print("cutoff_sigma    :", args.cutoff_sigma)
    print("continuity_core :", args.continuity_core_hz)
    print("adaptive_sigma  :", args.adaptive_sigma)
    print("=" * 100)

    df = pd.read_csv(args.csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    need_cols = ["start_freq_MHz", "end_freq_MHz"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"缺少必要列: {c}")

    bw_col = find_column(df, ["bandwidth_Hz"])
    drift_col = find_column(df, ["drift_rate_Hz_per_sec", "DriftRate"])

    # 中心频率和粗分组
    df["__freq_center"] = 0.5 * (df["start_freq_MHz"].astype(float) + df["end_freq_MHz"].astype(float))
    df["__coarse_group"] = np.floor(df["__freq_center"] / args.coarse_mhz) * args.coarse_mhz

    m01_reader = FastH5Reader(args.m01, dataset=args.dataset)
    upper_reader = FastH5Reader(args.upper, dataset=args.dataset)

    # 基本一致性检查
    if m01_reader.shape != upper_reader.shape:
        m01_reader.close()
        upper_reader.close()
        raise ValueError(f"M01 和 UPPER 的 shape 不一致: {m01_reader.shape} vs {upper_reader.shape}")

    if abs(m01_reader.fch1 - upper_reader.fch1) > 1e-12 or abs(m01_reader.foff - upper_reader.foff) > 1e-18:
        m01_reader.close()
        upper_reader.close()
        raise ValueError("M01 和 UPPER 的频率轴不一致（fch1 / foff 不同）")

    results = []

    group_iter = list(df.groupby("__coarse_group", sort=True))
    if tqdm is not None:
        group_iter = tqdm(group_iter, total=len(group_iter), desc="coarse groups", ncols=100)

    try:
        for coarse_val, g in group_iter:
            g = g.copy()

            # 该 coarse group 一次性读取的大频率范围
            group_low = min(g["start_freq_MHz"].min(), g["end_freq_MHz"].min()) - args.window_pad_hz / 1e6
            group_high = max(g["start_freq_MHz"].max(), g["end_freq_MHz"].max()) + args.window_pad_hz / 1e6

            # 读 chunk
            m01_chunk, freq_axis, gi0, gi1 = m01_reader.read_freq_chunk(group_low, group_high)
            upper_chunk, freq_axis2, ui0, ui1 = upper_reader.read_freq_chunk(group_low, group_high)

            if m01_chunk.shape != upper_chunk.shape:
                raise ValueError(f"组内读取后 shape 不一致: {m01_chunk.shape} vs {upper_chunk.shape}")

            # freq_axis 理论上应一致
            if len(freq_axis) != len(freq_axis2) or np.max(np.abs(freq_axis - freq_axis2)) > 1e-12:
                raise ValueError("组内读取后频率轴不一致")

            chan_hz = m01_reader.chan_width_hz

            row_iter = list(g.iterrows())
            if tqdm is not None:
                row_iter = tqdm(row_iter, total=len(row_iter), desc=f"group {coarse_val:.3f}", leave=False, ncols=100)

            for idx, row in row_iter:
                out = {
                    "beam_freq_low_MHz": np.nan,
                    "beam_freq_high_MHz": np.nan,
                    "beam_track_vote_score": np.nan,
                    "beam_track_continuity": np.nan,
                    "beam_status": "failed",
                    "beam_error": "",
                }

                try:
                    start_f = float(row["start_freq_MHz"])
                    end_f = float(row["end_freq_MHz"])

                    # 候选本身的小窗口（从共享大 chunk 中再切）
                    cand_low = min(start_f, end_f) - args.window_pad_hz / 1e6
                    cand_high = max(start_f, end_f) + args.window_pad_hz / 1e6

                    out["beam_freq_low_MHz"] = cand_low
                    out["beam_freq_high_MHz"] = cand_high

                    # 在 group chunk 内进一步切片
                    local_mask = (freq_axis >= cand_low) & (freq_axis <= cand_high)
                    if not np.any(local_mask):
                        raise ValueError(f"候选局部窗口为空: {cand_low}-{cand_high} MHz")

                    sub_cols = np.where(local_mask)[0]
                    c0 = sub_cols[0]
                    c1 = sub_cols[-1] + 1

                    m01_sub = m01_chunk[:, c0:c1]
                    upper_sub = upper_chunk[:, c0:c1]
                    freq_sub = freq_axis[c0:c1]

                    if m01_sub.size == 0 or upper_sub.size == 0:
                        raise ValueError("候选局部读取为空")

                    # 自适应 sigma
                    sigma_hz = float(args.sigma_hz)
                    if args.adaptive_sigma and bw_col is not None and pd.notna(row[bw_col]):
                        try:
                            bw_hz = float(row[bw_col])
                            sigma_hz = max(sigma_hz, 0.5 * bw_hz, 2.0 * chan_hz)
                        except Exception:
                            pass

                    vote_score, continuity = score_one_candidate(
                        m01_sub=m01_sub,
                        upper_sub=upper_sub,
                        freq_axis_mhz=freq_sub,
                        start_freq_mhz=start_f,
                        end_freq_mhz=end_f,
                        sigma_hz=sigma_hz,
                        cutoff_sigma=args.cutoff_sigma,
                        continuity_core_hz=args.continuity_core_hz,
                    )

                    out["beam_track_vote_score"] = vote_score
                    out["beam_track_continuity"] = continuity
                    out["beam_status"] = "ok"
                    out["beam_error"] = ""

                except Exception as e:
                    out["beam_status"] = "failed"
                    out["beam_error"] = str(e)

                results.append((idx, out))

    finally:
        m01_reader.close()
        upper_reader.close()

    # 还原行顺序
    res_df = pd.DataFrame(
        [r[1] for r in sorted(results, key=lambda x: x[0])],
        index=[r[0] for r in sorted(results, key=lambda x: x[0])]
    )

    # 对齐并写回
    for col in res_df.columns:
        df.loc[res_df.index, col] = res_df[col]

    ok_count = int((df["beam_status"] == "ok").sum())
    fail_count = int((df["beam_status"] != "ok").sum())

    # 删除内部列
    df.drop(columns=["__freq_center", "__coarse_group"], inplace=True, errors="ignore")

    df.to_csv(args.out, index=False)

    print("\nDone.")
    print("Output:", args.out)
    print("Total rows :", len(df))
    print("OK rows    :", ok_count)
    print("Failed rows:", fail_count)

    if fail_count > 0:
        fail_out = str(Path(args.out).with_name(Path(args.out).stem + "_failed_rows.csv"))
        df[df["beam_status"] != "ok"].to_csv(fail_out, index=False)
        print("Failed rows written to:", fail_out)


if __name__ == "__main__":
    main()
