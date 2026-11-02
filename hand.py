"""My hand-written kernels (third baseline, <=1h effort). Same spec format the LLM must produce."""
import json, os

A = {  # one threadgroup per row; fused add + rmsnorm; h kept in registers is not attempted (generic H)
    "threadgroup": [256, 1, 1], "grid": ["256*rows", "1", "1"],
    "source": """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    auto xr = x + row * H; auto rr = r + row * H;
    auto hr = h + row * H; auto yr = y + row * H;
    float acc = 0.0f;
    for (uint i = tid; i < (uint)H; i += 256) {
        half hv = xr[i] + rr[i];
        hr[i] = hv;
        float f = float(hv); acc += f * f;
    }
    threadgroup float part[8];
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint k = 0; k < 8; k++) tot += part[k];
    float inv = metal::rsqrt(tot / H + 1e-5f);
    for (uint i = tid; i < (uint)H; i += 256) {
        float hv = float(xr[i] + rr[i]);
        yr[i] = half(hv * inv * float(w[i]));
    }
""",
}

B = {  # 4 elements per thread
    "threadgroup": [256, 1, 1], "grid": ["cdiv(rows*I, 4)", "1", "1"],
    "source": """
    uint n = (uint)rows * (uint)I;
    uint i = thread_position_in_grid.x * 4;
    for (uint k = i; k < min(i + 4, n); k++) {
        float gv = float(g[k]);
        out[k] = half(gv / (1.0f + metal::exp(-gv)) * float(u[k]));
    }
""",
}

C = {  # one simdgroup per output row, 8 rows per threadgroup; each lane reads one uint32 (8 nibbles) per step
    "threadgroup": [256, 1, 1], "grid": ["256*cdiv(N, 8)", "1", "1"],
    "source": """
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint n = threadgroup_position_in_grid.x * 8 + sg;
    if (n >= (uint)N) return;
    const uint KW = K / 8, KG = K / 64;
    auto wr = wq + n * KW;
    float acc = 0.0f;
    for (uint j = lane; j < KW; j += 32) {
        uint q = wr[j];
        uint grp = j / 8;
        float sc = float(scales[n * KG + grp]), bi = float(biases[n * KG + grp]);
        float dot = 0.0f, xs = 0.0f;
        for (uint b = 0; b < 8; b++) {
            float xv = float(x[j * 8 + b]);
            dot += xv * float((q >> (4 * b)) & 0xF);
            xs += xv;
        }
        acc += sc * dot + bi * xs;
    }
    acc = simd_sum(acc);
    if (lane == 0) out[n] = half(acc);
""",
}

if __name__ == "__main__":
    os.makedirs("hand", exist_ok=True)
    for k, v in {"A": A, "B": B, "C": C}.items():
        json.dump(v, open(f"hand/{k}.json", "w"), indent=1)
