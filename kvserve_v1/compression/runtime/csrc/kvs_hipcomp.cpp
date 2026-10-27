// SPDX-License-Identifier: Apache-2.0
// Based on the supplied DCU hipCOMP/ctypes port. ABI 1 is intentionally NOT
// compatible with its ambiguously named libkvserve_hipcomp_snappy.so.
// All device allocations belong to the caller's per-slot bounded workspace.
#include <hip/hip_runtime.h>
#include <hipcomp/snappy.h>
#include <hipcomp/cascaded.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr uint64_t ALIGN = 256, HEADER = 40, ENTRY = 16;
constexpr unsigned char MAGIC[8] = {'K','V','S','H','P','0','0','1'};
static_assert(sizeof(size_t) == 8, "64-bit HIP runtime required");
int hc(hipcompStatus_t rc) { return rc == hipcompSuccess ? 0 : 2000 + int(rc); }
int hp(hipError_t rc) { return rc == hipSuccess ? 0 : 1000 + int(rc); }
bool add(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b; return true;
}
bool mul(uint64_t a, uint64_t b, uint64_t& out) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b; return true;
}
uint64_t aligned(uint64_t value) { return (value + ALIGN - 1) / ALIGN * ALIGN; }
bool carve(uint64_t bytes, uint64_t& total, uint64_t& offset) {
    uint64_t aligned;
    if (!add(total, ALIGN - 1, aligned)) return false;
    offset = aligned / ALIGN * ALIGN;
    return add(offset, bytes, total);
}
struct Plan {
    uint64_t count, slot, payload, workspace, temp_bytes;
    uint64_t packed, in_ptrs, out_ptrs, in_sizes, comp_sizes;
    uint64_t out_sizes, actual_sizes, statuses, temp;
};
hipcompBatchedCascadedOpts_t cascaded(uint64_t chunk) {
    auto opts = hipcompBatchedCascadedDefaultOpts;
    opts.type = HIPCOMP_TYPE_UCHAR;
    opts.chunk_size = static_cast<size_t>(chunk);
    return opts;
}
int make_plan(uint32_t algorithm, uint64_t input, uint64_t chunk, Plan& p) {
    if ((algorithm != 1 && algorithm != 2) || !input || chunk < 256 ||
        chunk > (16u << 20) || chunk % ALIGN) return 1;
    p.count = input / chunk + (input % chunk != 0);
    if (p.count > std::numeric_limits<unsigned int>::max()) return 2;
    size_t temp_enc = 0, temp_dec = 0, max_out = 0;
    int rc;
    if (algorithm == 1) {
        auto opts = hipcompBatchedSnappyDefaultOpts;
        rc = hc(hipcompBatchedSnappyCompressGetTempSize(p.count, chunk, opts, &temp_enc));
        if (rc) return rc;
        rc = hc(hipcompBatchedSnappyCompressGetMaxOutputChunkSize(chunk, opts, &max_out));
        if (rc) return rc;
        rc = hc(hipcompBatchedSnappyDecompressGetTempSize(p.count, chunk, &temp_dec));
    } else {
        auto opts = cascaded(chunk);
        rc = hc(hipcompBatchedCascadedCompressGetTempSize(p.count, chunk, opts, &temp_enc));
        if (rc) return rc;
        rc = hc(hipcompBatchedCascadedCompressGetMaxOutputChunkSize(chunk, opts, &max_out));
        if (rc) return rc;
        rc = hc(hipcompBatchedCascadedDecompressGetTempSize(p.count, chunk, &temp_dec));
    }
    if (rc) return rc;
    if (!max_out) return 2;
    uint64_t rounded;
    if (!add(max_out, ALIGN - 1, rounded)) return 2;
    p.slot = rounded / ALIGN * ALIGN;
    p.temp_bytes = std::max(temp_enc, temp_dec);
    uint64_t packed_bytes, array_bytes, status_bytes, header_bytes;
    if (!mul(p.count, p.slot, packed_bytes) || !mul(p.count, 8, array_bytes) ||
        !mul(p.count, sizeof(hipcompStatus_t), status_bytes) ||
        !mul(p.count, ENTRY, header_bytes) || !add(header_bytes, HEADER, header_bytes) ||
        !add(header_bytes, ALIGN - 1, rounded)) return 2;
    header_bytes = rounded / ALIGN * ALIGN;
    if (!add(header_bytes, packed_bytes, p.payload)) return 2;
    uint64_t total = 0, ignored;
    if (!carve(packed_bytes, total, p.packed) ||
        !carve(array_bytes, total, p.in_ptrs) || !carve(array_bytes, total, p.out_ptrs) ||
        !carve(array_bytes, total, p.in_sizes) || !carve(array_bytes, total, p.comp_sizes) ||
        !carve(array_bytes, total, p.out_sizes) || !carve(array_bytes, total, p.actual_sizes) ||
        !carve(status_bytes, total, p.statuses) || !carve(p.temp_bytes, total, p.temp) ||
        !carve(0, total, ignored)) return 2;
    p.workspace = total;
    return 0;
}
template<class T> T* at(void* base, uint64_t offset) {
    return reinterpret_cast<T*>(static_cast<unsigned char*>(base) + offset);
}
void put(unsigned char* dst, uint64_t value, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) dst[i] = (value >> (8 * i)) & 255;
}
uint64_t get(const unsigned char* src, size_t bytes) {
    uint64_t out = 0;
    for (size_t i = 0; i < bytes; ++i) out |= uint64_t(src[i]) << (8 * i);
    return out;
}
bool overlap(const void* a, uint64_t as, const void* b, uint64_t bs) {
    uint64_t ae, be;
    if (!add(reinterpret_cast<uintptr_t>(a), as, ae) ||
        !add(reinterpret_cast<uintptr_t>(b), bs, be)) return true;
    return std::max(reinterpret_cast<uintptr_t>(a), reinterpret_cast<uintptr_t>(b)) < std::min(ae, be);
}
int check_buffers(const void* input, uint64_t in_bytes, void* output, uint64_t out_bytes,
                  void* workspace, uint64_t work_bytes) {
    if (!input || !output || !workspace || reinterpret_cast<uintptr_t>(workspace) % ALIGN ||
        reinterpret_cast<uintptr_t>(input) % ALIGN || reinterpret_cast<uintptr_t>(output) % ALIGN ||
        overlap(input, in_bytes, output, out_bytes) ||
        overlap(input, in_bytes, workspace, work_bytes) ||
        overlap(output, out_bytes, workspace, work_bytes)) return 1;
    return 0;
}
__global__ void compact(void* const* inputs, void* const* outputs,
                        const size_t* sizes, size_t count) {
    const size_t i = blockIdx.x;
    if (i >= count) return;
    auto src = static_cast<const unsigned char*>(inputs[i]);
    auto dst = static_cast<unsigned char*>(outputs[i]);
    for (size_t j = threadIdx.x; j < sizes[i]; j += blockDim.x) dst[j] = src[j];
}
}  // namespace

extern "C" uint32_t kvs_hipcomp_abi_version() { return 1; }
extern "C" uint32_t kvs_hipcomp_algorithm_mask() { return (1u << 1) | (1u << 2); }
extern "C" int kvs_hipcomp_plan(uint32_t algo, uint64_t input, uint64_t chunk,
                                uint64_t* workspace, uint64_t* payload) try {
    if (!workspace || !payload) return 1;
    Plan p{};
    int rc = make_plan(algo, input, chunk, p);
    if (rc) return rc;
    *workspace = p.workspace; *payload = p.payload;
    return 0;
} catch (...) { return 6; }

// Synchronous completion is deliberate in this initial DCU backend. No host
// metadata can outlive its vector; no device scratch escapes its runtime slot.
extern "C" int kvs_hipcomp_compress(
    uint32_t algo, const void* input, uint64_t bytes, uint64_t max_input, uint64_t chunk,
    void* output, uint64_t capacity, void* workspace, uint64_t work_capacity,
    uint64_t stream_handle, uint64_t* written) try {
    Plan p{};
    int rc = make_plan(algo, max_input, chunk, p);
    if (rc) return rc;
    if (!written || !bytes || bytes > max_input || work_capacity < p.workspace ||
        capacity < p.payload || check_buffers(input, bytes, output, capacity, workspace, work_capacity)) return 1;
    const size_t n = bytes / chunk + (bytes % chunk != 0);
    if (n > std::numeric_limits<unsigned int>::max()) return 2;
    auto stream = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream_handle));
    std::vector<void*> in(n), out(n);
    std::vector<size_t> sizes(n), compressed(n);
    for (size_t i = 0; i < n; ++i) {
        in[i] = const_cast<unsigned char*>(static_cast<const unsigned char*>(input)) + i * chunk;
        out[i] = at<void>(workspace, p.packed + i * p.slot);
        sizes[i] = std::min(chunk, bytes - i * chunk);
    }
    if ((rc = hp(hipMemcpy(at<void*>(workspace, p.in_ptrs), in.data(), n * 8, hipMemcpyHostToDevice)))) return rc;
    if ((rc = hp(hipMemcpy(at<void*>(workspace, p.out_ptrs), out.data(), n * 8, hipMemcpyHostToDevice)))) return rc;
    if ((rc = hp(hipMemcpy(at<size_t>(workspace, p.in_sizes), sizes.data(), n * 8, hipMemcpyHostToDevice)))) return rc;
    auto inputs = reinterpret_cast<const void* const*>(at<void*>(workspace, p.in_ptrs));
    if (algo == 1) {
        rc = hc(hipcompBatchedSnappyCompressAsync(inputs, at<size_t>(workspace, p.in_sizes),
            chunk, n, at<void>(workspace, p.temp), p.temp_bytes,
            at<void*>(workspace, p.out_ptrs), at<size_t>(workspace, p.comp_sizes),
            hipcompBatchedSnappyDefaultOpts, stream));
    } else {
        rc = hc(hipcompBatchedCascadedCompressAsync(inputs, at<size_t>(workspace, p.in_sizes),
            chunk, n, at<void>(workspace, p.temp), p.temp_bytes,
            at<void*>(workspace, p.out_ptrs), at<size_t>(workspace, p.comp_sizes), cascaded(chunk), stream));
    }
    if (rc) return rc;
    if ((rc = hp(hipStreamSynchronize(stream)))) return rc;
    if ((rc = hp(hipMemcpy(compressed.data(), at<size_t>(workspace, p.comp_sizes), n * 8, hipMemcpyDeviceToHost)))) return rc;
    const uint64_t header_size = aligned(HEADER + ENTRY * n);
    std::vector<unsigned char> header(header_size);
    std::memcpy(header.data(), MAGIC, 8);
    put(header.data() + 8, 1, 4); put(header.data() + 12, algo, 4);
    put(header.data() + 16, chunk, 8); put(header.data() + 24, bytes, 8);
    put(header.data() + 32, n, 8);
    uint64_t end = header_size;
    for (size_t i = 0; i < n; ++i) {
        end = aligned(end);
        if (!compressed[i] || compressed[i] > p.slot || compressed[i] > capacity - end) return 3;
        put(header.data() + HEADER + ENTRY * i, sizes[i], 8);
        put(header.data() + HEADER + ENTRY * i + 8, compressed[i], 8);
        in[i] = out[i]; out[i] = at<void>(output, end);
        end += compressed[i];
    }
    // Padding is transmitted too: never expose stale data from a previous slot.
    if ((rc = hp(hipMemset(output, 0, end)))) return rc;
    if ((rc = hp(hipMemcpy(output, header.data(), header_size, hipMemcpyHostToDevice)))) return rc;
    if ((rc = hp(hipMemcpy(at<void*>(workspace, p.in_ptrs), in.data(), n * 8, hipMemcpyHostToDevice)))) return rc;
    if ((rc = hp(hipMemcpy(at<void*>(workspace, p.out_ptrs), out.data(), n * 8, hipMemcpyHostToDevice)))) return rc;
    hipLaunchKernelGGL(compact, dim3(static_cast<unsigned int>(n)), dim3(256), 0, stream,
        at<void*>(workspace, p.in_ptrs), at<void*>(workspace, p.out_ptrs), at<size_t>(workspace, p.comp_sizes), n);
    if ((rc = hp(hipGetLastError()))) return rc;
    if ((rc = hp(hipStreamSynchronize(stream)))) return rc;
    *written = end;
    return 0;
} catch (...) { return 6; }

extern "C" int kvs_hipcomp_decompress(
    uint32_t algo, const void* input, uint64_t bytes, uint64_t raw_bytes, uint64_t max_input,
    uint64_t chunk, void* output, uint64_t out_capacity, void* workspace,
    uint64_t work_capacity, uint64_t stream_handle) try {
    Plan p{};
    int rc = make_plan(algo, max_input, chunk, p);
    if (rc) return rc;
    if (bytes < HEADER || bytes > p.payload || !raw_bytes || raw_bytes > max_input ||
        out_capacity < raw_bytes || work_capacity < p.workspace ||
        check_buffers(input, bytes, output, out_capacity, workspace, work_capacity)) return 1;
    auto stream = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(stream_handle));
    if ((rc = hp(hipStreamSynchronize(stream)))) return rc;
    unsigned char fixed[HEADER];
    if ((rc = hp(hipMemcpy(fixed, input, HEADER, hipMemcpyDeviceToHost)))) return rc;
    const size_t n = raw_bytes / chunk + (raw_bytes % chunk != 0);
    if (std::memcmp(fixed, MAGIC, 8) || get(fixed + 8, 4) != 1 || get(fixed + 12, 4) != algo ||
        get(fixed + 16, 8) != chunk || get(fixed + 24, 8) != raw_bytes || get(fixed + 32, 8) != n) return 4;
    const uint64_t header_size = aligned(HEADER + ENTRY * n);
    if (header_size > bytes) return 4;
    std::vector<unsigned char> header(header_size);
    if ((rc = hp(hipMemcpy(header.data(), input, header_size, hipMemcpyDeviceToHost)))) return rc;
    std::vector<void*> in(n), out(n);
    std::vector<size_t> compressed(n), expected(n), actual(n);
    std::vector<hipcompStatus_t> status(n);
    uint64_t src = header_size, dst = 0;
    for (size_t i = 0; i < n; ++i) {
        src = aligned(src);
        expected[i] = get(header.data() + HEADER + ENTRY * i, 8);
        compressed[i] = get(header.data() + HEADER + ENTRY * i + 8, 8);
        if (src > bytes || expected[i] != std::min(chunk, raw_bytes - dst) || !compressed[i] ||
            compressed[i] > p.slot || compressed[i] > bytes - src) return 4;
        in[i] = const_cast<unsigned char*>(static_cast<const unsigned char*>(input)) + src;
        out[i] = at<void>(output, dst);
        src += compressed[i]; dst += expected[i];
    }
    if (src != bytes || dst != raw_bytes) return 4;
    if ((rc = hp(hipMemcpy(at<void*>(workspace, p.in_ptrs), in.data(), n * 8, hipMemcpyHostToDevice)))) return rc;
    if ((rc = hp(hipMemcpy(at<void*>(workspace, p.out_ptrs), out.data(), n * 8, hipMemcpyHostToDevice)))) return rc;
    if ((rc = hp(hipMemcpy(at<size_t>(workspace, p.comp_sizes), compressed.data(), n * 8, hipMemcpyHostToDevice)))) return rc;
    if ((rc = hp(hipMemcpy(at<size_t>(workspace, p.out_sizes), expected.data(), n * 8, hipMemcpyHostToDevice)))) return rc;
    auto inputs = reinterpret_cast<const void* const*>(at<void*>(workspace, p.in_ptrs));
    if (algo == 1) {
        rc = hc(hipcompBatchedSnappyDecompressAsync(inputs, at<size_t>(workspace, p.comp_sizes),
            at<size_t>(workspace, p.out_sizes), at<size_t>(workspace, p.actual_sizes), n,
            at<void>(workspace, p.temp), p.temp_bytes, at<void*>(workspace, p.out_ptrs),
            at<hipcompStatus_t>(workspace, p.statuses), stream));
    } else {
        rc = hc(hipcompBatchedCascadedDecompressAsync(inputs, at<size_t>(workspace, p.comp_sizes),
            at<size_t>(workspace, p.out_sizes), at<size_t>(workspace, p.actual_sizes), n,
            at<void>(workspace, p.temp), p.temp_bytes, at<void*>(workspace, p.out_ptrs),
            at<hipcompStatus_t>(workspace, p.statuses), stream));
    }
    if (rc) return rc;
    if ((rc = hp(hipStreamSynchronize(stream)))) return rc;
    if ((rc = hp(hipMemcpy(actual.data(), at<size_t>(workspace, p.actual_sizes), n * 8, hipMemcpyDeviceToHost)))) return rc;
    if ((rc = hp(hipMemcpy(status.data(), at<hipcompStatus_t>(workspace, p.statuses), n * sizeof(hipcompStatus_t), hipMemcpyDeviceToHost)))) return rc;
    for (size_t i = 0; i < n; ++i) {
        if (status[i] != hipcompSuccess) return 3000 + int(status[i]);
        if (actual[i] != expected[i]) return 5;
    }
    return 0;
} catch (...) { return 6; }
