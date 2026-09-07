package simd

// HasSgemmAsm reports whether SIMD-accelerated SGEMM kernels are available
// on this architecture (amd64 AVX2+FMA, arm64 NEON). This reflects
// architecture support only; on amd64 it does not by itself guarantee the
// AVX2+FMA fast path runs — see FastPathEnabled for the runtime check.
const HasSgemmAsm = hasSgemmAsm
