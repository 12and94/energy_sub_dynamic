#include "csr_builder.h"

#include <array>

CsrAdjacency BuildTetVertexCsr(std::size_t num_vertices, const std::vector<Tet>& tets) {
    CsrAdjacency csr;
    csr.offsets.assign(num_vertices + 1, 0u);

    for (const Tet& t : tets) {
        csr.offsets[t.i0 + 1] += 1;
        csr.offsets[t.i1 + 1] += 1;
        csr.offsets[t.i2 + 1] += 1;
        csr.offsets[t.i3 + 1] += 1;
    }
    for (std::size_t i = 0; i < num_vertices; ++i) {
        csr.offsets[i + 1] += csr.offsets[i];
    }

    csr.tet_ids.assign(csr.offsets.back(), 0u);
    csr.local_ids.assign(csr.offsets.back(), 0u);

    std::vector<std::uint32_t> cursor = csr.offsets;
    for (std::size_t ti = 0; ti < tets.size(); ++ti) {
        const Tet& t = tets[ti];
        const std::array<std::uint32_t, 4> vids{t.i0, t.i1, t.i2, t.i3};
        for (std::uint32_t local = 0; local < 4; ++local) {
            const std::uint32_t v = vids[local];
            const std::uint32_t idx = cursor[v]++;
            csr.tet_ids[idx] = static_cast<std::uint32_t>(ti);
            csr.local_ids[idx] = local;
        }
    }

    return csr;
}
