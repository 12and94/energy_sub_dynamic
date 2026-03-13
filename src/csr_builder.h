#pragma once

#include "types.h"

CsrAdjacency BuildTetVertexCsr(std::size_t num_vertices, const std::vector<Tet>& tets);
