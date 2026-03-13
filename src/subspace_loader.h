#pragma once

#include <string>

#include "types.h"

bool LoadSubspaceBasisJson(const std::string& path, int max_cols, SubspaceBasisData& out_data, std::string& error);
