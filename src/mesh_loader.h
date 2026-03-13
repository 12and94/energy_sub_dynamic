#pragma once

#include <string>
#include <vector>

#include "types.h"

bool LoadTetMeshMsh(const std::string& path, MeshData& mesh, std::string& error);
bool LoadTetMeshJson(const std::string& path, MeshData& mesh, std::string& error);
bool LoadTetMeshAuto(const std::string& path, MeshData& mesh, std::string& error);

bool BuildRestTetData(const std::vector<Vec3>& rest_positions, float density, MeshData& mesh, std::string& error);
