#include "mesh_loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace {

std::string ReadFileText(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

bool ExtractArrayText(const std::string& text, const std::string& key, std::string& out_array_text) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }
    const size_t bracket_pos = text.find('[', key_pos);
    if (bracket_pos == std::string::npos) {
        return false;
    }
    int depth = 0;
    size_t end_pos = std::string::npos;
    for (size_t i = bracket_pos; i < text.size(); ++i) {
        if (text[i] == '[') {
            depth++;
        } else if (text[i] == ']') {
            depth--;
            if (depth == 0) {
                end_pos = i;
                break;
            }
        }
    }
    if (end_pos == std::string::npos || end_pos <= bracket_pos) {
        return false;
    }
    out_array_text = text.substr(bracket_pos + 1, end_pos - bracket_pos - 1);
    return true;
}

std::vector<double> ParseNumbers(const std::string& text) {
    std::vector<double> numbers;
    std::string token;
    auto flush = [&]() {
        if (!token.empty()) {
            numbers.push_back(std::stod(token));
            token.clear();
        }
    };
    for (char c : text) {
        const bool is_num = std::isdigit(static_cast<unsigned char>(c)) ||
                            c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E';
        if (is_num) {
            token.push_back(c);
        } else {
            flush();
        }
    }
    flush();
    return numbers;
}

double TetSignedVolume6(const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3) {
    const Vec3 a = v1 - v0;
    const Vec3 b = v2 - v0;
    const Vec3 c = v3 - v0;
    return static_cast<double>(Dot(a, Cross(b, c)));
}

int NodesPerElementType(int element_type) {
    switch (element_type) {
        case 1: return 2;
        case 2: return 3;
        case 3: return 4;
        case 4: return 4;
        case 5: return 8;
        case 6: return 6;
        case 7: return 5;
        case 8: return 3;
        case 9: return 6;
        case 10: return 9;
        case 11: return 10;
        case 12: return 27;
        case 13: return 18;
        case 14: return 14;
        case 15: return 1;
        case 16: return 8;
        case 17: return 20;
        case 18: return 15;
        case 19: return 13;
        case 20: return 9;
        case 21: return 10;
        case 22: return 12;
        case 23: return 15;
        case 24: return 15;
        case 25: return 21;
        case 26: return 4;
        case 27: return 5;
        case 28: return 6;
        case 29: return 20;
        case 30: return 35;
        case 31: return 56;
        default: return -1;
    }
}

std::array<float, 9> Inverse3x3(const std::array<float, 9>& m) {
    const float det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);

    const float inv_det = 1.0f / det;
    return std::array<float, 9>{
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det
    };
}

} // namespace

bool LoadTetMeshJson(const std::string& path, MeshData& mesh, std::string& error) {
    const std::string text = ReadFileText(path);
    if (text.empty()) {
        error = "Failed to read mesh json: " + path;
        return false;
    }

    std::string vertices_text;
    std::string tets_text;
    if (!ExtractArrayText(text, "vertices", vertices_text)) {
        error = "Missing vertices array in json mesh.";
        return false;
    }
    if (!ExtractArrayText(text, "tetrahedra", tets_text)) {
        error = "Missing tetrahedra array in json mesh.";
        return false;
    }

    const auto vertex_numbers = ParseNumbers(vertices_text);
    const auto tet_numbers = ParseNumbers(tets_text);
    if (vertex_numbers.size() % 3 != 0 || tet_numbers.size() % 4 != 0) {
        error = "Invalid mesh json numeric array lengths.";
        return false;
    }

    mesh.verts.clear();
    mesh.tets.clear();
    mesh.verts.reserve(vertex_numbers.size() / 3);
    mesh.tets.reserve(tet_numbers.size() / 4);

    for (size_t i = 0; i < vertex_numbers.size(); i += 3) {
        mesh.verts.push_back(Vec3{
            static_cast<float>(vertex_numbers[i + 0]),
            static_cast<float>(vertex_numbers[i + 1]),
            static_cast<float>(vertex_numbers[i + 2])
        });
    }

    for (size_t i = 0; i < tet_numbers.size(); i += 4) {
        Tet t;
        t.i0 = static_cast<std::uint32_t>(tet_numbers[i + 0]);
        t.i1 = static_cast<std::uint32_t>(tet_numbers[i + 1]);
        t.i2 = static_cast<std::uint32_t>(tet_numbers[i + 2]);
        t.i3 = static_cast<std::uint32_t>(tet_numbers[i + 3]);
        mesh.tets.push_back(t);
    }

    for (Tet& t : mesh.tets) {
        const double det6 = TetSignedVolume6(
            mesh.verts[t.i0], mesh.verts[t.i1], mesh.verts[t.i2], mesh.verts[t.i3]);
        if (det6 < 0.0) {
            std::swap(t.i1, t.i2);
        }
    }

    return true;
}

bool LoadTetMeshMsh(const std::string& path, MeshData& mesh, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "Failed to open .msh file: " + path;
        return false;
    }

    bool parsed_nodes = false;
    bool parsed_elements = false;
    std::unordered_map<long long, std::uint32_t> tag_to_index;

    std::string token;
    while (in >> token) {
        if (token == "$MeshFormat") {
            double version = 0.0;
            int file_type = 0;
            int data_size = 0;
            in >> version >> file_type >> data_size;
            if (!in || file_type != 0) {
                error = "Only ASCII .msh is supported.";
                return false;
            }
            (void)version;
            (void)data_size;
        } else if (token == "$Nodes") {
            int num_entity_blocks = 0;
            int num_nodes = 0;
            long long min_tag = 0;
            long long max_tag = 0;
            in >> num_entity_blocks >> num_nodes >> min_tag >> max_tag;
            if (!in || num_entity_blocks <= 0 || num_nodes <= 0) {
                error = "Invalid $Nodes header.";
                return false;
            }

            mesh.verts.clear();
            mesh.verts.reserve(static_cast<size_t>(num_nodes));
            tag_to_index.clear();
            tag_to_index.reserve(static_cast<size_t>(num_nodes * 2));

            for (int b = 0; b < num_entity_blocks; ++b) {
                int entity_dim = 0;
                int entity_tag = 0;
                int parametric = 0;
                int num_nodes_in_block = 0;
                in >> entity_dim >> entity_tag >> parametric >> num_nodes_in_block;
                if (!in || num_nodes_in_block <= 0) {
                    error = "Invalid node block header.";
                    return false;
                }

                std::vector<long long> node_tags(static_cast<size_t>(num_nodes_in_block));
                for (int i = 0; i < num_nodes_in_block; ++i) {
                    in >> node_tags[static_cast<size_t>(i)];
                }

                for (int i = 0; i < num_nodes_in_block; ++i) {
                    double x = 0.0;
                    double y = 0.0;
                    double z = 0.0;
                    in >> x >> y >> z;
                    if (!in) {
                        error = "Invalid node coordinates.";
                        return false;
                    }
                    if (parametric) {
                        for (int p = 0; p < entity_dim; ++p) {
                            double dummy = 0.0;
                            in >> dummy;
                        }
                    }

                    const auto idx = static_cast<std::uint32_t>(mesh.verts.size());
                    mesh.verts.push_back(Vec3{
                        static_cast<float>(x),
                        static_cast<float>(y),
                        static_cast<float>(z)
                    });
                    tag_to_index[node_tags[static_cast<size_t>(i)]] = idx;
                }
            }
            parsed_nodes = true;
        } else if (token == "$Elements") {
            int num_entity_blocks = 0;
            int num_elements = 0;
            long long min_tag = 0;
            long long max_tag = 0;
            in >> num_entity_blocks >> num_elements >> min_tag >> max_tag;
            if (!in || num_entity_blocks <= 0 || num_elements <= 0) {
                error = "Invalid $Elements header.";
                return false;
            }

            mesh.tets.clear();
            mesh.tets.reserve(static_cast<size_t>(num_elements));

            for (int b = 0; b < num_entity_blocks; ++b) {
                int entity_dim = 0;
                int entity_tag = 0;
                int element_type = 0;
                int num_elements_in_block = 0;
                in >> entity_dim >> entity_tag >> element_type >> num_elements_in_block;
                if (!in || num_elements_in_block <= 0) {
                    error = "Invalid element block header.";
                    return false;
                }

                const int nodes_per = NodesPerElementType(element_type);
                if (nodes_per <= 0) {
                    error = "Unsupported gmsh element type encountered.";
                    return false;
                }

                for (int e = 0; e < num_elements_in_block; ++e) {
                    long long elem_tag = 0;
                    in >> elem_tag;
                    (void)elem_tag;

                    std::vector<long long> node_tags(static_cast<size_t>(nodes_per));
                    for (int i = 0; i < nodes_per; ++i) {
                        in >> node_tags[static_cast<size_t>(i)];
                    }
                    if (!in) {
                        error = "Invalid element row.";
                        return false;
                    }

                    if (element_type == 4 || element_type == 11 || element_type == 29 || element_type == 30) {
                        Tet t;
                        t.i0 = tag_to_index[node_tags[0]];
                        t.i1 = tag_to_index[node_tags[1]];
                        t.i2 = tag_to_index[node_tags[2]];
                        t.i3 = tag_to_index[node_tags[3]];
                        mesh.tets.push_back(t);
                    }
                }
            }
            parsed_elements = true;
        }
    }

    if (!parsed_nodes || !parsed_elements || mesh.verts.empty() || mesh.tets.empty()) {
        error = "Failed to parse nodes/elements from .msh.";
        return false;
    }

    for (Tet& t : mesh.tets) {
        const double det6 = TetSignedVolume6(
            mesh.verts[t.i0], mesh.verts[t.i1], mesh.verts[t.i2], mesh.verts[t.i3]);
        if (det6 < 0.0) {
            std::swap(t.i1, t.i2);
        }
    }

    return true;
}

bool LoadTetMeshAuto(const std::string& path, MeshData& mesh, std::string& error) {
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".msh") {
        return LoadTetMeshMsh(path, mesh, error);
    }
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
        return LoadTetMeshJson(path, mesh, error);
    }
    error = "Unsupported mesh extension: " + path;
    return false;
}

bool BuildRestTetData(const std::vector<Vec3>& rest_positions, float density, MeshData& mesh, std::string& error) {
    if (rest_positions.size() != mesh.verts.size()) {
        error = "Rest positions size mismatch.";
        return false;
    }

    const size_t n = rest_positions.size();
    mesh.dm_inv.assign(mesh.tets.size(), std::array<float, 9>{});
    mesh.vol_rest.assign(mesh.tets.size(), 0.0f);
    mesh.mass.assign(n, 0.0f);
    mesh.inv_mass.assign(n, 0.0f);

    for (size_t i = 0; i < mesh.tets.size(); ++i) {
        Tet& t = mesh.tets[i];
        Vec3 v0 = rest_positions[t.i0];
        Vec3 v1 = rest_positions[t.i1];
        Vec3 v2 = rest_positions[t.i2];
        Vec3 v3 = rest_positions[t.i3];

        const auto det6 = TetSignedVolume6(v0, v1, v2, v3);
        if (det6 < 0.0) {
            std::swap(t.i1, t.i2);
            v1 = rest_positions[t.i1];
            v2 = rest_positions[t.i2];
        }

        std::array<float, 9> dm{
            v1.x - v0.x, v2.x - v0.x, v3.x - v0.x,
            v1.y - v0.y, v2.y - v0.y, v3.y - v0.y,
            v1.z - v0.z, v2.z - v0.z, v3.z - v0.z
        };
        const float det =
            dm[0] * (dm[4] * dm[8] - dm[5] * dm[7]) -
            dm[1] * (dm[3] * dm[8] - dm[5] * dm[6]) +
            dm[2] * (dm[3] * dm[7] - dm[4] * dm[6]);
        if (std::abs(det) < 1e-12f) {
            error = "Degenerate tet detected while building Dm_inv.";
            return false;
        }
        mesh.dm_inv[i] = Inverse3x3(dm);

        float vol = det / 6.0f;
        if (vol < 0.0f) {
            vol = -vol;
        }
        mesh.vol_rest[i] = vol;

        const float m_tet = density * vol;
        mesh.mass[t.i0] += m_tet * 0.25f;
        mesh.mass[t.i1] += m_tet * 0.25f;
        mesh.mass[t.i2] += m_tet * 0.25f;
        mesh.mass[t.i3] += m_tet * 0.25f;
    }

    for (size_t i = 0; i < n; ++i) {
        mesh.inv_mass[i] = mesh.mass[i] > 0.0f ? 1.0f / mesh.mass[i] : 0.0f;
    }

    return true;
}
