#include "nuri/pch.h"

#include "mesh_importer.h"

#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#if __has_include(<assimp/GltfMaterial.h>)
#include <assimp/GltfMaterial.h>
#define NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS 1
#else
#define NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS 0
#endif

namespace nuri {
namespace {
constexpr float kMeshoptOverdrawThreshold = 1.05f;
constexpr size_t kTriangleIndexCount = 3;

std::string normalizeExternalTexturePath(const std::filesystem::path &modelPath,
                                         std::string_view rawPath) {
  if (rawPath.empty()) {
    return {};
  }
  std::filesystem::path texturePath{std::string(rawPath)};
  if (!texturePath.is_absolute()) {
    texturePath = modelPath.parent_path() / texturePath;
  }
  return texturePath.lexically_normal().string();
}

ImportedMaterialTexture readMaterialTextureSlot(const aiMaterial &material,
                                                aiTextureType textureType,
                                                const std::filesystem::path &modelPath) {
  aiString texturePath;
  aiTextureMapping textureMapping = aiTextureMapping_UV;
  uint32_t uvIndex = 0;
  ai_real blendFactor = 1.0f;
  aiTextureOp textureOp = aiTextureOp_Multiply;
  aiTextureMapMode mapMode[2] = {aiTextureMapMode_Wrap, aiTextureMapMode_Wrap};
  const aiReturn result =
      material.GetTexture(textureType, 0, &texturePath, &textureMapping,
                          &uvIndex, &blendFactor, &textureOp, mapMode);
  if (result != aiReturn_SUCCESS || texturePath.length == 0) {
    return {};
  }

  ImportedMaterialTexture out{};
  out.uvSet = uvIndex;
  const std::string_view rawPath(texturePath.C_Str(), texturePath.length);
  out.isEmbedded = !rawPath.empty() && rawPath.front() == '*';
  out.path = out.isEmbedded ? std::string(rawPath)
                            : normalizeExternalTexturePath(modelPath, rawPath);
  return out;
}

ImportedMaterialTexture firstAvailableTextureSlot(
    const aiMaterial &material, const std::filesystem::path &modelPath,
    std::span<const aiTextureType> textureTypes) {
  for (const aiTextureType textureType : textureTypes) {
    ImportedMaterialTexture slot =
        readMaterialTextureSlot(material, textureType, modelPath);
    if (!slot.path.empty()) {
      return slot;
    }
  }
  return {};
}

ImportedMaterialInfo parseMaterial(const aiMaterial &material,
                                   const std::filesystem::path &modelPath) {
  ImportedMaterialInfo parsed{};

  if (material.GetName().length > 0) {
    parsed.name.assign(material.GetName().C_Str(), material.GetName().length);
  }

  aiColor4D baseColor(1.0f, 1.0f, 1.0f, 1.0f);
#if NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS
  if (material.Get(AI_MATKEY_BASE_COLOR, baseColor) == aiReturn_SUCCESS) {
    parsed.baseColorFactor =
        glm::vec4(baseColor.r, baseColor.g, baseColor.b, baseColor.a);
  } else
#endif
      if (material.Get(AI_MATKEY_COLOR_DIFFUSE, baseColor) ==
          aiReturn_SUCCESS) {
    parsed.baseColorFactor =
        glm::vec4(baseColor.r, baseColor.g, baseColor.b, baseColor.a);
  }

  aiColor3D emissiveColor(0.0f, 0.0f, 0.0f);
  if (material.Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor) ==
      aiReturn_SUCCESS) {
    parsed.emissiveFactor =
        glm::vec3(emissiveColor.r, emissiveColor.g, emissiveColor.b);
  }

  ai_real metallicFactor = 1.0f;
  if (material.Get(AI_MATKEY_METALLIC_FACTOR, metallicFactor) ==
      aiReturn_SUCCESS) {
    parsed.metallicFactor = static_cast<float>(metallicFactor);
  }

  ai_real roughnessFactor = 1.0f;
  if (material.Get(AI_MATKEY_ROUGHNESS_FACTOR, roughnessFactor) ==
      aiReturn_SUCCESS) {
    parsed.roughnessFactor = static_cast<float>(roughnessFactor);
  }

  aiColor3D sheenColor(1.0f, 1.0f, 1.0f);
  if (material.Get(AI_MATKEY_SHEEN_COLOR_FACTOR, sheenColor) ==
      aiReturn_SUCCESS) {
    parsed.sheenColorFactor =
        glm::vec3(sheenColor.r, sheenColor.g, sheenColor.b);
    const float sheenMax =
        std::max(parsed.sheenColorFactor.x,
                 std::max(parsed.sheenColorFactor.y, parsed.sheenColorFactor.z));
    parsed.sheenWeight = sheenMax > 0.0f ? 1.0f : 0.0f;
  }
  ai_real sheenRoughness = 0.0f;
  if (material.Get(AI_MATKEY_SHEEN_ROUGHNESS_FACTOR, sheenRoughness) ==
      aiReturn_SUCCESS) {
    parsed.sheenRoughnessFactor =
        std::clamp(static_cast<float>(sheenRoughness), 0.0f, 1.0f);
    parsed.sheenWeight = parsed.sheenRoughnessFactor > 0.0f ? 1.0f : 0.0f;
  }

#if NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS
  ai_real normalScale = 1.0f;
  if (material.Get(AI_MATKEY_GLTF_TEXTURE_SCALE(aiTextureType_NORMALS, 0),
                   normalScale) == aiReturn_SUCCESS) {
    parsed.normalScale = static_cast<float>(normalScale);
  }
  ai_real occlusionStrength = 1.0f;
  if (material.Get(
          AI_MATKEY_GLTF_TEXTURE_SCALE(aiTextureType_AMBIENT_OCCLUSION, 0),
          occlusionStrength) == aiReturn_SUCCESS) {
    parsed.occlusionStrength = static_cast<float>(occlusionStrength);
  }
#endif

  int32_t doubleSided = 0;
  if (material.Get(AI_MATKEY_TWOSIDED, doubleSided) == aiReturn_SUCCESS) {
    parsed.doubleSided = doubleSided != 0;
  }

#if NURI_ASSIMP_HAS_GLTF_MATERIAL_KEYS
  aiString alphaMode;
  if (material.Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == aiReturn_SUCCESS) {
    const std::string_view alphaModeStr(alphaMode.C_Str(), alphaMode.length);
    if (alphaModeStr == "MASK") {
      parsed.alphaMode = ImportedMaterialAlphaMode::Mask;
    } else if (alphaModeStr == "BLEND") {
      parsed.alphaMode = ImportedMaterialAlphaMode::Blend;
    } else {
      parsed.alphaMode = ImportedMaterialAlphaMode::Opaque;
    }
  }

  ai_real alphaCutoff = parsed.alphaCutoff;
  if (material.Get(AI_MATKEY_GLTF_ALPHACUTOFF, alphaCutoff) ==
      aiReturn_SUCCESS) {
    parsed.alphaCutoff = static_cast<float>(alphaCutoff);
  }
#endif

  parsed.baseColor = firstAvailableTextureSlot(
      material, modelPath, std::array<aiTextureType, 2>{
                             aiTextureType_BASE_COLOR, aiTextureType_DIFFUSE});
  parsed.metallicRoughness = firstAvailableTextureSlot(
      material, modelPath,
      std::array<aiTextureType, 2>{aiTextureType_METALNESS,
                                   aiTextureType_DIFFUSE_ROUGHNESS});
  parsed.normal = firstAvailableTextureSlot(
      material, modelPath, std::array<aiTextureType, 1>{aiTextureType_NORMALS});
  parsed.occlusion = firstAvailableTextureSlot(
      material, modelPath,
      std::array<aiTextureType, 2>{aiTextureType_AMBIENT_OCCLUSION,
                                   aiTextureType_LIGHTMAP});
  parsed.emissive = firstAvailableTextureSlot(
      material, modelPath, std::array<aiTextureType, 1>{aiTextureType_EMISSIVE});

  return parsed;
}

size_t meshIndexCount(const aiMesh &mesh) {
  size_t count = 0;
  for (unsigned int faceIndex = 0; faceIndex < mesh.mNumFaces; ++faceIndex) {
    const aiFace &face = mesh.mFaces[faceIndex];
    if (face.mNumIndices != kTriangleIndexCount) {
      continue;
    }
    count += kTriangleIndexCount;
  }
  return count;
}

float lodRatioFor(const MeshImportOptions &options, uint32_t lodIndex) {
  if (options.lodTriangleRatios.empty()) {
    return 0.5f;
  }

  const size_t ratioIndex =
      std::min<size_t>(lodIndex - 1, options.lodTriangleRatios.size() - 1);
  return std::max(options.lodTriangleRatios[ratioIndex], 0.0f);
}

BoundingBox computeSubmeshBounds(std::span<const Vertex> vertices) {
  if (vertices.empty()) {
    return BoundingBox(glm::vec3(0.0f), glm::vec3(0.0f));
  }

  glm::vec3 minPos = vertices.front().position;
  glm::vec3 maxPos = vertices.front().position;
  for (size_t i = 1; i < vertices.size(); ++i) {
    minPos = glm::min(minPos, vertices[i].position);
    maxPos = glm::max(maxPos, vertices[i].position);
  }
  return BoundingBox(minPos, maxPos);
}

uint32_t clampLodCount(const MeshImportOptions &options) {
  const uint32_t maxLodCount =
      std::min(MeshImportOptions::kMaxLodCount, Submesh::kMaxLodCount);
  return std::clamp(options.lodCount, 1u, maxLodCount);
}

size_t sanitizeTargetIndexCount(size_t targetIndexCount,
                                size_t sourceIndexCount) {
  if (sourceIndexCount < kTriangleIndexCount) {
    return 0;
  }
  targetIndexCount =
      std::clamp(targetIndexCount, kTriangleIndexCount, sourceIndexCount);
  targetIndexCount -= targetIndexCount % kTriangleIndexCount;
  return std::max(kTriangleIndexCount, targetIndexCount);
}

size_t targetLodIndexCount(const MeshImportOptions &options, uint32_t lodIndex,
                           size_t sourceIndexCount) {
  return sanitizeTargetIndexCount(
      static_cast<size_t>(static_cast<double>(sourceIndexCount) *
                          lodRatioFor(options, lodIndex)),
      sourceIndexCount);
}

void optimizeIndexOrder(std::span<uint32_t> indices,
                        std::span<const Vertex> vertices) {
  if (indices.empty() || vertices.empty()) {
    return;
  }

  meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(),
                              vertices.size());
  meshopt_optimizeOverdraw(indices.data(), indices.data(), indices.size(),
                           &vertices.front().position.x, vertices.size(),
                           sizeof(Vertex), kMeshoptOverdrawThreshold);
}

void remapMeshVertices(std::pmr::vector<Vertex> &vertices,
                       std::pmr::vector<uint32_t> &indices) {
  if (vertices.empty() || indices.empty()) {
    return;
  }

  std::pmr::memory_resource *mem = vertices.get_allocator().resource();
  std::pmr::vector<unsigned int> remap(mem);
  remap.resize(vertices.size());
  const size_t uniqueVertexCount = meshopt_generateVertexRemap(
      remap.data(), indices.data(), indices.size(), vertices.data(),
      vertices.size(), sizeof(Vertex));
  if (uniqueVertexCount == 0 || uniqueVertexCount > vertices.size()) {
    return;
  }

  std::pmr::vector<uint32_t> remappedIndices(mem);
  remappedIndices.resize(indices.size());
  meshopt_remapIndexBuffer(remappedIndices.data(), indices.data(),
                           indices.size(), remap.data());

  std::pmr::vector<Vertex> remappedVertices(mem);
  remappedVertices.resize(uniqueVertexCount);
  meshopt_remapVertexBuffer(remappedVertices.data(), vertices.data(),
                            vertices.size(), sizeof(Vertex), remap.data());

  indices.swap(remappedIndices);
  vertices.swap(remappedVertices);
}

void extractMeshGeometry(const aiMesh &mesh,
                         std::pmr::vector<Vertex> &outVertices,
                         std::pmr::vector<uint32_t> &outIndices) {
  outVertices.clear();
  outVertices.reserve(mesh.mNumVertices);

  for (unsigned int vertexIndex = 0; vertexIndex < mesh.mNumVertices;
       ++vertexIndex) {
    Vertex vertex{};
    const aiVector3D &pos = mesh.mVertices[vertexIndex];
    vertex.position = {pos.x, pos.y, pos.z};

    if (mesh.HasNormals()) {
      const aiVector3D &normal = mesh.mNormals[vertexIndex];
      vertex.normal = {normal.x, normal.y, normal.z};
    }

    if (mesh.HasTextureCoords(0)) {
      const aiVector3D &uv = mesh.mTextureCoords[0][vertexIndex];
      vertex.uv = {uv.x, uv.y};
    }
    if (mesh.HasTextureCoords(1)) {
      const aiVector3D &uv1 = mesh.mTextureCoords[1][vertexIndex];
      vertex.uv1 = {uv1.x, uv1.y};
    }

    if (mesh.HasTangentsAndBitangents()) {
      const aiVector3D &tangent = mesh.mTangents[vertexIndex];
      const aiVector3D &bitangent = mesh.mBitangents[vertexIndex];
      const glm::vec3 n = vertex.normal;
      const glm::vec3 t = glm::vec3(tangent.x, tangent.y, tangent.z);
      const glm::vec3 b = glm::vec3(bitangent.x, bitangent.y, bitangent.z);
      const float sign = (glm::dot(glm::cross(n, t), b) < 0.0f) ? -1.0f : 1.0f;
      vertex.tangent = {tangent.x, tangent.y, tangent.z, sign};
    }

    outVertices.push_back(vertex);
  }

  outIndices.clear();
  outIndices.reserve(mesh.mNumFaces * kTriangleIndexCount);
  for (unsigned int faceIndex = 0; faceIndex < mesh.mNumFaces; ++faceIndex) {
    const aiFace &face = mesh.mFaces[faceIndex];
    if (face.mNumIndices != kTriangleIndexCount) {
      continue;
    }
    for (unsigned int i = 0; i < kTriangleIndexCount; ++i) {
      outIndices.push_back(face.mIndices[i]);
    }
  }
}

std::array<std::pmr::vector<uint32_t>, Submesh::kMaxLodCount>
makeLodIndexBuffers(std::pmr::memory_resource *mem) {
  static_assert(Submesh::kMaxLodCount == 4,
                "Update makeLodIndexBuffers for new max LOD count");
  return {std::pmr::vector<uint32_t>(mem), std::pmr::vector<uint32_t>(mem),
          std::pmr::vector<uint32_t>(mem), std::pmr::vector<uint32_t>(mem)};
}

uint32_t
buildLodIndexBuffers(const MeshImportOptions &options,
                     uint32_t requestedLodCount, uint32_t meshIndex,
                     std::span<const Vertex> vertices, bool optimize,
                     std::span<std::pmr::vector<uint32_t>> lodIndexBuffers,
                     std::array<float, Submesh::kMaxLodCount> &lodErrors) {
  uint32_t generatedLodCount = 1;

  if (lodIndexBuffers.empty()) {
    return generatedLodCount;
  }

  if (!options.generateLods || requestedLodCount <= 1 ||
      lodIndexBuffers[0].size() < 2 * kTriangleIndexCount) {
    return generatedLodCount;
  }

  NURI_PROFILER_ZONE("MeshImporter.meshopt_lod_generation",
                     NURI_PROFILER_COLOR_CREATE);
  std::pmr::memory_resource *mem =
      lodIndexBuffers[0].get_allocator().resource();
  const size_t baseIndexCount = lodIndexBuffers[0].size();
  for (uint32_t lodIndex = 1; lodIndex < requestedLodCount; ++lodIndex) {
    const size_t targetIndexCount =
        targetLodIndexCount(options, lodIndex, baseIndexCount);
    if (targetIndexCount == 0 || targetIndexCount >= baseIndexCount) {
      continue;
    }

    std::pmr::vector<uint32_t> simplifiedIndices(mem);
    simplifiedIndices.resize(baseIndexCount);
    float lodError = 0.0f;
    size_t simplifiedCount = meshopt_simplify(
        simplifiedIndices.data(), lodIndexBuffers[0].data(), baseIndexCount,
        &vertices.front().position.x, vertices.size(), sizeof(Vertex),
        targetIndexCount, options.lodTargetError, 0, &lodError);
    simplifiedCount -= simplifiedCount % kTriangleIndexCount;
    if (simplifiedCount < kTriangleIndexCount) {
      NURI_LOG_WARNING(
          "MeshImporter::loadFromFile: Mesh %u LOD%u simplification failed, "
          "keeping previous LODs",
          meshIndex, lodIndex);
      break;
    }

    simplifiedIndices.resize(simplifiedCount);
    if (optimize) {
      optimizeIndexOrder(simplifiedIndices, vertices);
    }
    lodIndexBuffers[lodIndex] = std::move(simplifiedIndices);
    lodErrors[lodIndex] = lodError;
    generatedLodCount = lodIndex + 1;
  }
  NURI_PROFILER_ZONE_END();
  return generatedLodCount;
}

void optimizeVertexFetchForAllLods(
    std::pmr::vector<Vertex> &vertices, uint32_t lodCount,
    std::span<std::pmr::vector<uint32_t>> lodIndexBuffers) {
  if (vertices.empty() || lodCount == 0 || lodIndexBuffers.empty() ||
      lodIndexBuffers[0].empty()) {
    return;
  }

  NURI_PROFILER_ZONE("MeshImporter.meshopt_vertex_fetch",
                     NURI_PROFILER_COLOR_CREATE);
  std::pmr::memory_resource *mem = vertices.get_allocator().resource();
  std::pmr::vector<unsigned int> vertexFetchRemap(mem);
  vertexFetchRemap.resize(vertices.size());
  const size_t optimizedVertexCount = meshopt_optimizeVertexFetchRemap(
      vertexFetchRemap.data(), lodIndexBuffers[0].data(),
      lodIndexBuffers[0].size(), vertices.size());
  if (optimizedVertexCount > 0 && optimizedVertexCount <= vertices.size()) {
    std::pmr::vector<Vertex> remappedVertices(mem);
    remappedVertices.resize(optimizedVertexCount);
    meshopt_remapVertexBuffer(remappedVertices.data(), vertices.data(),
                              vertices.size(), sizeof(Vertex),
                              vertexFetchRemap.data());
    vertices.swap(remappedVertices);

    for (uint32_t lodIndex = 0; lodIndex < lodCount; ++lodIndex) {
      std::pmr::vector<uint32_t> remappedIndices(mem);
      remappedIndices.resize(lodIndexBuffers[lodIndex].size());
      meshopt_remapIndexBuffer(
          remappedIndices.data(), lodIndexBuffers[lodIndex].data(),
          lodIndexBuffers[lodIndex].size(), vertexFetchRemap.data());
      lodIndexBuffers[lodIndex].swap(remappedIndices);
    }
  }
  NURI_PROFILER_ZONE_END();
}

void appendSubmeshToMeshData(
    MeshData &data, const aiMesh &mesh, std::span<const Vertex> vertices,
    const BoundingBox &bounds, uint32_t lodCount,
    std::span<const std::pmr::vector<uint32_t>> lodIndexBuffers,
    const std::array<float, Submesh::kMaxLodCount> &lodErrors,
    uint32_t meshIndex) {
  const uint32_t vertexBase = static_cast<uint32_t>(data.vertices.size());
  data.vertices.insert(data.vertices.end(), vertices.begin(), vertices.end());

  Submesh submesh{};
  submesh.materialIndex = mesh.mMaterialIndex;
  submesh.bounds = bounds;
  submesh.lodCount = lodCount;

  for (uint32_t lodIndex = 0; lodIndex < lodCount; ++lodIndex) {
    const uint32_t lodOffset = static_cast<uint32_t>(data.indices.size());
    for (uint32_t localIndex : lodIndexBuffers[lodIndex]) {
      data.indices.push_back(vertexBase + localIndex);
    }

    const uint32_t lodIndexCount =
        static_cast<uint32_t>(data.indices.size() - lodOffset);
    submesh.lods[lodIndex] = SubmeshLod{
        .indexOffset = lodOffset,
        .indexCount = lodIndexCount,
        .error = lodErrors[lodIndex],
    };
    if (lodIndex == 0) {
      submesh.indexOffset = lodOffset;
      submesh.indexCount = lodIndexCount;
    }
  }

  if (submesh.indexCount == 0) {
    NURI_LOG_WARNING("MeshImporter::loadFromFile: Mesh %u LOD0 is empty",
                     meshIndex);
    return;
  }

  data.submeshes.push_back(submesh);
}

unsigned int buildAssimpFlags(const MeshImportOptions &options) {
  // Keep flags: baseline import sanitation for the meshopt pipeline.
  unsigned int flags = aiProcess_SortByPType | aiProcess_FindDegenerates |
                       aiProcess_FindInvalidData;

  if (options.triangulate) {
    flags |= aiProcess_Triangulate;
  }

  if (options.joinIdenticalVertices) {
    flags |= aiProcess_JoinIdenticalVertices;
  }

  if (options.genNormals) {
    flags |= aiProcess_GenSmoothNormals;
  }

  if (options.genTangents) {
    flags |= aiProcess_CalcTangentSpace;
  }

  if (options.flipUVs) {
    flags |= aiProcess_FlipUVs;
  }

  if (options.genUVCoords) {
    flags |= aiProcess_GenUVCoords;
  }

  if (options.removeRedundantMaterials) {
    flags |= aiProcess_RemoveRedundantMaterials;
  }

  if (options.limitBoneWeights) {
    flags |= aiProcess_LimitBoneWeights;
  }

  if (options.optimize) {
    flags |= aiProcess_OptimizeMeshes;
  }

  return flags;
}
} // namespace

nuri::Result<MeshData, std::string>
MeshImporter::loadFromFile(std::string_view path,
                           const MeshImportOptions &options,
                           std::pmr::memory_resource *mem) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (path.empty()) {
    NURI_LOG_WARNING("MeshImporter::loadFromFile: Path is empty");
    return nuri::Result<MeshData, std::string>::makeError("Path is empty");
  }

  if (!mem) {
    mem = std::pmr::get_default_resource();
  }

  Assimp::Importer importer;
  const std::string pathStr(path);
  const unsigned int flags = buildAssimpFlags(options);

  NURI_LOG_DEBUG(
      "MeshImporter::loadFromFile: Importing mesh '%s' with flags %u",
      pathStr.c_str(), flags);
  const aiScene *scene = importer.ReadFile(pathStr, flags);
  if (!scene || !scene->HasMeshes()) {
    const std::string error =
        scene ? "Assimp scene has no meshes" : importer.GetErrorString();
    NURI_LOG_WARNING(
        "MeshImporter::loadFromFile: Failed to import mesh '%s': %s",
        pathStr.c_str(), error.c_str());
    return nuri::Result<MeshData, std::string>::makeError(error);
  }

  NURI_LOG_DEBUG(
      "MeshImporter::loadFromFile: Imported model '%s' with %u meshes",
      pathStr.c_str(), scene->mNumMeshes);

  MeshData data(mem);
  const uint32_t requestedLodCount = clampLodCount(options);

  size_t totalVertices = 0;
  size_t totalIndices = 0;
  for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
    const aiMesh *mesh = scene->mMeshes[i];
    if (!mesh) {
      continue;
    }

    totalVertices += mesh->mNumVertices;
    const size_t baseIndexCount = meshIndexCount(*mesh);
    totalIndices += baseIndexCount;
    if (options.generateLods && requestedLodCount > 1) {
      for (uint32_t lodIndex = 1; lodIndex < requestedLodCount; ++lodIndex) {
        totalIndices += targetLodIndexCount(options, lodIndex, baseIndexCount);
      }
    }
  }

  data.vertices.reserve(totalVertices);
  data.indices.reserve(totalIndices);
  data.submeshes.reserve(scene->mNumMeshes);

  if (scene->mName.length > 0) {
    data.name.assign(scene->mName.C_Str(), scene->mName.length);
  } else {
    const std::string stem = std::filesystem::path(pathStr).stem().string();
    data.name.assign(stem.data(), stem.size());
  }

  ScratchArena scratch(mem);
  size_t insufficientGeometryMeshCount = 0;
  std::array<uint32_t, 8> insufficientGeometryMeshSamples{};
  size_t insufficientGeometrySampleCount = 0;

  NURI_LOG_DEBUG("MeshImporter::loadFromFile: Mesh optimization processing");
  for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
    const aiMesh *mesh = scene->mMeshes[i];
    if (!mesh) {
      continue;
    }

    ScopedScratch scopedScratch(scratch);
    std::pmr::vector<Vertex> meshVertices(scopedScratch.resource());
    std::pmr::vector<uint32_t> lod0Indices(scopedScratch.resource());
    std::array<std::pmr::vector<uint32_t>, Submesh::kMaxLodCount>
        lodIndexBuffers = makeLodIndexBuffers(scopedScratch.resource());
    std::array<float, Submesh::kMaxLodCount> lodErrors{};

    extractMeshGeometry(*mesh, meshVertices, lod0Indices);

    if (meshVertices.empty() || lod0Indices.size() < kTriangleIndexCount) {
      ++insufficientGeometryMeshCount;
      if (insufficientGeometrySampleCount <
          insufficientGeometryMeshSamples.size()) {
        insufficientGeometryMeshSamples[insufficientGeometrySampleCount++] = i;
      }
      continue;
    }

    if (options.optimize) {
      NURI_PROFILER_ZONE("MeshImporter.meshopt_base_optimize",
                         NURI_PROFILER_COLOR_CREATE);
      remapMeshVertices(meshVertices, lod0Indices);
      optimizeIndexOrder(lod0Indices, meshVertices);
      NURI_PROFILER_ZONE_END();
    }

    lodIndexBuffers[0] = std::move(lod0Indices);
    const uint32_t generatedLodCount =
        buildLodIndexBuffers(options, requestedLodCount, i, meshVertices,
                             options.optimize, lodIndexBuffers, lodErrors);

    if (options.optimize) {
      optimizeVertexFetchForAllLods(meshVertices, generatedLodCount,
                                    lodIndexBuffers);
    }

    const BoundingBox submeshBounds = computeSubmeshBounds(meshVertices);
    appendSubmeshToMeshData(data, *mesh, meshVertices, submeshBounds,
                            generatedLodCount, lodIndexBuffers, lodErrors, i);
  }
  NURI_LOG_DEBUG(
      "MeshImporter::loadFromFile: Mesh optimization processing complete");

  if (insufficientGeometryMeshCount > 0) {
    std::ostringstream sampleStream;
    for (size_t sampleIndex = 0; sampleIndex < insufficientGeometrySampleCount;
         ++sampleIndex) {
      if (sampleIndex > 0) {
        sampleStream << ", ";
      }
      sampleStream << insufficientGeometryMeshSamples[sampleIndex];
    }
    NURI_LOG_WARNING(
        "MeshImporter::loadFromFile: skipped %zu mesh(es) with insufficient "
        "triangle geometry (sample indices: %s)",
        insufficientGeometryMeshCount, sampleStream.str().c_str());
  }

  return nuri::Result<MeshData, std::string>::makeResult(std::move(data));
}

nuri::Result<ImportedMaterialSet, std::string>
MeshImporter::loadMaterialInfoFromFile(std::string_view path) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (path.empty()) {
    return nuri::Result<ImportedMaterialSet, std::string>::makeError(
        "MeshImporter::loadMaterialInfoFromFile: path is empty");
  }

  Assimp::Importer importer;
  const std::string pathStr(path);
  constexpr unsigned int kMaterialOnlyFlags =
      aiProcess_SortByPType | aiProcess_FindInvalidData;
  const aiScene *scene = importer.ReadFile(pathStr, kMaterialOnlyFlags);
  if (!scene) {
    return nuri::Result<ImportedMaterialSet, std::string>::makeError(
        std::string("MeshImporter::loadMaterialInfoFromFile: Assimp error: ") +
        importer.GetErrorString());
  }

  ImportedMaterialSet set{};
  const std::filesystem::path modelPath(pathStr);
  if (!scene->HasMaterials()) {
    NURI_LOG_WARNING("MeshImporter::loadMaterialInfoFromFile: scene '%s' has "
                     "no materials",
                     pathStr.c_str());
    return nuri::Result<ImportedMaterialSet, std::string>::makeResult(
        std::move(set));
  }

  set.materials.reserve(scene->mNumMaterials);
  for (uint32_t materialIndex = 0; materialIndex < scene->mNumMaterials;
       ++materialIndex) {
    const aiMaterial *material = scene->mMaterials[materialIndex];
    if (!material) {
      ImportedMaterialInfo fallback{};
      fallback.name = "material_" + std::to_string(materialIndex);
      set.materials.push_back(std::move(fallback));
      continue;
    }

    ImportedMaterialInfo parsed = parseMaterial(*material, modelPath);
    if (parsed.name.empty()) {
      parsed.name = "material_" + std::to_string(materialIndex);
    }
    set.materials.push_back(std::move(parsed));
  }

  NURI_LOG_DEBUG("MeshImporter::loadMaterialInfoFromFile: extracted %zu "
                 "material(s) from '%s'",
                 set.materials.size(), pathStr.c_str());
  return nuri::Result<ImportedMaterialSet, std::string>::makeResult(
      std::move(set));
}

} // namespace nuri
