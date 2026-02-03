#include "mesh_importer.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace nuri {
namespace {
unsigned int buildAssimpFlags(const MeshImportOptions &options) {
  unsigned int flags = 0;
  if (options.triangulate) {
    flags |= aiProcess_Triangulate;
  }

  if (options.joinIdenticalVertices) {
    flags |= aiProcess_JoinIdenticalVertices;
  }

  if (options.genNormals) {
    flags |= aiProcess_GenNormals;
  }

  if (options.genTangents) {
    flags |= aiProcess_CalcTangentSpace;
  }

  if (options.flipUVs) {
    flags |= aiProcess_FlipUVs;
  }

  if (options.optimize) {
    flags |= aiProcess_ImproveCacheLocality | aiProcess_OptimizeMeshes;
  }

  flags |= aiProcess_SortByPType;
  return flags;
}
} // namespace

nuri::Result<MeshData, std::string>
MeshImporter::loadFromFile(std::string_view path,
                           const MeshImportOptions &options,
                           std::pmr::memory_resource *mem) {
  if (path.empty()) {
    return nuri::Result<MeshData, std::string>::makeError("Path is empty");
  }

  if (!mem) {
    mem = std::pmr::get_default_resource();
  }

  Assimp::Importer importer;
  const std::string pathStr(path);
  const unsigned int flags = buildAssimpFlags(options);
  const aiScene *scene = importer.ReadFile(pathStr, flags);
  if (!scene || !scene->HasMeshes()) {
    const std::string error =
        scene ? "Assimp scene has no meshes" : importer.GetErrorString();
    return nuri::Result<MeshData, std::string>::makeError(error);
  }

  MeshData data(mem);

  size_t totalVertices = 0;
  size_t totalIndices = 0;
  for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
    const aiMesh *mesh = scene->mMeshes[i];
    if (!mesh) {
      continue;
    }

    totalVertices += mesh->mNumVertices;
    for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
      totalIndices += mesh->mFaces[f].mNumIndices;
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

  for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
    const aiMesh *mesh = scene->mMeshes[i];
    if (!mesh) {
      continue;
    }

    const uint32_t vertexBase = static_cast<uint32_t>(data.vertices.size());
    for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
      Vertex vertex{};
      const aiVector3D &pos = mesh->mVertices[v];
      vertex.position = {pos.x, pos.y, pos.z};

      if (mesh->HasNormals()) {
        const aiVector3D &normal = mesh->mNormals[v];
        vertex.normal = {normal.x, normal.y, normal.z};
      }

      if (mesh->HasTextureCoords(0)) {
        const aiVector3D &uv = mesh->mTextureCoords[0][v];
        vertex.uv = {uv.x, uv.y};
      }

      if (mesh->HasTangentsAndBitangents()) {
        const aiVector3D &tangent = mesh->mTangents[v];
        vertex.tangent = {tangent.x, tangent.y, tangent.z, 1.0f};
      }

      data.vertices.push_back(vertex);
    }

    const uint32_t indexOffset = static_cast<uint32_t>(data.indices.size());
    for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
      const aiFace &face = mesh->mFaces[f];
      for (unsigned int j = 0; j < face.mNumIndices; ++j) {
        data.indices.push_back(vertexBase + face.mIndices[j]);
      }
    }

    const uint32_t indexCount =
        static_cast<uint32_t>(data.indices.size() - indexOffset);
    data.submeshes.push_back(Submesh{
        .indexOffset = indexOffset,
        .indexCount = indexCount,
        .materialIndex = mesh->mMaterialIndex,
    });
  }

  return nuri::Result<MeshData, std::string>::makeResult(std::move(data));
}

} // namespace nuri
