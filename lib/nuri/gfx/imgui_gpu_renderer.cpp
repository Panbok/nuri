#include "nuri/gfx/imgui_gpu_renderer.h"

#include "nuri/pch.h"

#include "nuri/core/log.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"

namespace nuri {

namespace {

// ImGui provides ImDrawIdx as 16-bit by default
static_assert(sizeof(ImDrawIdx) == 2, "This renderer assumes 16-bit indices");

constexpr std::string_view kImGuiVS = R"(
#version 460

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec4 outColor;

layout(push_constant) uniform PushConstants {
  vec4 lrtb; // left, right, top, bottom
  uint textureId;
} pc;

void main() {
  float L = pc.lrtb.x;
  float R = pc.lrtb.y;
  float T = pc.lrtb.z;
  float B = pc.lrtb.w;

  mat4 proj = mat4(
    2.0 / (R - L),                   0.0,  0.0, 0.0,
    0.0,                   2.0 / (T - B),  0.0, 0.0,
    0.0,                             0.0, -1.0, 0.0,
    (R + L) / (L - R), (T + B) / (B - T),  0.0, 1.0);

  outUV = inUV;
  outColor = inColor;
  gl_Position = proj * vec4(inPos.xy, 0.0, 1.0);
}
)";

constexpr std::string_view kImGuiFS = R"(
#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(set = 0, binding = 1) uniform sampler kSamplers[];

vec4 textureBindless2D(uint textureid, uint samplerid, vec2 uv) {
  return texture(
      nonuniformEXT(sampler2D(kTextures2D[textureid], kSamplers[samplerid])),
      uv);
}

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
  vec4 lrtb; // left, right, top, bottom
  uint textureId;
} pc;

void main() {
  // Sampler index 0 is assumed to exist in the bindless sampler array.
  outColor = inColor * textureBindless2D(pc.textureId, 0, inUV);
}
)";

inline ImTextureID toImTextureID(uint32_t id) {
  if constexpr (std::is_pointer_v<ImTextureID>) {
    return reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(id));
  } else {
    return static_cast<ImTextureID>(static_cast<uintptr_t>(id));
  }
}

inline uint32_t fromImTextureID(ImTextureID id) {
  uintptr_t raw = 0;
  if constexpr (std::is_pointer_v<ImTextureID>) {
    raw = reinterpret_cast<uintptr_t>(id);
  } else {
    raw = static_cast<uintptr_t>(id);
  }
  if (raw > static_cast<uintptr_t>(UINT32_MAX)) {
    NURI_LOG_WARNING("fromImTextureID: ImTextureID value 0x%" PRIxPTR
                     " exceeds UINT32_MAX",
                     raw);
    NURI_ASSERT(raw <= static_cast<uintptr_t>(UINT32_MAX),
                "ImTextureID must fit in uint32_t");
    return 0;
  }
  return static_cast<uint32_t>(raw);
}

} // namespace

std::unique_ptr<ImGuiGpuRenderer> ImGuiGpuRenderer::create(GPUDevice &gpu) {
  return std::unique_ptr<ImGuiGpuRenderer>(new ImGuiGpuRenderer(gpu));
}

ImGuiGpuRenderer::ImGuiGpuRenderer(GPUDevice &gpu) : gpu_(gpu) {
  NURI_LOG_DEBUG(
      "ImGuiGpuRenderer::ImGuiGpuRenderer: ImGui GPU renderer created");
}

Result<bool, std::string> ImGuiGpuRenderer::ensureFontTexture() {
  if (nuri::isValid(fontTexture_) && fontTextureId_ != 0) {
    return Result<bool, std::string>::makeResult(true);
  }

  ImGuiIO &io = ImGui::GetIO();
  unsigned char *pixels = nullptr;
  int width = 0;
  int height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  if (!pixels || width <= 0 || height <= 0) {
    return Result<bool, std::string>::makeError("ImGui font atlas is empty");
  }

  const size_t dataSize =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
  const std::span<const std::byte> data{
      reinterpret_cast<const std::byte *>(pixels), dataSize};

  if (nuri::isValid(fontTexture_)) {
    gpu_.destroyTexture(fontTexture_);
    fontTexture_ = TextureHandle{};
    fontTextureId_ = 0;
  }

  TextureDesc desc{
      .type = TextureType::Texture2D,
      .format = Format::RGBA8_UNORM,
      .dimensions = {static_cast<uint32_t>(width),
                     static_cast<uint32_t>(height), 1},
      .usage = TextureUsage::Sampled,
      .storage = Storage::Device,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .data = data,
      .dataNumMipLevels = 1,
      .generateMipmaps = false,
  };

  auto texResult = gpu_.createTexture(desc, "ImGui Font Atlas");
  if (texResult.hasError()) {
    return Result<bool, std::string>::makeError(texResult.error());
  }
  fontTexture_ = texResult.value();
  fontTextureId_ = gpu_.getTextureBindlessIndex(fontTexture_);
  if (fontTextureId_ == 0) {
    return Result<bool, std::string>::makeError(
        "Failed to get bindless index for ImGui font texture");
  }

  io.Fonts->SetTexID(toImTextureID(fontTextureId_));
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ImGuiGpuRenderer::ensurePipeline(Format swapchainFormat) {
  if (nuri::isValid(pipeline_) && pipelineFormat_ == swapchainFormat) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (nuri::isValid(vs_)) {
    gpu_.destroyShaderModule(vs_);
    vs_ = ShaderHandle{};
  }
  if (nuri::isValid(fs_)) {
    gpu_.destroyShaderModule(fs_);
    fs_ = ShaderHandle{};
  }

  ShaderDesc vsDesc{
      .moduleName = "imgui_vs",
      .source = kImGuiVS,
      .stage = ShaderStage::Vertex,
  };
  auto vsResult = gpu_.createShaderModule(vsDesc);
  if (vsResult.hasError()) {
    return Result<bool, std::string>::makeError(vsResult.error());
  }

  ShaderDesc fsDesc{
      .moduleName = "imgui_fs",
      .source = kImGuiFS,
      .stage = ShaderStage::Fragment,
  };
  auto fsResult = gpu_.createShaderModule(fsDesc);
  if (fsResult.hasError()) {
    gpu_.destroyShaderModule(vsResult.value());
    return Result<bool, std::string>::makeError(fsResult.error());
  }

  vs_ = vsResult.value();
  fs_ = fsResult.value();

  const VertexBinding bindings[] = {
      {.stride = static_cast<uint32_t>(sizeof(ImDrawVert))},
  };
  const VertexAttribute attrs[] = {
      {.location = 0,
       .binding = 0,
       .offset = static_cast<uint32_t>(IM_OFFSETOF(ImDrawVert, pos)),
       .format = VertexFormat::Float2},
      {.location = 1,
       .binding = 0,
       .offset = static_cast<uint32_t>(IM_OFFSETOF(ImDrawVert, uv)),
       .format = VertexFormat::Float2},
      {.location = 2,
       .binding = 0,
       .offset = static_cast<uint32_t>(IM_OFFSETOF(ImDrawVert, col)),
       .format = VertexFormat::UByte4_Norm},
  };

  RenderPipelineDesc pipelineDesc{
      .vertexInput =
          {
              .attributes = attrs,
              .bindings = bindings,
          },
      .vertexShader = vs_,
      .fragmentShader = fs_,
      .colorFormats = {swapchainFormat},
      .depthFormat = Format::Count,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = true,
  };

  auto pipeResult = gpu_.createRenderPipeline(pipelineDesc, "ImGui Pipeline");
  if (pipeResult.hasError()) {
    gpu_.destroyShaderModule(vs_);
    gpu_.destroyShaderModule(fs_);
    vs_ = ShaderHandle{};
    fs_ = ShaderHandle{};
    return Result<bool, std::string>::makeError(pipeResult.error());
  }
  pipeline_ = pipeResult.value();
  pipelineFormat_ = swapchainFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ImGuiGpuRenderer::ensureBuffers(uint64_t frameIndex,
                                                          size_t vertexBytes,
                                                          size_t indexBytes) {
  if (frameIndex >= static_cast<uint64_t>(frames_.size())) {
    return Result<bool, std::string>::makeError("Invalid ImGui frame index");
  }

  FrameBuffers &fb = frames_[static_cast<size_t>(frameIndex)];

  if (!nuri::isValid(fb.vb) || fb.vbCapacityBytes < vertexBytes) {
    if (nuri::isValid(fb.vb)) {
      gpu_.destroyBuffer(fb.vb);
      fb.vb = BufferHandle{};
    }
    const size_t newSize =
        std::max({vertexBytes, fb.vbCapacityBytes * 2, size_t{1}});
    BufferDesc desc{
        .usage = BufferUsage::Vertex,
        .storage = Storage::HostVisible,
        .size = newSize,
    };
    auto vbResult = gpu_.createBuffer(desc, "ImGui VB");
    if (vbResult.hasError()) {
      return Result<bool, std::string>::makeError(vbResult.error());
    }
    fb.vb = vbResult.value();
    fb.vbCapacityBytes = newSize;
  }

  if (!nuri::isValid(fb.ib) || fb.ibCapacityBytes < indexBytes) {
    if (nuri::isValid(fb.ib)) {
      gpu_.destroyBuffer(fb.ib);
      fb.ib = BufferHandle{};
    }
    const size_t newSize =
        std::max({indexBytes, fb.ibCapacityBytes * 2, size_t{1}});
    BufferDesc desc{
        .usage = BufferUsage::Index,
        .storage = Storage::HostVisible,
        .size = newSize,
    };
    auto ibResult = gpu_.createBuffer(desc, "ImGui IB");
    if (ibResult.hasError()) {
      return Result<bool, std::string>::makeError(ibResult.error());
    }
    fb.ib = ibResult.value();
    fb.ibCapacityBytes = newSize;
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<RenderPass, std::string>
ImGuiGpuRenderer::buildRenderPass(Format swapchainFormat, uint64_t frameIndex) {
  ImDrawData *dd = ImGui::GetDrawData();
  if (!dd) {
    return Result<RenderPass, std::string>::makeError(
        "ImGui draw data is null");
  }

  const float fbWidth = dd->DisplaySize.x * dd->FramebufferScale.x;
  const float fbHeight = dd->DisplaySize.y * dd->FramebufferScale.y;
  if (fbWidth <= 0.0f || fbHeight <= 0.0f || dd->CmdListsCount == 0) {
    RenderPass empty{};
    empty.color.loadOp = LoadOp::Load;
    return Result<RenderPass, std::string>::makeResult(empty);
  }

  auto pipelineOk = ensurePipeline(swapchainFormat);
  if (pipelineOk.hasError()) {
    return Result<RenderPass, std::string>::makeError(pipelineOk.error());
  }
  auto fontOk = ensureFontTexture();
  if (fontOk.hasError()) {
    return Result<RenderPass, std::string>::makeError(fontOk.error());
  }

  const uint32_t imageCount = std::max(1u, gpu_.getSwapchainImageCount());
  if (frames_.size() != imageCount) {
    frames_.assign(imageCount, FrameBuffers{});
  }
  const uint64_t frameSlot = frameIndex % static_cast<uint64_t>(imageCount);
  const FrameBuffers &fb = frames_[static_cast<size_t>(frameSlot)];

  const size_t vtxBytes =
      static_cast<size_t>(dd->TotalVtxCount) * sizeof(ImDrawVert);
  const size_t idxBytes =
      static_cast<size_t>(dd->TotalIdxCount) * sizeof(ImDrawIdx);
  auto bufOk = ensureBuffers(frameSlot, vtxBytes, idxBytes);
  if (bufOk.hasError()) {
    return Result<RenderPass, std::string>::makeError(bufOk.error());
  }

  // Upload combined vertex/index data.
  {
    ScopedScratch scopedScratch(scratch_);
    std::pmr::vector<std::byte> vtxData(scopedScratch.resource());
    std::pmr::vector<std::byte> idxData(scopedScratch.resource());
    vtxData.resize(vtxBytes);
    idxData.resize(idxBytes);

    std::byte *vtxDst = vtxData.data();
    std::byte *idxDst = idxData.data();
    for (const ImDrawList *cmdList : dd->CmdLists) {
      const size_t vbSize =
          static_cast<size_t>(cmdList->VtxBuffer.Size) * sizeof(ImDrawVert);
      const size_t ibSize =
          static_cast<size_t>(cmdList->IdxBuffer.Size) * sizeof(ImDrawIdx);
      std::memcpy(vtxDst, cmdList->VtxBuffer.Data, vbSize);
      std::memcpy(idxDst, cmdList->IdxBuffer.Data, ibSize);
      vtxDst += vbSize;
      idxDst += ibSize;
    }

    auto upVb =
        gpu_.updateBuffer(fb.vb, std::span<const std::byte>(vtxData), 0);
    if (upVb.hasError()) {
      return Result<RenderPass, std::string>::makeError(upVb.error());
    }
    auto upIb =
        gpu_.updateBuffer(fb.ib, std::span<const std::byte>(idxData), 0);
    if (upIb.hasError()) {
      return Result<RenderPass, std::string>::makeError(upIb.error());
    }
  }

  draws_.clear();
  pushConstants_.clear();

  // We create one DrawItem per ImDrawCmd.
  size_t totalCmdCount = 0;
  for (const ImDrawList *cmdList : dd->CmdLists) {
    totalCmdCount += static_cast<size_t>(cmdList->CmdBuffer.Size);
  }
  draws_.reserve(totalCmdCount);
  pushConstants_.reserve(totalCmdCount);

  const float L = dd->DisplayPos.x;
  const float R = dd->DisplayPos.x + dd->DisplaySize.x;
  const float T = dd->DisplayPos.y;
  const float B = dd->DisplayPos.y + dd->DisplaySize.y;

  const ImVec2 clipOff = dd->DisplayPos;
  const ImVec2 clipScale = dd->FramebufferScale;

  uint32_t globalIdxOffset = 0;
  uint32_t globalVtxOffset = 0;

  for (const ImDrawList *cmdList : dd->CmdLists) {
    for (int cmd_i = 0; cmd_i < cmdList->CmdBuffer.Size; ++cmd_i) {
      const ImDrawCmd &cmd = cmdList->CmdBuffer[cmd_i];
      if (cmd.UserCallback != nullptr) {
        // Not supported (editor UI shouldn't rely on callbacks for now).
        continue;
      }

      ImVec2 clipMin((cmd.ClipRect.x - clipOff.x) * clipScale.x,
                     (cmd.ClipRect.y - clipOff.y) * clipScale.y);
      ImVec2 clipMax((cmd.ClipRect.z - clipOff.x) * clipScale.x,
                     (cmd.ClipRect.w - clipOff.y) * clipScale.y);

      if (clipMin.x < 0.0f)
        clipMin.x = 0.0f;
      if (clipMin.y < 0.0f)
        clipMin.y = 0.0f;
      if (clipMax.x > fbWidth)
        clipMax.x = fbWidth;
      if (clipMax.y > fbHeight)
        clipMax.y = fbHeight;
      if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y) {
        continue;
      }

      PushConstants pc{};
      pc.lrtb[0] = L;
      pc.lrtb[1] = R;
      pc.lrtb[2] = T;
      pc.lrtb[3] = B;
      pc.textureId = fromImTextureID(cmd.GetTexID());

      pushConstants_.push_back(pc);
      const PushConstants &storedPc = pushConstants_.back();
      const std::span<const std::byte> pcBytes{
          reinterpret_cast<const std::byte *>(&storedPc), sizeof(storedPc)};

      DrawItem di{};
      di.pipeline = pipeline_;
      di.vertexBuffer = fb.vb;
      di.vertexBufferOffset = 0;
      di.indexBuffer = fb.ib;
      di.indexBufferOffset = 0;
      di.indexFormat = IndexFormat::U16;
      di.indexCount = cmd.ElemCount;
      di.instanceCount = 1;
      di.firstIndex = globalIdxOffset + cmd.IdxOffset;
      di.vertexOffset = static_cast<int32_t>(globalVtxOffset + cmd.VtxOffset);
      di.firstInstance = 0;
      di.useScissor = true;
      di.scissor = {
          static_cast<uint32_t>(clipMin.x),
          static_cast<uint32_t>(clipMin.y),
          static_cast<uint32_t>(clipMax.x - clipMin.x),
          static_cast<uint32_t>(clipMax.y - clipMin.y),
      };
      di.pushConstants = pcBytes;
      di.debugLabel = "ImGui DrawCmd";
      di.debugColor = 0xff00ff00u;

      draws_.push_back(di);
    }

    globalIdxOffset += static_cast<uint32_t>(cmdList->IdxBuffer.Size);
    globalVtxOffset += static_cast<uint32_t>(cmdList->VtxBuffer.Size);
  }

  RenderPass pass{};
  pass.color.loadOp = LoadOp::Load;
  pass.color.storeOp = StoreOp::Store;
  pass.useViewport = true;
  pass.viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = fbWidth,
      .height = fbHeight,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  pass.draws = std::span<const DrawItem>(draws_.data(), draws_.size());
  pass.debugLabel = "ImGui Pass";
  pass.debugColor = 0xff00ff00u;

  return Result<RenderPass, std::string>::makeResult(pass);
}

} // namespace nuri
