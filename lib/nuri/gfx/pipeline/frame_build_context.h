#pragma once

#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/render_graph/render_graph.h"

namespace nuri {

class ResourceManager;

struct FrameBuildContext {
  RenderFrameContext &frame;
  RenderGraphBuilder &graph;
  const ResourceManager &resources;
  FrameSharedResources &shared;
};

[[nodiscard]] inline Result<FrameBuildContext, std::string>
makeFrameBuildContext(RenderFrameContext &frame, RenderGraphBuilder &graph,
                      std::string_view callerName) {
  if (frame.resources == nullptr) {
    return Result<FrameBuildContext, std::string>::makeError(
        std::string(callerName) + ": frame resources are null");
  }

  return Result<FrameBuildContext, std::string>::makeResult(FrameBuildContext{
      .frame = frame,
      .graph = graph,
      .resources = *frame.resources,
      .shared = frame.sharedResources,
  });
}

} // namespace nuri
