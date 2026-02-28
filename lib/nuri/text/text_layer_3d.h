#pragma once

#include "nuri/core/layer.h"
#include "nuri/text/text_system.h"

#include <memory>

namespace nuri {

class NURI_API TextLayer3D final : public Layer {
public:
  struct CreateDesc {
    TextSystem &text;
  };

  static std::unique_ptr<TextLayer3D> create(const CreateDesc &desc);
  ~TextLayer3D() override = default;

  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  explicit TextLayer3D(const CreateDesc &desc);

  TextSystem &text_;
};

} // namespace nuri
