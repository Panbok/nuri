#pragma once

#include "nuri/core/layer.h"
#include "nuri/text/text_system.h"

#include <memory>

namespace nuri {

class NURI_API TextLayer2D final : public Layer {
public:
  struct CreateDesc {
    TextSystem &text;
  };

  static std::unique_ptr<TextLayer2D> create(const CreateDesc &desc);
  ~TextLayer2D() override = default;

  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  explicit TextLayer2D(const CreateDesc &desc);

  TextSystem &text_;
};

} // namespace nuri
